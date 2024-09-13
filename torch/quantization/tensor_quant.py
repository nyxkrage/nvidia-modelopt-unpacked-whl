# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Basic tensor quantization functions."""

import torch
import torch._C._onnx as _C_onnx
from packaging.version import Version
from torch.autograd import Function
from torch.onnx import symbolic_helper

from modelopt.torch.quantization.utils import is_torch_library_supported

from .config import QuantizerAttributeConfig
from .extensions import get_cuda_ext, get_cuda_ext_fp8

onnx_dtype_map = {
    "Float": _C_onnx.TensorProtoDataType.FLOAT,
    "Half": _C_onnx.TensorProtoDataType.FLOAT16,
    "BFloat16": _C_onnx.TensorProtoDataType.BFLOAT16,
}

torch_dtype_map = {"Float": torch.float32, "Half": torch.float16, "BFloat16": torch.bfloat16}


def scaled_e4m3_impl(
    inputs: torch.Tensor,  # TODO: check support for multiple inputs
    amax: torch.Tensor,
    disable_fused_kernel=False,
) -> torch.Tensor:
    """Implementation of fake quantizing input to FP8.

    Args:
        inputs: Torch tensor.
        amax: Absolute max range of the input tensor.

    Returns:
        Input tensors faked quantized to FP8.
    """
    cuda_ext_fp8 = get_cuda_ext_fp8()

    assert (
        cuda_ext_fp8 is not None
    ), "cuda_ext_fp8 could not be imported. E4M3 quantization requires CUDA and cuda_ext_fp8."

    def is_fusable():
        # ignore no scaling and shape([]) cases
        if amax is None or len(amax.shape) == 0:
            return False
        else:
            # can't have amax.shape = [1, 1, 4, 1] and the like
            amax_last_dim_only = amax.numel() == amax.shape[-1]
            # must be cuda
            all_cuda = inputs.is_cuda and amax.is_cuda

            # also check explicit disable.
            return amax_last_dim_only and all_cuda and (not disable_fused_kernel)

    with torch.cuda.device(
        None if inputs.device.index == torch.cuda.current_device() else inputs.device.index
    ):
        # differentiate between fused & unfused cases
        if is_fusable():
            zero_threshold = 1.0 / (1 << 24)
            outputs = cuda_ext_fp8.fused_fake_e4m3fy(inputs, amax.float(), zero_threshold)
        else:
            zero_mask = inputs.abs() < 1.0 / (1 << 24)

            if amax is None:
                outputs = cuda_ext_fp8.fake_e4m3fy(inputs)
            else:
                scale = 448.0 / amax
                outputs = cuda_ext_fp8.fake_e4m3fy(inputs * scale) / scale

            # Zero out values that are tiny.
            # Tiny values could lead to tiny amax and then large scale which cause overflow/saturation
            # and won't go back to normal value after dividing by scale. The right behavior is to mark them
            # as zero which also get rid of inf/nan
            outputs[zero_mask] = 0.0

        return outputs


def fake_quant_impl(
    inputs: torch.Tensor,
    amax: torch.Tensor,
    num_bits=8,
    unsigned=False,
    narrow_range=True,
):
    """Implementation of fake quantizing input according to number of bits."""
    cuda_ext = get_cuda_ext()

    with torch.cuda.device(
        None if inputs.device.index == torch.cuda.current_device() else inputs.device.index
    ):
        if amax.numel() == 1:
            outputs = cuda_ext.fake_tensor_quant(inputs, amax, num_bits, unsigned, narrow_range)
        else:
            axis = amax.shape.index(amax.numel())
            outputs = cuda_ext.fake_tensor_quant_with_axis(
                inputs, amax.squeeze(), axis, num_bits, unsigned, narrow_range
            )
        return outputs


def _quantize_impl(
    inputs: torch.Tensor,
    amax: torch.Tensor,
    num_bits: int = 8,
    exponent_bits: int = 0,
    unsigned: bool = False,
    narrow_range: bool = True,
):
    if num_bits == 8 and exponent_bits == 4:
        return scaled_e4m3_impl(inputs=inputs, amax=amax)
    elif isinstance(num_bits, int):
        return fake_quant_impl(
            inputs=inputs,
            amax=amax,
            num_bits=num_bits,
            unsigned=unsigned,
            narrow_range=narrow_range,
        )
    else:
        raise ValueError(
            f"Invalid combination of (num_bits, exponent_bits): ({num_bits}, {exponent_bits})."
        )


def _quantize_impl_abstract(
    input: torch.Tensor,
    amax: torch.Tensor,
    num_bits: int = 8,
    exponent_bits: int = 0,
    unsigned: bool = False,
    narrow_range: bool = True,
) -> torch.Tensor:
    """Register an abstract implementation for quantizing tensor.

    This abstract function returns an empty tensor with the same shape and dtype.
    """
    output = torch.empty_like(input)

    return output


quantize_op = _quantize_impl
# Define torch.library custom op if supported
if is_torch_library_supported():
    try:
        torch.library.define(
            "tensorrt::quantize_op",
            "(Tensor input, Tensor amax, int num_bits, int exponent_bits, "
            "bool unsigned, bool narrow_range) -> Tensor",
        )
        quantize_op_impl = torch.library.impl("tensorrt::quantize_op", ["cpu", "cuda"])(
            _quantize_impl
        )
        if Version(torch.__version__) < Version("2.4.0"):
            quantize_op_abstract = torch.library.impl_abstract("tensorrt::quantize_op")(
                _quantize_impl_abstract
            )
        else:
            quantize_op_abstract = torch.library.register_fake("tensorrt::quantize_op")(
                _quantize_impl_abstract
            )
        quantize_op = torch.ops.tensorrt.quantize_op
    except (AttributeError, RuntimeError):
        # torch.library is an experiemental feature, the function signatures may change overtime.
        print(
            "Unable to register operators with torch.library. Exporting quantized models with"
            " torch.export will not be supported."
        )

# Predefined descriptors
QUANT_DESC_8BIT_PER_TENSOR = QuantizerAttributeConfig(num_bits=8)
QUANT_DESC_UNSIGNED_8BIT_PER_TENSOR = QuantizerAttributeConfig(num_bits=8, unsigned=True)
QUANT_DESC_8BIT_CONV1D_WEIGHT_PER_CHANNEL = QuantizerAttributeConfig(num_bits=8, axis=(0))
QUANT_DESC_8BIT_CONV2D_WEIGHT_PER_CHANNEL = QuantizerAttributeConfig(num_bits=8, axis=(0))
QUANT_DESC_8BIT_CONV3D_WEIGHT_PER_CHANNEL = QuantizerAttributeConfig(num_bits=8, axis=(0))
QUANT_DESC_8BIT_LINEAR_WEIGHT_PER_ROW = QuantizerAttributeConfig(num_bits=8, axis=(0))
QUANT_DESC_8BIT_CONVTRANSPOSE1D_WEIGHT_PER_CHANNEL = QuantizerAttributeConfig(num_bits=8, axis=(0))
QUANT_DESC_8BIT_CONVTRANSPOSE2D_WEIGHT_PER_CHANNEL = QuantizerAttributeConfig(num_bits=8, axis=(0))
QUANT_DESC_8BIT_CONVTRANSPOSE3D_WEIGHT_PER_CHANNEL = QuantizerAttributeConfig(num_bits=8, axis=(0))


@torch.jit.script
def _fake_tensor_quant_backward(inputs, amax, grad_outputs):
    zero = grad_outputs.new_zeros(1)
    grad_inputs = torch.where(inputs.abs() <= amax, grad_outputs, zero)
    return grad_inputs


def _onnx_int8_helper(g, inputs, amax, num_bits, unsigned, narrow_range, trt_high_precision_dtype):
    assert num_bits == 8, "Only INT8 ONNX export is supported for now."
    output_shape = torch.onnx.symbolic_helper._get_tensor_sizes(inputs)
    maxbound = (1 << (num_bits - 1 + int(unsigned))) - 1

    if amax.numel() == 1:
        zero_point, axis = torch.tensor(0.0, device=amax.device), None
    else:
        amax_init_shape = amax.shape
        amax = amax.squeeze().data
        assert len(amax.shape) == 1, "ONNX does not support multi-axis quantization."
        zero_point = torch.zeros_like(amax, dtype=torch.int32).data
        axis = list(amax_init_shape).index(list(amax.shape)[0])

    zero_point = g.op("Constant", value_t=zero_point.to(torch_dtype_map[trt_high_precision_dtype]))

    if not unsigned:
        assert not narrow_range, "ONNX does not support unsigned narrow range INT8."
        zero_point = g.op("Cast", zero_point, to_i=_C_onnx.TensorProtoDataType.INT8)
    else:
        zero_point = g.op("Cast", zero_point, to_i=_C_onnx.TensorProtoDataType.UINT8)

    amax = amax.to(torch_dtype_map[trt_high_precision_dtype])
    scale = amax / maxbound
    scale.masked_fill_(scale == 0, 1.0)
    scale = g.op("Constant", value_t=scale)

    input_type = inputs.type().scalarType()

    assert (
        trt_high_precision_dtype == input_type or trt_high_precision_dtype == "Float"
    ), "TRT StronglyType requires both weights and amax to be in the BF16/FP16, or the QDQ in Float."

    # custom ops, so cast the input if needed.
    if trt_high_precision_dtype != input_type:
        inputs = g.op("Cast", inputs, to_i=onnx_dtype_map[trt_high_precision_dtype])
    quantized = g.op("QuantizeLinear", inputs, scale, zero_point, axis_i=axis)
    out = g.op("DequantizeLinear", quantized, scale, zero_point, axis_i=axis).setType(
        inputs.type().with_dtype(torch_dtype_map[trt_high_precision_dtype]).with_sizes(output_shape)
    )

    # custom ops, so cast the output if needed.
    if trt_high_precision_dtype != input_type:
        inputs = g.op("Cast", inputs, to_i=onnx_dtype_map[input_type])

    return out


class FakeTensorQuantFunction(Function):
    """Fake version of TensorQuantFunction use CUDA extension."""

    @staticmethod
    @symbolic_helper.parse_args("v", "t", "i", "b", "b", "s")
    def symbolic(
        g,
        inputs,
        amax,
        num_bits=8,
        unsigned=False,
        narrow_range=True,
        trt_high_precision_dtype="Float",
    ):
        """ONNX symbolic function."""
        return _onnx_int8_helper(
            g, inputs, amax, num_bits, unsigned, narrow_range, trt_high_precision_dtype
        )

    @staticmethod
    def forward(
        ctx,
        inputs,
        amax,
        num_bits=8,
        unsigned=False,
        narrow_range=True,
        trt_high_precision_dtype="Float",
    ):
        """Forward method."""
        ctx.save_for_backward(inputs, amax)

        def legacy_quant_func():
            # The LegacyFakeTensorQuantFunction support cpu and amax with any shape that can be broadcasted to inputs.
            outputs, scale = _tensor_quant(inputs, amax, num_bits, unsigned, narrow_range)
            return outputs / scale.to(inputs.dtype)

        if not inputs.is_cuda:
            outputs = legacy_quant_func()
        else:
            try:
                outputs = quantize_op(
                    inputs,
                    amax,
                    num_bits=num_bits,
                    exponent_bits=0,
                    unsigned=unsigned,
                    narrow_range=narrow_range,
                )
            except (AttributeError, ValueError):
                # AttributeError: cuda_ext is not imported, possibly due to CPU only installation
                # ValueError: cuda_ext is installed, but trying to perform multidimensional quantization (amax dim > 1)
                outputs = legacy_quant_func()

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        """Implements straight through estimation with clipping."""
        inputs, amax = ctx.saved_tensors
        return _fake_tensor_quant_backward(inputs, amax, grad_outputs), None, None, None, None, None


def _onnx_fp8_quantize(g, inputs, scale_inv, trt_high_precision_dtype):
    """Helper Function for Quantization."""
    output_shape = torch.onnx.symbolic_helper._get_tensor_sizes(inputs)

    # TRT StronglyType only supports FP16 QDQs
    # custom ops, so cast the input if needed.
    input_type = inputs.type().scalarType()
    assert (
        trt_high_precision_dtype == input_type or trt_high_precision_dtype == "Float"
    ), "TRT StronglyType requires both weights and amax to be in the BF16/FP16, or the QDQ in Float."
    if trt_high_precision_dtype != input_type:
        inputs = g.op("Cast", inputs, to_i=onnx_dtype_map[trt_high_precision_dtype])

    scale = g.op(
        "Constant",
        value_t=torch.tensor(scale_inv).to(torch_dtype_map[trt_high_precision_dtype]),
    )
    q_op = g.op("trt::TRT_FP8QuantizeLinear", inputs, scale).setType(
        inputs.type().with_dtype(torch.uint8).with_sizes(output_shape)
    )
    return q_op


def _onnx_fp8_dequantize(g, inputs, scale_inv, otype=None, trt_high_precision_dtype="Float"):
    """Helper Function for Dequantization."""
    output_shape = torch.onnx.symbolic_helper._get_tensor_sizes(inputs)
    assert (
        trt_high_precision_dtype == otype or trt_high_precision_dtype == "Float"
    ), "TRT StronglyType requires both weights and amax to be in the BF16/FP16, or the QDQ in Float."
    scale = g.op(
        "Constant",
        value_t=torch.tensor(scale_inv, dtype=torch_dtype_map[otype]),
    )
    out = g.op("trt::TRT_FP8DequantizeLinear", inputs, scale).setType(
        inputs.type().with_dtype(torch_dtype_map[trt_high_precision_dtype]).with_sizes(output_shape)
    )

    # DQ outputs are currently constrained to FP32 due to a similar limitation in ORT
    # custom ops, so cast the output if needed.
    if trt_high_precision_dtype != otype:
        out = g.op("Cast", out, to_i=onnx_dtype_map[otype])
    return out


class ScaledE4M3Function(Function):
    """E4M3fy input with scale."""

    @staticmethod
    @symbolic_helper.parse_args("v", "t", "i", "b", "b", "s")
    def symbolic(g, inputs, amax=None, E=4, M=3, trt_high_precision_dtype="Float"):  # noqa: N803
        """ONNX symbolic function."""
        if amax is None:
            scale = 1.0
        else:
            scale = 448.0 / float(amax)
        otype = inputs.type().scalarType()
        q_tensor = _onnx_fp8_quantize(g, inputs, 1.0 / scale, trt_high_precision_dtype)
        return _onnx_fp8_dequantize(g, q_tensor, 1.0 / scale, otype, trt_high_precision_dtype)

    @staticmethod
    # Default values could cause errors from TorchDynamo during torch.export
    def forward(ctx, inputs, amax, E, M, trt_high_precision_dtype="Float"):  # noqa: N803
        """Forward method."""
        if E != 4 or M != 3:
            raise NotImplementedError("Only support E=4 & M=3 for now.")

        ctx.save_for_backward(inputs)
        ctx.amax = amax
        outputs = quantize_op(
            inputs, amax, num_bits=8, exponent_bits=4, unsigned=False, narrow_range=False
        )

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        """Implements straight through estimation with clipping."""
        (inputs,) = ctx.saved_tensors
        amax = torch.tensor(
            ctx.amax if ctx.amax is not None else 448.0, dtype=torch.float32, device=inputs.device
        )
        grad_inputs = _fake_tensor_quant_backward(inputs, amax, grad_outputs)
        return grad_inputs, None, None, None, None


class TensorQuantFunction(Function):
    """A universal tensor quantization function.

    Take an input tensor, output an quantized tensor. The granularity of scale can be interpreted from the
    shape of amax.
    output_dtype indicates whether the quantized value will be stored in integer or float. The reason we want to store
    it in float is the pytorch function takes the quantized value may not accept integer input, e.g. Conv2D.

    It uses 2^num_bits -1 values instead of 2^num_bits. e.g., for num_bits=8, it uses [-127, 127] instead of [-128, 127]
    """

    @staticmethod
    @symbolic_helper.parse_args("v", "t", "i", "b", "b", "s")
    def symbolic(
        g,
        inputs,
        amax,
        num_bits=8,
        unsigned=False,
        narrow_range=True,
        trt_high_precision_dtype="Float",
    ):
        """ONNX symbolic function."""
        return _onnx_int8_helper(
            g, inputs, amax, num_bits, unsigned, narrow_range, trt_high_precision_dtype
        )

    @staticmethod
    def forward(
        ctx,
        inputs,
        amax,
        num_bits=8,
        unsigned=False,
        narrow_range=True,
        trt_high_precision_dtype="Float",
    ):
        """Forward method.

        Follow tensorflow convention, max value is passed in and used to decide scale, instead of inputing scale
        directly. Though inputing scale directly may be more natural to use.

        Args:
            ctx: A Context object to store tensors for backward.
            inputs: A Tensor of type float32.
            amax: A Tensor of type float32. Inputs will be quantized within range [-amax, amax]
                amax will be broadcasted to inputs tensor.
            num_bits: A integer used to calculate scaling factor, scale = (2^(num_bits-1) - 1) / max
                Effectively, it indicates how many integer bits is used to represent the value. Default 8.
            output_dtype: A type of Tensor. torch.int32 or torch.float32.
            unsigned: A boolean. Use unsigned integer range. E.g. [0, 255] for num_bits=8. Default False.
            narrow_range: A boolean. Use symmetric integer range for signed quantization
                E.g. [-127,127] instead of [-128,127] for num_bits=8. Default True.

        Returns:
            outputs: A Tensor of type output_dtype.
            scale: A Tensor of type float32. outputs / scale will dequantize outputs tensor.

        Raises:
            ValueError:
        """
        ctx.save_for_backward(inputs, amax)
        outputs, scale = _tensor_quant(inputs, amax, num_bits, unsigned, narrow_range)
        # Check if scale overflows FP16
        if outputs.dtype == torch.half and scale.max() > 65504:
            raise ValueError(f"scale is too large for FP16 with amax={amax}")
        return outputs, scale.to(inputs.dtype)

    @staticmethod
    def backward(ctx, grad_outputs, grad_scale):
        """Implements straight through estimation with clipping.

        For -amax <= input <= amax the gradient passes straight through, otherwise the gradient is zero.

        Args:
            ctx: A Context object with saved tensors from forward.
            grad_outputs: A tensor of gradient of outputs.
            grad_scale: A tensor of gradient of scale.

        Returns:
            grad_inputs: A tensor of gradient.
        """
        inputs, amax = ctx.saved_tensors
        zero = grad_outputs.new_zeros(1)  # create a zero tensor with the same type and device
        grad_inputs = torch.where(inputs.abs() <= amax, grad_outputs, zero)
        return grad_inputs, None, None, None, None, None


class LegacyFakeTensorQuantFunction(Function):
    """Fake version of TensorQuantFunction.

    See comments of TensorQuantFunction, arguments are the same.
    """

    @staticmethod
    def forward(ctx, inputs, amax, num_bits=8, unsigned=False, narrow_range=True):
        """Forward method."""
        ctx.save_for_backward(inputs, amax)
        outputs, scale = _tensor_quant(inputs, amax, num_bits, unsigned, narrow_range)
        return outputs / scale.to(inputs.dtype)

    @staticmethod
    def backward(ctx, grad_outputs):
        """Implements straight through estimation."""
        inputs, amax = ctx.saved_tensors
        zero = grad_outputs.new_zeros(1)
        grad_inputs = torch.where(inputs.abs() <= amax, grad_outputs, zero)
        return grad_inputs, None, None, None, None


def _tensor_quant(inputs, amax, num_bits=8, unsigned=False, narrow_range=True):
    """Shared function body between TensorQuantFunction and FakeTensorQuantFunction."""
    # Fine scale, per channel scale will be handled by broadcasting, which could be tricky. Pop a warning.
    if unsigned:
        if inputs.min() < 0.0:
            raise TypeError("Negative values encountered in unsigned quantization.")

    # Computation can be done in FP32 to prevent potential over flow.
    input_dtype = inputs.dtype
    if inputs.dtype == torch.half:
        inputs = inputs.float()
    if amax.dtype == torch.half:
        amax = amax.float()

    min_amax = amax.min()
    if min_amax < 0:
        raise ValueError("Negative values in amax")

    max_bound = torch.tensor((2.0 ** (num_bits - 1 + int(unsigned))) - 1.0, device=amax.device)
    if unsigned:
        min_bound = 0
    elif narrow_range:
        min_bound = -max_bound
    else:
        min_bound = -max_bound - 1
    scale = max_bound / amax

    epsilon = 1.0 / (1 << 24)
    if min_amax <= epsilon:  # Treat amax smaller than minimum representable of fp16 0
        zero_amax_mask = amax <= epsilon
        scale[zero_amax_mask] = 0  # Value quantized with amax=0 should all be 0

    outputs = torch.clamp((inputs * scale).round_(), min_bound, max_bound)

    if min_amax <= epsilon:
        scale[zero_amax_mask] = (
            1.0  # Return 1 makes more sense for values quantized to 0 with amax=0
        )

    if input_dtype == torch.half:
        outputs = outputs.half()

    return outputs, scale


class FakeAffineTensorQuantFunction(Function):
    """Fake version of affine quantization.

    gemmlowp style scale+shift quantization. See more details in
    https://github.com/google/gemmlowp/blob/master/doc/quantization.md.

    We DO NOT recommend affine quantization on weights for performance reason. There might be value to affine quantize
    activation as it can be cancelled by bias and comes with no performance penalty. This functionality is only added
    for experimental purpose.
    """

    @staticmethod
    def forward(ctx, inputs, min_range, max_range, num_bits=8):
        """As it will be only applied on activation with per tensor granularity, broadcast is not needed.

        Args:
            ctx: Pytorch convention.
            inputs: A Tensor of type float32.
            min_range: A float.
            max_range: A float.
            num_bits: An integer

        Returns:
            outputs: A Tensor of type output_dtype
        """
        ctx.save_for_backward(inputs, min_range, max_range)

        step_size = (max_range - min_range) / (2.0**num_bits - 1)

        min_bound = -(2.0 ** (num_bits - 1))
        max_bound = 2.0 ** (num_bits - 1) - 1

        quant_zero = torch.round(min_range / step_size) - min_bound
        quantized = torch.round(inputs / step_size) - quant_zero
        quantized = torch.clamp(quantized, min_bound, max_bound)

        outputs = (quantized + quant_zero) * step_size

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        """Implements straight through estimation with clipping.

        Args:
            ctx: Pytorch convention.
            grad_output: A tensor of gradient of outputs.

        Returns:
            grad_inputs: A tensor of gradient
        """
        inputs, min_range, max_range = ctx.saved_tensors
        zero = grad_outputs.new_zeros(1)
        grad_inputs = torch.where((inputs <= max_range) * (inputs >= min_range), grad_outputs, zero)
        return grad_inputs, None, None, None


tensor_quant = TensorQuantFunction.apply
legacy_fake_tensor_quant = LegacyFakeTensorQuantFunction.apply
fake_tensor_quant = FakeTensorQuantFunction.apply
fake_affine_tensor_quant = FakeAffineTensorQuantFunction.apply
scaled_e4m3 = ScaledE4M3Function.apply
