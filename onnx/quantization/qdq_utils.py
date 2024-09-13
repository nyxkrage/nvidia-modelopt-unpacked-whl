# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Various utils to support inserting Q/DQ nodes."""

import logging
from typing import Any, Dict, Sequence, Set

import numpy as np
import onnx
import onnx_graphsurgeon as gs
from onnx import numpy_helper
from onnx.reference.custom_element_types import float8e4m3fn
from onnx.reference.ops.op_cast import Cast_19 as Cast

from modelopt.onnx.quantization.graph_utils import (
    get_tensor_consumer_nodes,
    get_tensor_producer_nodes,
)

QUANTIZE_NODE_NAME = "QuantizeLinear"
DEQUANTIZE_NODE_NAME = "DequantizeLinear"


def use_trt_qdq_ops():
    """Globally set node names to TRT custom names."""
    global QUANTIZE_NODE_NAME
    QUANTIZE_NODE_NAME = "TRT_INT4QuantizeLinear"
    global DEQUANTIZE_NODE_NAME
    DEQUANTIZE_NODE_NAME = "TRT_INT4DequantizeLinear"


def _wq_name(name: str):
    return name + "_i4"


def _scale_name(name: str):
    return name + "_scale"


def _awq_scale_name(name: str):
    return name + "_awq_scale"


def _zp_name(name: str):
    return name + "_zp"


def _q_name(name: str):
    return name + "_QuantizeLinear"


def _q_out_name(name: str):
    return name + "_QuantizeLinear_Output"


def _dq_name(name: str):
    return name + "_DequantizeLinear"


def _pqs_name(name: str):
    return name + "_PQS"


def _dq_out_name(name: str):
    return name + "_DequantizeLinear_Output"


def _pqs_out_name(name: str):
    return name + "_PQS_Tensor"


def make_gs_quantized_weight(name: str, wq: np.ndarray, dtype) -> gs.Constant:
    """Create a GraphSurgeon tensor from a quantized weight tensor.

    `name` is the desired _basename_ of the tensor.
    """
    return gs.make_constant(_wq_name(name), np.asarray(wq), dtype)


def make_gs_zp(name: str, shape: Sequence[int], dtype) -> gs.Constant:
    """Create a GraphSurgeon zero-point tensor of all zeroes with the given shape.

    `name` is the desired _basename_ of the tensor.
    """
    return gs.make_constant(
        _zp_name(name),
        np.zeros(shape, dtype=onnx.mapping.TENSOR_TYPE_MAP[int(dtype)].np_dtype),
        dtype,
    )


def make_gs_scale(name: str, scale: np.ndarray) -> gs.Constant:
    """Create a GraphSurgeon scale tensor from the given numpy array.

    `name` is the desired _basename_ of the tensor.
    """
    return gs.Constant(_scale_name(name), np.asarray(scale))


def make_gs_awq_scale(name: str, scale: np.ndarray) -> gs.Constant:
    """Create a GraphSurgeon scale tensor from the given numpy array.

    `name` is the desired _basename_ of the tensor.
    """
    return gs.Constant(_awq_scale_name(name), np.asarray(scale))


def make_gs_quantize_output(
    name: str, shape: Sequence[int], dtype: onnx.TensorProto.DataType
) -> gs.Variable:
    """Create a GraphSurgeon variable representing the output of a quantize node.

    `name` is the desired _basename_ of the node.
    """
    return gs.make_variable(_q_out_name(name), dtype=dtype, shape=shape)


def make_gs_quantize_node(
    name: str, inputs: Sequence[gs.Tensor], outputs: Sequence[gs.Tensor]
) -> gs.Node:
    """Create a GraphSurgeon Quantize node.

    `name` is the desired _basename_ of the node.
    """
    return gs.Node(
        QUANTIZE_NODE_NAME,
        name=_q_name(name),
        inputs=inputs,
        outputs=outputs,
    )


def make_gs_pre_quant_scale_output(
    name: str,
    shape: Sequence[int],
    dtype: np.dtype,
) -> gs.Variable:
    """Create a GraphSurgeon variable representing the output of a quantize node.

    `name` is the desired _basename_ of the node.
    """
    return gs.Variable(_pqs_out_name(name), dtype=dtype, shape=shape)


def make_gs_dequantize_output(
    name: str,
    shape: Sequence[int],
    dtype: np.dtype,
) -> gs.Variable:
    """Create a GraphSurgeon variable representing the output of a quantize node.

    `name` is the desired _basename_ of the node.
    """
    return gs.Variable(_dq_out_name(name), dtype=dtype, shape=shape)


def make_gs_pre_quant_scale_node(
    name: str, inputs: Sequence[gs.Tensor], outputs: Sequence[gs.Tensor]
) -> gs.Node:
    """Create a GraphSurgeon Dequantize node.

    `name` is the desired _basename_ of the node.
    """
    return gs.Node(
        "Mul",
        name=_pqs_name(name),
        inputs=inputs,
        outputs=outputs,
    )


def make_gs_dequantize_node(
    name: str,
    inputs: Sequence[gs.Tensor],
    outputs: Sequence[gs.Tensor],
    attributes: Dict[str, Any] = None,
) -> gs.Node:
    """Create a GraphSurgeon Dequantize node.

    `name` is the desired _basename_ of the node.
    """
    return gs.Node(
        DEQUANTIZE_NODE_NAME,
        name=_dq_name(name),
        inputs=inputs,
        outputs=outputs,
        attrs=attributes,
    )


def _postprocess_qdq(
    graph: gs.Graph,
    orig_weight_names: Set[str],
    q_nodes: Dict[str, gs.Node] = {},
    dq_nodes: Dict[str, gs.Node] = {},
):
    # Inserts all newly created nodes to graph.
    # Update all consumers of original initializers to point to the DQ nodes.
    for node in graph.nodes:
        for i in range(len(node.inputs)):
            key = node.inputs[i].name
            if key not in orig_weight_names:
                continue
            node.inputs[i] = dq_nodes[key].outputs[0]

    # Insert new nodes.
    graph.nodes.extend(q_nodes.values())
    graph.nodes.extend(dq_nodes.values())

    graph.cleanup()
    graph.toposort()


def insert_pre_quant_scale_nodes(
    graph: gs.Graph, input_tensors: Dict[str, str], pre_quant_scale: Dict[str, np.ndarray]
):
    """Insert new mul nodes into graph.

    Args:
        graph: The graph to modify.
        input_tensors: A dictionary of weight tensor names mapped to corresponding input tensor names
        pre_quant_scale: A map from ONNX input tensor name to corresponding pre-quant scale.
    """

    def _insert_helper(
        weight_tensor_name: str,
        input_tensor_name: str,
        scale: np.ndarray,
        mul_nodes: Dict[str, gs.Node],
    ):
        pre_quant_scale_tensor = make_gs_awq_scale(weight_tensor_name, scale)
        # TODO: Study effects of caching Gemm/Matmul nodes on perf and mem usage.
        gemm_nodes = [node for node in graph.nodes if node.op in ["Gemm", "MatMul"]]
        for node in gemm_nodes:
            input_set = set([input.name for input in node.inputs])
            input_idxs = {input.name: idx for idx, input in enumerate(node.inputs)}
            if _dq_out_name(weight_tensor_name) in input_set and input_tensor_name in input_set:
                pqs_in = node.inputs[input_idxs[input_tensor_name]]
                pqs_out = make_gs_pre_quant_scale_output(
                    weight_tensor_name, shape=pqs_in.shape, dtype=scale.dtype
                )
                mul_node = make_gs_pre_quant_scale_node(
                    weight_tensor_name, inputs=[pqs_in, pre_quant_scale_tensor], outputs=[pqs_out]
                )
                node.inputs[input_idxs[input_tensor_name]] = mul_node.outputs[0]
                mul_nodes[weight_tensor_name] = mul_node

    mul_nodes = {}
    for w_name, scale in pre_quant_scale.items():
        _insert_helper(w_name, input_tensors[w_name], scale, mul_nodes)

    graph.nodes.extend(mul_nodes.values())

    graph.cleanup()
    graph.toposort()


def insert_dq_nodes(
    graph: gs.Graph,
    scales: Dict[str, np.ndarray],
    quantized_weights: Dict[str, np.ndarray],
    attributes: Dict[str, Any] = None,
):
    """Insert new initializers and DQ nodes into graph.

    Args:
        graph: The graph to modify.
        weights: A map from ONNX initializer name to tensor.
        scales: A map from ONNX initializer name to desired scale factor for that initializer.
        dq_only: Whether to only insert dq nodes.
    """

    def _insert_helper(
        name: str,
        wq: np.ndarray,
        scale: np.ndarray,
        dq_nodes: Dict[str, gs.Node],
        attrs: Dict[str, Any],
    ):
        wq_tensor = make_gs_quantized_weight(name, wq, onnx.TensorProto.INT4)
        scale_tensor = make_gs_scale(name, scale)
        zp_tensor = make_gs_zp(name, scale.shape, onnx.TensorProto.INT4)
        dq_out = make_gs_dequantize_output(name, shape=wq.shape, dtype=scale.dtype)
        dq_node = make_gs_dequantize_node(
            name,
            inputs=[wq_tensor, scale_tensor, zp_tensor],
            outputs=[dq_out],
            attributes=attrs,
        )
        dq_nodes[name] = dq_node

    dq_nodes = {}
    for name, scale in scales.items():
        _insert_helper(name, quantized_weights[name], scale, dq_nodes, attributes)

    _postprocess_qdq(
        graph,
        orig_weight_names=set(scales.keys()),
        dq_nodes=dq_nodes,
    )


def insert_qdq_nodes(
    graph: gs.Graph,
    scales: Dict[str, np.ndarray],
    weight_map: Dict[str, gs.Tensor],
):
    """Insert scales and QDQ nodes into graph.

    Args:
        graph: The graph to modify.
        scales: A map from ONNX initializer name to desired scale factor for that initializer.
        weight_map: A map from ONNX initializer name to graphsurgeon tensor.
    """

    def _insert_helper(
        name: str,
        weight_to_quantize: gs.Tensor,
        scale: np.ndarray,
        q_nodes: Dict[str, gs.Node],
        dq_nodes: Dict[str, gs.Node],
    ):
        scale_tensor = make_gs_scale(name, scale)
        zp_tensor = make_gs_zp(name, scale.shape, onnx.TensorProto.INT4)
        q_out = make_gs_quantize_output(name, weight_to_quantize.shape, onnx.TensorProto.INT4)
        q_node = make_gs_quantize_node(
            name, inputs=[weight_to_quantize, scale_tensor, zp_tensor], outputs=[q_out]
        )
        dq_out = make_gs_dequantize_output(name, shape=weight_to_quantize.shape, dtype=scale.dtype)
        dq_node = make_gs_dequantize_node(
            name, inputs=[q_out, scale_tensor, zp_tensor], outputs=[dq_out]
        )
        q_nodes[name] = q_node
        dq_nodes[name] = dq_node

    q_nodes, dq_nodes = {}, {}
    for name, scale in scales.items():
        _insert_helper(name, weight_map[name], scale, q_nodes, dq_nodes)

    _postprocess_qdq(
        graph,
        orig_weight_names=set(scales.keys()),
        q_nodes=q_nodes,
        dq_nodes=dq_nodes,
    )


def replace_scale_values(graph: onnx.onnx_ml_pb2.GraphProto, act_scales_dict: Dict[str, float]):
    """Replaces the scales values from calibration cache."""
    initializers = graph.initializer

    def _get_initializer_index(name: str):
        for idx, init in enumerate(initializers):
            if init.name == name:
                return idx
        return None

    for node in graph.node:
        if node.op_type == "QuantizeLinear":
            scale_input_name = node.input[1]
            if scale_input_name in act_scales_dict:
                idx = _get_initializer_index(scale_input_name)
                scale = numpy_helper.from_array(
                    np.float32(act_scales_dict[scale_input_name]), scale_input_name
                )
                initializers[idx].CopyFrom(scale)
            else:
                # If the scale is not present in the act_scales_dict
                # then the current node must be an weight quantizer and
                # the weight should be available in the graph initializer
                assert (
                    _get_initializer_index(node.input[0]) is not None
                ), f"Tensor {node.input[0]} not found in initializers."


def qdq_to_dq(
    onnx_model: onnx.onnx_pb.ModelProto, verbose: bool = False
) -> onnx.onnx_pb.ModelProto:
    """Convert FP32/FP16 weights of the given ONNX model to INT8/FP8 weights.

    Q nodes will get removed from the weights and have only DQ nodes with those converted INT8/FP8
    weights in the output model. Also dangling Q nodes get fused and update its consumer's weight.

    Args:
        onnx_model: ONNX model protobuf.

    Returns:
        ONNX model protobuf with only DQ nodes for weights and QDQ nodes for activations.
    """
    graph = onnx_model.graph
    initializers = graph.initializer

    def _get_initializer_index(name: str):
        for idx, init in enumerate(initializers):
            if init.name == name:
                return idx

    def _get_tensor_type(tensor_name):
        for value_info in graph.value_info:
            if value_info.name == tensor_name:
                return value_info.type.tensor_type.elem_type
        return None

    def _remove_unnecessary_cast():
        # Remove two pattern of unnecessary Cast node
        cast_indices = []

        tensor_consumers = get_tensor_consumer_nodes(graph)

        # find all Cast node with same input and output type
        for node_idx, node in enumerate(graph.node):
            if node.op_type != "Cast":
                continue

            # if input type matches attribute "to", this is a useless Cast node
            assert len(node.input) == 1
            input_name = node.input[0]
            idx = _get_initializer_index(input_name)
            if idx is not None:
                data_type = initializers[idx].data_type
            else:
                data_type = _get_tensor_type(input_name)

            attr = node.attribute[0]
            assert attr.name == "to"

            # Pattern 1: Input and Output Type are the same.
            if data_type == attr.i:
                cast_indices.append(node_idx)
            else:
                # Pattern 2: Input and Output Type differ but Cast node doesn't have a producer
                # We do the conversion and fuse Cast node.
                if idx is not None:
                    cast_indices.append(node_idx)
                    # Replace Q node input with new input
                    cast_input = onnx.numpy_helper.to_array(initializers[idx])

                    dtype = onnx.helper.tensor_dtype_to_np_dtype(attr.i)
                    converted_tensor = onnx.numpy_helper.from_array(
                        cast_input.astype(dtype), input_name
                    )

                    initializers[idx].CopyFrom(converted_tensor)
                else:
                    continue

            # Renew input of consumer nodes
            output_name = node.output[0]
            consumers = tensor_consumers[output_name]
            for q_node in consumers:
                for i in range(len(q_node.input)):
                    if q_node.input[i] == output_name:
                        q_node.input[i] = input_name
                        break

        # Delete Cast node
        for node_idx in sorted(cast_indices, reverse=True):
            del graph.node[node_idx]

    def _convert(node: onnx.onnx_ml_pb2.NodeProto):
        if verbose:
            logging.info(f"Processing {node.name}")

        idx1 = _get_initializer_index(node.input[0])
        w = initializers[idx1]

        w32 = onnx.numpy_helper.to_array(w)

        idx2 = _get_initializer_index(node.input[1])
        if idx2 is not None:
            y_scale = initializers[idx2]
        else:
            producer_node = tensor_producers[node.input[1]]
            attr = producer_node.attribute[0]
            assert attr.name == "value"
            y_scale = attr.t

        np_y_scale = onnx.numpy_helper.to_array(y_scale)

        idx3 = _get_initializer_index(node.input[2])
        if idx3 is not None:
            zero_point = initializers[idx3]
        else:
            producer_node = tensor_producers[node.input[2]]
            attr = producer_node.attribute[0]
            assert attr.name == "value"
            zero_point = attr.t

        np_zero_point = onnx.numpy_helper.to_array(zero_point)

        dq_node = tensor_consumers[node.output[0]][0]
        next_node = tensor_consumers[dq_node.output[0]][0]

        # No transpose is needed for 2D "MatMul", only for 3D (fails with PETR otherwise)
        transpose_nodes = ["Conv", "Transpose", "Gemm"]
        is_3d_matmul = next_node.op_type in "MatMul" and len(np.shape(w32)) == 3
        do_transpose = next_node.op_type in transpose_nodes or is_3d_matmul

        if do_transpose:
            w32 = np.transpose(w32, axes=[0, 2, 1]) if is_3d_matmul else np.transpose(w32)

        # Scale should be a scaler or vector with the same length as the last dimension of the weight
        assert not np_y_scale.shape or w32.shape[-1] == np_y_scale.shape[0]

        fp8 = np_zero_point.dtype == float8e4m3fn

        if fp8:
            scaled = np.asarray(w32 / np_y_scale) + np_zero_point
        else:
            scaled = np.asarray((w32 / np_y_scale).round())
            np.clip(scaled + np_zero_point, -128, 127, out=scaled)

        if do_transpose:
            scaled = np.transpose(scaled, axes=[0, 2, 1]) if is_3d_matmul else np.transpose(scaled)

        if fp8:
            w8 = numpy_helper.from_array(
                Cast.eval(scaled, to=onnx.TensorProto.FLOAT8E4M3FN), w.name
            )
        else:
            w8 = numpy_helper.from_array(scaled.astype("int8"), w.name)

        initializers[idx1].CopyFrom(w8)

        return idx2, idx3

    _remove_unnecessary_cast()

    tensor_producers = get_tensor_producer_nodes(graph)
    tensor_consumers = get_tensor_consumer_nodes(graph)

    dangling_q_indices = []
    dangling_init_indices = []

    for node_idx, node in enumerate(graph.node):
        if node.op_type == "QuantizeLinear":
            weight_name = node.input[0]

            # Const input to quantize linear means weighted layer
            if weight_name not in tensor_producers:
                scale_init_idx, zp_init_idx = _convert(node)
                dangling_q_indices.append(node_idx)
                dangling_init_indices.extend([scale_init_idx, zp_init_idx])

                # Update following DQ nodes input name, each q should only have one dq consumer
                consumers = tensor_consumers[node.output[0]]
                assert len(consumers) == 1
                dq_node = consumers[0]
                assert dq_node.op_type == "DequantizeLinear"
                dq_node.input[0] = weight_name

    # Remove Q nodes
    for node_idx in sorted(dangling_q_indices, reverse=True):
        del graph.node[node_idx]

    return onnx_model
