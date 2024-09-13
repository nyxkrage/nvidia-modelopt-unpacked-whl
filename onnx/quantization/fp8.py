# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Performs FP8 GEMM only quantization of an ONNX model, and returns the ONNX ModelProto."""

import logging
import os
import tempfile
from typing import List

import numpy as np
import onnx
import onnx_graphsurgeon as gs
from onnx import numpy_helper
from onnx_graphsurgeon.ir.graph import Graph
from onnxmltools.utils.float16_converter import convert_float_to_float16
from onnxruntime.quantization import (
    CalibrationMethod,
    quantize_static,
)
from onnxruntime.quantization.calibrate import CalibrationDataReader

from modelopt.onnx.quantization.graph_utils import (
    expand_node_names_from_patterns,
    find_nodes_to_exclude,
    get_tensor_producer_nodes,
    insert_fp8_mha_casts,
)
from modelopt.onnx.quantization.ort_utils import configure_ort


def _find_unsupported_fp8_convs_to_exclude(graph: Graph):
    """Find unsupported FP8 Conv nodes to exclude.

    The input and output channel alignment requirement for FP8
    conv kernels for input and output type FP8E4M3 should be both 16.
    The filter size for FP8 conv kernels should be less than 32.

    Args:
        graph: Onnx model graph.

    Returns:
        List of Conv nodes.
    """
    unsupported_conv_nodes = []
    for node in graph.nodes:
        if node.op == "Conv":
            weight = node.inputs[1]
            output_channel = weight.shape[0]
            input_channel = weight.shape[1]
            if output_channel % 16 != input_channel % 16:
                logging.info(f"Found unpaddable conv for FP8: {node.name}")
                unsupported_conv_nodes.append(node.name)
                continue

            filter_size = weight.shape[2] * weight.shape[3]
            if len(weight.shape) == 5:
                filter_size *= weight.shape[4]
            if filter_size > 32:
                logging.info(f"Found large filter conv for FP8: {node.name}")
                unsupported_conv_nodes.append(node.name)

    return unsupported_conv_nodes


def int8_to_fp8(onnx_path: str, verbose: bool = False) -> onnx.onnx_pb.ModelProto:
    """Converts the INT8 quantized model to FP8 quantized model.

    Note. This conversion works only for max calibrated INT8 models.

    Args:
        onnx_path: Path to the INT8 quantized ONNX model.
        verbose: Whether to print verbose logs or not.

    Returns:
        FP8 quantized ONNX model.
    """
    onnx_model = onnx.load(onnx_path)
    graph = onnx_model.graph
    initializers = graph.initializer
    tensor_producers = get_tensor_producer_nodes(graph)
    processed_tensor = set()

    def _get_initializer_index(name: str):
        for idx, init in enumerate(initializers):
            if init.name == name:
                return idx

    def _int8_scale_to_fp8_scale(scale: np.ndarray, scale_name: str):
        np_scale = onnx.numpy_helper.to_array(scale)
        np_fp8_scale = (np_scale * 448.0) / 127.0
        dtype = onnx.helper.tensor_dtype_to_np_dtype(scale.data_type)
        return numpy_helper.from_array(np_fp8_scale.astype(dtype), scale_name)

    def _convert(node: onnx.onnx_ml_pb2.NodeProto):
        if verbose:
            logging.info(f"Processing {node.name}")

        scale_name = node.input[1]
        zero_point_name = node.input[2]

        if scale_name not in processed_tensor:
            scale_idx = _get_initializer_index(scale_name)
            if scale_idx is not None:
                scale = initializers[scale_idx]
                fp8_scale = _int8_scale_to_fp8_scale(scale, scale_name)
                initializers[scale_idx].CopyFrom(fp8_scale)
            else:
                producer_node = tensor_producers[scale_name]
                scale = producer_node.attribute[0].t
                fp8_scale = _int8_scale_to_fp8_scale(scale, scale_name)
                producer_node.attribute[0].t.CopyFrom(fp8_scale)
            processed_tensor.add(scale_name)

        if zero_point_name not in processed_tensor:
            zero_point_idx = _get_initializer_index(zero_point_name)
            zero_point = initializers[zero_point_idx]
            dtype = onnx.helper.tensor_dtype_to_np_dtype(zero_point.data_type)
            vals = np.array(zero_point.int32_data, dtype=dtype).tolist()

            np_zero_point = onnx.helper.make_tensor(
                zero_point_name, onnx.TensorProto.FLOAT8E4M3FN, zero_point.dims, vals
            )
            initializers[zero_point_idx].CopyFrom(np_zero_point)
            processed_tensor.add(zero_point_name)

    # Iterate through the nodes and convert the scales and zero points
    for node in graph.node:
        if node.op_type in ["DequantizeLinear", "QuantizeLinear"]:
            _convert(node)

    return onnx_model


def quantize(
    onnx_path: str,
    calibration_method: str = "max",
    calibration_data_reader: CalibrationDataReader = None,
    calibration_cache_path: str = None,
    op_types_to_quantize: List[str] = None,
    op_types_to_exclude: List[str] = None,
    nodes_to_quantize: List[str] = None,
    nodes_to_exclude: List[str] = None,
    use_external_data_format: bool = True,
    intermediate_generated_files: List[str] = [],
    verbose: bool = False,
    trt_extra_plugin_lib_paths: str = None,
    high_precision_dtype: str = "fp16",
    mha_accumulation_dtype: str = "fp32",
) -> onnx.onnx_pb.ModelProto:
    """Applies FP8 GEMM only quantization to an ONNX file.

    Currently, ['Conv', 'Gemm', 'MatMul'] quantization is supported.
    """
    logging.info("Quantization Mode: fp8")
    if calibration_method != "max":
        raise RuntimeError("Only the max calibration method is supported for FP8 quantization.")

    # Load the onnx graph
    onnx_model = onnx.load(onnx_path, load_external_data=use_external_data_format)
    graph = gs.import_onnx(onnx_model)
    graph.toposort()

    # The quantizable op types for FP8 are limited to Conv, Gemm, and Matmul
    fp8_supported_op_types = ["Gemm", "MatMul", "Conv"]
    op_types_to_quantize = op_types_to_quantize or fp8_supported_op_types
    if not set(op_types_to_quantize) <= set(fp8_supported_op_types):
        raise RuntimeError(
            f"Unsupported op types in fp8 mode: '{set(op_types_to_quantize) - set(fp8_supported_op_types)}'"
        )

    # Change the default configuration of ORT quantization
    op_types = set([node.op for node in graph.nodes])
    trt_guided_options, quantizable_op_types = configure_ort(
        list(op_types), op_types_to_quantize, trt_extra_plugin_lib_paths
    )
    logging.info(
        "Quantizable op types in the model:"
        f" {[t for t in quantizable_op_types if t in op_types]}"
    )

    # Collect node names to include in quantization
    nodes_to_quantize = expand_node_names_from_patterns(graph, nodes_to_quantize)
    if not nodes_to_quantize:
        nodes_to_quantize = [node.name for node in graph.nodes if node.op in op_types_to_quantize]
    if not nodes_to_quantize:
        logging.info(
            "No node or node type is selected for quantization or model does not have them!"
        )
        return

    # Collect node names to exclude from quantization
    nodes_to_exclude = find_nodes_to_exclude(graph, nodes_to_exclude, op_types_to_exclude)
    nodes_to_exclude.extend(_find_unsupported_fp8_convs_to_exclude(graph))

    logging.info(f"Total number of nodes: {len(graph.nodes)}")
    logging.info(f"Skipped node count: {len(nodes_to_exclude)}")
    if verbose:
        logging.info(f"Skipped nodes: {nodes_to_exclude}")

    # Create a temp file for intermediate model
    tmp_onnx_file, tmp_onnx_path = tempfile.mkstemp(suffix=".onnx")
    os.close(tmp_onnx_file)

    # Quantize in INT8 mode using ORT's MinMax calibration method, with
    # ActivationSymmetric as True, which is equivalent to max calibration
    quantize_static(
        onnx_path,
        tmp_onnx_path,
        calibration_data_reader,
        op_types_to_quantize=op_types_to_quantize,
        nodes_to_quantize=nodes_to_quantize,
        nodes_to_exclude=nodes_to_exclude,
        per_channel=True,
        extra_options=trt_guided_options,
        use_external_data_format=use_external_data_format,
        calibrate_method=CalibrationMethod.MinMax,
    )
    intermediate_generated_files.append(tmp_onnx_path)
    if use_external_data_format:
        intermediate_generated_files.append(tmp_onnx_path + ".data")

    # Post-processing of the onnx model after ORT quantization
    onnx_model = int8_to_fp8(tmp_onnx_path, verbose)

    if high_precision_dtype == "fp16":
        # We need to convert float to float16 so as to speed up layers like LayerNorm or GroupNorm.
        logging.info("Converting float tensors to float16")
        onnx_model = convert_float_to_float16(
            onnx_model, keep_io_types=True, disable_shape_infer=True
        )

        if mha_accumulation_dtype == "fp32":
            # Insert Cast nodes in MHA's BMM1 and BMM2's input and output tensors because
            # The compiler only has FP32 accumulation kernels for FP8 MHAs.
            logging.info("Inserting Cast nodes to enable FP8+FP16 MHA")
            onnx_model = insert_fp8_mha_casts(onnx_model)

    return onnx_model
