#!/usr/bin/env python3
"""
convert_pytorch_fp8_to_mil.py

Constructs a PyTorch Convolution model with native FP8 (Float8_e4m3fn) weights,
and converts it directly to a CoreML .mlpackage and compiled MIL program (.mlmodelc / model.mil).

Supported Modes:
  - weight-only: FP8 compressed weights (Float8_e4m3fn) dequantized to FP16,
                 native FP16 activations (recommended for high throughput on ANE).
  - full-qdq:    FP8 weights and FP8 activation Quantize-Dequantize (QDQ) flow.
"""

import argparse
import json
import os
import shutil
import struct
import subprocess
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from coremltools.proto import Model_pb2, MIL_pb2


# ==============================================================================
# 1. PyTorch FP8 Model Definition
# ==============================================================================

class PyTorchFP8ConvModel(nn.Module):
    """
    PyTorch Conv2D Model holding native torch.float8_e4m3fn weight buffers.
    In PyTorch, inference executes by dequantizing weights to float16 with a scale factor.
    """
    def __init__(self, channels: int = 64, layers: int = 2, kernel_size: int = 3, scale: float = 0.0625):
        super().__init__()
        self.channels = channels
        self.layers = layers
        self.kernel_size = kernel_size
        self.scale = scale

        for i in range(layers):
            # Generate random FP8 weights using PyTorch
            # Weight shape: [C_out, C_in, K, K]
            w_float = torch.randn(channels, channels, kernel_size, kernel_size)
            w_fp8 = w_float.to(torch.float8_e4m3fn)
            self.register_buffer(f"weight_fp8_{i}", w_fp8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass in PyTorch using dequantized FP16 weights.
        """
        pad = self.kernel_size // 2
        for i in range(self.layers):
            w_fp8 = getattr(self, f"weight_fp8_{i}")
            w_fp16 = w_fp8.to(torch.float16) * self.scale
            x = F.conv2d(x, w_fp16, padding=pad)
        return x

    def get_layer_fp8_bytes(self, layer_idx: int) -> bytes:
        """
        Extracts raw byte representation of the torch.float8_e4m3fn weight tensor.
        """
        w_fp8 = getattr(self, f"weight_fp8_{layer_idx}")
        # View as uint8 byte stream to extract without numerical conversion
        return w_fp8.view(torch.uint8).cpu().numpy().tobytes()


# ==============================================================================
# 2. MIL Protobuf Builders
# ==============================================================================

def make_tensor_type(data_type: int, shape: list) -> MIL_pb2.ValueType:
    vt = MIL_pb2.ValueType()
    tt = vt.tensorType
    tt.dataType = data_type
    tt.rank = len(shape)
    for dim in shape:
        tt.dimensions.add().constant.size = dim
    return vt


def make_scalar_string_type() -> MIL_pb2.ValueType:
    vt = MIL_pb2.ValueType()
    tt = vt.tensorType
    tt.dataType = MIL_pb2.STRING
    tt.rank = 0
    return vt


def make_scalar_fp16_type() -> MIL_pb2.ValueType:
    vt = MIL_pb2.ValueType()
    tt = vt.tensorType
    tt.dataType = MIL_pb2.FLOAT16
    tt.rank = 0
    return vt


def make_scalar_int32_type() -> MIL_pb2.ValueType:
    vt = MIL_pb2.ValueType()
    tt = vt.tensorType
    tt.dataType = MIL_pb2.INT32
    tt.rank = 0
    return vt


def make_const_bytes_op(name: str, shape: list, data: bytes, data_type: int) -> MIL_pb2.Operation:
    op = MIL_pb2.Operation()
    op.type = "const"
    out = op.outputs.add()
    out.name = name
    out.type.CopyFrom(make_tensor_type(data_type, shape))

    val = op.attributes["val"]
    val.type.CopyFrom(out.type)
    val.immediateValue.tensor.bytes.values = data
    return op


def make_const_fp16_scalar_op(name: str, val_float: float) -> MIL_pb2.Operation:
    op = MIL_pb2.Operation()
    op.type = "const"
    out = op.outputs.add()
    out.name = name
    out.type.CopyFrom(make_scalar_fp16_type())

    val = op.attributes["val"]
    val.type.CopyFrom(out.type)
    val.immediateValue.tensor.bytes.values = struct.pack("<e", val_float)
    return op


def make_const_int32_scalar_op(name: str, val_int: int) -> MIL_pb2.Operation:
    op = MIL_pb2.Operation()
    op.type = "const"
    out = op.outputs.add()
    out.name = name
    out.type.CopyFrom(make_scalar_int32_type())

    val = op.attributes["val"]
    val.type.CopyFrom(out.type)
    val.immediateValue.tensor.ints.values.append(val_int)
    return op


def make_const_int32_tensor_op(name: str, shape: list, values: list) -> MIL_pb2.Operation:
    op = MIL_pb2.Operation()
    op.type = "const"
    out = op.outputs.add()
    out.name = name
    out.type.CopyFrom(make_tensor_type(MIL_pb2.INT32, shape))

    val = op.attributes["val"]
    val.type.CopyFrom(out.type)
    for v in values:
        val.immediateValue.tensor.ints.values.append(v)
    return op


def make_const_string_op(name: str, str_val: str) -> MIL_pb2.Operation:
    op = MIL_pb2.Operation()
    op.type = "const"
    out = op.outputs.add()
    out.name = name
    out.type.CopyFrom(make_scalar_string_type())

    val = op.attributes["val"]
    val.type.CopyFrom(out.type)
    val.immediateValue.tensor.strings.values.append(str_val)
    return op


def make_dequantize_op(out_name: str, in_name: str, scale_name: str, out_shape: list) -> MIL_pb2.Operation:
    op = MIL_pb2.Operation()
    op.type = "dequantize"
    op.inputs["input"].arguments.add().name = in_name
    op.inputs["scale"].arguments.add().name = scale_name

    out = op.outputs.add()
    out.name = out_name
    out.type.CopyFrom(make_tensor_type(MIL_pb2.FLOAT16, out_shape))
    return op


def make_quantize_op(out_name: str, in_name: str, scale_name: str, dtype_name: str, out_shape: list) -> MIL_pb2.Operation:
    op = MIL_pb2.Operation()
    op.type = "quantize"
    op.inputs["input"].arguments.add().name = in_name
    op.inputs["scale"].arguments.add().name = scale_name
    op.inputs["output_dtype"].arguments.add().name = dtype_name

    out = op.outputs.add()
    out.name = out_name
    out.type.CopyFrom(make_tensor_type(MIL_pb2.FLOAT8E4M3FN, out_shape))
    return op


def make_conv_op(out_name: str, in_name: str, weight_name: str, out_shape: list) -> MIL_pb2.Operation:
    op = MIL_pb2.Operation()
    op.type = "conv"
    op.inputs["x"].arguments.add().name = in_name
    op.inputs["weight"].arguments.add().name = weight_name
    op.inputs["strides"].arguments.add().name = "strides"
    op.inputs["pad_type"].arguments.add().name = "pad_type"
    op.inputs["pad"].arguments.add().name = "pad"
    op.inputs["dilations"].arguments.add().name = "dilations"
    op.inputs["groups"].arguments.add().name = "groups"

    out = op.outputs.add()
    out.name = out_name
    out.type.CopyFrom(make_tensor_type(MIL_pb2.FLOAT16, out_shape))
    return op


# ==============================================================================
# 3. Model Spec Construction
# ==============================================================================

def build_coreml_model_from_pytorch(
    model: PyTorchFP8ConvModel,
    batch: int = 1,
    spatial: int = 256,
    mode: str = "weight-only"
) -> Model_pb2.Model:
    channels = model.channels
    layers = model.layers
    kernel = model.kernel_size
    scale = model.scale

    in_shape = [batch, channels, spatial, spatial]
    w_shape = [channels, channels, kernel, kernel]

    model_proto = Model_pb2.Model()
    model_proto.specificationVersion = 9  # iOS 18+ / iOS 19

    # Model Description (I/O Feature descriptions)
    desc = model_proto.description
    in_desc = desc.input.add()
    in_desc.name = "x"
    in_desc.type.multiArrayType.dataType = Model_pb2.ArrayFeatureType.ArrayDataType.FLOAT16
    in_desc.type.multiArrayType.shape.extend(in_shape)

    final_out_name = f"conv_{layers - 1}"
    out_desc = desc.output.add()
    out_desc.name = final_out_name
    out_desc.type.multiArrayType.dataType = Model_pb2.ArrayFeatureType.ArrayDataType.FLOAT16
    out_desc.type.multiArrayType.shape.extend(in_shape)

    # Metadata
    meta = desc.metadata.userDefined
    meta["com.apple.coremltools.source"] = f"torch=={torch.__version__}"
    meta["workload.precision"] = f"FP8-{mode.upper()}"
    meta["workload.layers"] = str(layers)
    meta["workload.channels"] = str(channels)

    # Program & Function
    prog = model_proto.mlProgram
    prog.version = 1
    fn = prog.functions["main"]

    input_arg = fn.inputs.add()
    input_arg.name = "x"
    input_arg.type.CopyFrom(make_tensor_type(MIL_pb2.FLOAT16, in_shape))

    fn.opset = "CoreML9"
    block = fn.block_specializations["CoreML9"]
    block.outputs.append(final_out_name)

    # Shared Conv Constants
    block.operations.append(make_const_int32_tensor_op("strides", [2], [1, 1]))
    block.operations.append(make_const_int32_tensor_op("dilations", [2], [1, 1]))
    block.operations.append(make_const_int32_tensor_op("pad", [4], [0, 0, 0, 0]))
    block.operations.append(make_const_int32_scalar_op("groups", 1))
    block.operations.append(make_const_string_op("pad_type", "same"))

    # Dequantization scale
    block.operations.append(make_const_fp16_scalar_op("w_scale", scale))
    if mode == "full-qdq":
        block.operations.append(make_const_fp16_scalar_op("act_scale", scale))
        block.operations.append(make_const_string_op("dtype_fp8", "fp8e4m3fn"))

    current_in = "x"
    for i in range(layers):
        # Extract FP8 weight bytes from PyTorch model
        w_bytes = model.get_layer_fp8_bytes(i)
        raw_w_name = f"raw_w_{i}"
        w_name = f"weights_{i}"

        # 1. Constant FP8 raw weights
        block.operations.append(make_const_bytes_op(raw_w_name, w_shape, w_bytes, MIL_pb2.FLOAT8E4M3FN))

        # 2. Dequantize weights to FP16
        block.operations.append(make_dequantize_op(w_name, raw_w_name, "w_scale", w_shape))

        # 3. Activation QDQ (if full-qdq)
        conv_in = current_in
        if mode == "full-qdq":
            q_name = f"quant_{i}"
            dq_name = f"dequant_{i}"
            block.operations.append(make_quantize_op(q_name, current_in, "act_scale", "dtype_fp8", in_shape))
            block.operations.append(make_dequantize_op(dq_name, q_name, "act_scale", in_shape))
            conv_in = dq_name

        # 4. Conv2D
        conv_name = f"conv_{i}"
        block.operations.append(make_conv_op(conv_name, conv_in, w_name, in_shape))
        current_in = conv_name

    return model_proto


# ==============================================================================
# 4. Packaging and Compilation
# ==============================================================================

def export_mlpackage(model_proto: Model_pb2.Model, output_path: str) -> str:
    pkg_dir = output_path if output_path.endswith(".mlpackage") else f"{output_path}.mlpackage"
    data_dir = os.path.join(pkg_dir, "Data", "com.apple.CoreML")
    os.makedirs(data_dir, exist_ok=True)

    # Write serialized model.mlmodel
    model_file = os.path.join(data_dir, "model.mlmodel")
    with open(model_file, "wb") as f:
        f.write(model_proto.SerializeToString())

    # Write Manifest.json
    manifest = {
        "fileFormatVersion": "1.0.0",
        "itemInfoEntries": {
            "model-coreml": {
                "author": "com.apple.CoreML",
                "description": "CoreML Model Specification",
                "name": "model.mlmodel",
                "path": "com.apple.CoreML/model.mlmodel"
            }
        },
        "rootModelIdentifier": "model-coreml"
    }
    with open(os.path.join(pkg_dir, "Manifest.json"), "w") as f:
        json.dump(manifest, f, indent=4)

    return pkg_dir


def compile_with_coremlcompiler(pkg_dir: str, output_parent: str = ".") -> str:
    res = subprocess.run(
        ["xcrun", "coremlcompiler", "compile", pkg_dir, output_parent],
        capture_output=True,
        text=True
    )
    if res.returncode != 0:
        raise RuntimeError(f"coremlcompiler failed (code {res.returncode}):\n{res.stderr}\n{res.stdout}")

    pkg_base = os.path.basename(pkg_dir)
    stem = os.path.splitext(pkg_base)[0]
    compiled_dir = os.path.join(output_parent, f"{stem}.mlmodelc")
    return compiled_dir


# ==============================================================================
# 5. Main CLI Entry Point
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Convert PyTorch FP8 (Float8_e4m3fn) model to CoreML .mlpackage and MIL")
    parser.add_argument("--batch", type=int, default=1, help="Batch size (default: 1)")
    parser.add_argument("--channels", type=int, default=64, help="Channel dimension (default: 64)")
    parser.add_argument("--size", type=int, default=256, help="Spatial height/width (default: 256)")
    parser.add_argument("--kernel", type=int, default=3, help="Kernel size (default: 3)")
    parser.add_argument("--layers", type=int, default=2, help="Number of chained conv layers (default: 2)")
    parser.add_argument("--scale", type=float, default=0.0625, help="FP8 scale factor (default: 0.0625 = 2^-4)")
    parser.add_argument("--mode", choices=["weight-only", "full-qdq"], default="weight-only",
                        help="FP8 mode: weight-only (recommended) or full-qdq")
    parser.add_argument("--output", type=str, default="pytorch_fp8_conv", help="Output base path")
    parser.add_argument("--check", action="store_true", help="Run PyTorch forward verification pass")
    parser.add_argument("--dump-mil", action="store_true", help="Print the compiled model.mil program text")

    args = parser.parse_args()

    print("=================================================================")
    print("      PyTorch FP8 (Float8_e4m3fn) to CoreML MIL Converter       ")
    print("=================================================================")
    print(f"Architecture : Conv2D [{args.channels} -> {args.channels}], K={args.kernel}x{args.kernel}, L={args.layers}")
    print(f"Input Tensor : [{args.batch}, {args.channels}, {args.size}, {args.size}] FP16")
    print(f"Precision    : FP8 (Float8_e4m3fn), Mode: {args.mode}, Scale: {args.scale}")
    print(f"PyTorch ver  : {torch.__version__}")
    print()

    # 1. Instantiate PyTorch Model
    print("🛠️  [Step 1/4] Constructing PyTorch FP8 Model...")
    model = PyTorchFP8ConvModel(
        channels=args.channels,
        layers=args.layers,
        kernel_size=args.kernel,
        scale=args.scale
    )
    print(f"   Created {args.layers} layer(s) with native torch.float8_e4m3fn weight buffers.")
    for i in range(args.layers):
        w = getattr(model, f"weight_fp8_{i}")
        print(f"   - Layer {i}: shape={list(w.shape)}, dtype={w.dtype}, numel={w.numel()} ({w.numel()} bytes)")

    # Optional PyTorch Check
    if args.check:
        print("🔍 Running PyTorch forward pass...")
        test_in = torch.randn(args.batch, args.channels, min(args.size, 32), min(args.size, 32), dtype=torch.float16)
        with torch.no_grad():
            test_out = model(test_in)
        print(f"   PyTorch test input : {list(test_in.shape)} {test_in.dtype}")
        print(f"   PyTorch test output: {list(test_out.shape)} {test_out.dtype}, mean={test_out.mean():.4f}, std={test_out.std():.4f}")

    # 2. Build MIL Program
    print("\n📐 [Step 2/4] Generating CoreML MIL Program from PyTorch Model...")
    proto = build_coreml_model_from_pytorch(
        model=model,
        batch=args.batch,
        spatial=args.size,
        mode=args.mode
    )
    print("   Built MIL specification with FP8 tensor constants and ops.")

    # 3. Export .mlpackage
    print("\n📦 [Step 3/4] Packaging to .mlpackage...")
    pkg_path = export_mlpackage(proto, args.output)
    print(f"   Saved CoreML package: {pkg_path}")

    # 4. Compile with coremlcompiler
    print("\n⚡ [Step 4/4] Compiling package using coremlcompiler...")
    compiled_path = compile_with_coremlcompiler(pkg_path, output_parent=".")
    print(f"   Compiled binary model: {compiled_path}")

    mil_file = os.path.join(compiled_path, "model.mil")
    if os.path.exists(mil_file):
        print(f"   Generated MIL program: {mil_file} ({os.path.getsize(mil_file)} bytes)")
        if args.dump_mil or args.layers <= 3:
            print("\n------------------- Disassembled model.mil -------------------")
            with open(mil_file, "r") as f:
                lines = f.readlines()
            # If the file is very long due to weights, truncate lines
            for line in lines[:60]:
                if len(line) > 140:
                    print(line[:137] + "...")
                else:
                    print(line, end="")
            if len(lines) > 60:
                print(f"\n... [{len(lines) - 60} more lines in {mil_file}] ...")
            print("--------------------------------------------------------------")

    print("\n✅ Conversion completed successfully!")
    print(f"   Deployable artifact: {compiled_path}")


if __name__ == "__main__":
    main()
