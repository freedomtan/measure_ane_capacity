#!/usr/bin/env python3
"""
convert_pytorch_fp8_to_mil.py

Constructs a PyTorch Convolution model with native FP8 (Float8_e4m3fn) weights,
and converts it directly to a CoreML .mlpackage and compiled MIL program (.mlmodelc / model.mil).
Also compiles to ANE HWX binary format using ~/work/coreml_to_ane_hwx_hacks/mil_to_hwx.
"""

import argparse
import json
import os
import shutil
import struct
import subprocess
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import coremltools as ct
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import types


# ==============================================================================
# 1. PyTorch FP8 Model Definition
# ==============================================================================

class PyTorchFP8ConvModel(nn.Module):
    """
    PyTorch Conv2D Model holding native torch.float8_e4m3fn weight buffers.
    In PyTorch, reference forward pass executes by dequantizing weights to float16 with scale.
    """
    def __init__(self, channels: int = 64, layers: int = 2, kernel_size: int = 3, scale: float = 0.0625):
        super().__init__()
        self.channels = channels
        self.layers = layers
        self.kernel_size = kernel_size
        self.scale = scale

        for i in range(layers):
            # Generate random FP8 weights using PyTorch
            w_float = torch.randn(channels, channels, kernel_size, kernel_size)
            w_fp8 = w_float.to(torch.float8_e4m3fn)
            self.register_buffer(f"weight_fp8_{i}", w_fp8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass in PyTorch using dequantized FP16 weights."""
        pad = self.kernel_size // 2
        for i in range(self.layers):
            w_fp8 = getattr(self, f"weight_fp8_{i}")
            w_fp16 = w_fp8.to(torch.float16) * self.scale
            x = F.conv2d(x, w_fp16, padding=pad)
        return x

    def get_layer_fp8_bytes(self, layer_idx: int) -> bytes:
        """Extracts raw byte representation of the torch.float8_e4m3fn weight tensor."""
        w_fp8 = getattr(self, f"weight_fp8_{layer_idx}")
        return w_fp8.view(torch.uint8).cpu().numpy().tobytes()


# ==============================================================================
# 2. CoreML MIL Program Builder
# ==============================================================================

def build_coreml_model_from_pytorch(
    model: PyTorchFP8ConvModel,
    batch: int = 1,
    spatial: int = 256,
) -> ct.models.MLModel:
    """
    Constructs a CoreML ML Program using constexpr_lut_to_dense palettization.
    This faithfully preserves 1 byte/param weight storage and unpacks with the
    exact 256-entry IEEE Float8_e4m3fn to FP16 conversion table.
    """
    channels = model.channels
    layers = model.layers
    kernel = model.kernel_size
    scale = model.scale

    # Exact IEEE Float8_e4m3fn -> FP16 table, scaled by model.scale
    uint8_indices = torch.arange(256, dtype=torch.uint8)
    fp8_vals = uint8_indices.view(torch.float8_e4m3fn)
    fp16_vals = fp8_vals.to(torch.float16) * scale
    fp16_vals[torch.isnan(fp16_vals)] = 0.0
    lut_np = fp16_vals.numpy().reshape(1, 1, 1, 1, 256, 1)

    in_shape = (batch, channels, spatial, spatial)

    @mb.program(input_specs=[mb.TensorSpec(shape=in_shape, dtype=types.fp16)], opset_version=ct.target.iOS18)
    def prog(x):
        curr = x
        lut_const = mb.const(val=lut_np)
        for i in range(layers):
            w_bytes = model.get_layer_fp8_bytes(i)
            w_uint8 = np.frombuffer(w_bytes, dtype=np.uint8).reshape(channels, channels, kernel, kernel)
            indices_const = mb.const(val=w_uint8)
            w_decomp = mb.constexpr_lut_to_dense(indices=indices_const, lut=lut_const)
            curr = mb.conv(x=curr, weight=w_decomp, pad_type="same", strides=[1, 1], dilations=[1, 1], groups=1)
        return curr

    mlmodel = ct.convert(prog, minimum_deployment_target=ct.target.iOS18)
    return mlmodel


# ==============================================================================
# 3. Compilation & HWX Tools
# ==============================================================================

def compile_with_coremlcompiler(pkg_dir: str, output_parent: str = ".") -> str:
    res = subprocess.run(
        ["xcrun", "coremlcompiler", "compile", pkg_dir, output_parent],
        capture_output=True,
        text=True
    )
    if res.returncode != 0:
        raise RuntimeError(f"coremlcompiler failed (code {res.returncode}):\n{res.stderr}\n{res.stdout}")

    pkg_base = os.path.basename(pkg_dir.rstrip("/"))
    stem = os.path.splitext(pkg_base)[0]
    compiled_dir = os.path.join(output_parent, f"{stem}.mlmodelc")
    return compiled_dir


def compile_with_mil_to_hwx(mlmodelc_path: str, arch: str = "h18p", output_dir: str = "/tmp/hwx_output") -> bool:
    mil_to_hwx_path = os.path.expanduser("~/work/coreml_to_ane_hwx_hacks/mil_to_hwx")
    if not os.path.exists(mil_to_hwx_path):
        print(f"⚠️  mil_to_hwx tool not found at {mil_to_hwx_path}")
        return False

    model_name = os.path.basename(mlmodelc_path.rstrip("/")).replace(".mlmodelc", "")
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        mil_to_hwx_path,
        "-v",
        "-a", arch,
        "-i", mlmodelc_path + "/",
        "-o", output_dir + "/",
        model_name
    ]
    print(f"   Executing: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode == 0 and "Compilation successful" in res.stdout:
        print(f"   ✓ ANE HWX Compilation Successful for target {arch}!")
        analytics_file = os.path.join(output_dir, f"{model_name}_{arch}", "analytics.json")
        if os.path.exists(analytics_file):
            with open(analytics_file) as f:
                data = json.load(f)
            na = data.get("NetworkAnalytics", {})
            print(f"   === Hardware Analytics ({arch}) ===")
            print(f"   Static Procedure Time: {na.get('StaticProcedureTime')} cycles")
            print(f"   NE Frequency          : {na.get('NEFreq')} Hz")
            print(f"   L2 Frequency          : {na.get('L2Freq')} Hz")
            print(f"   DRAM Bandwidth        : {na.get('DramBandwidth')} B/s")
            for grp in data.get("LayerAnalytics", {}).get("Groups", []):
                for task in grp.get("Tasks", []):
                    print(f"   - Task {task.get('TaskID')}: NE Time={task.get('StaticNETime')} cycles, Total={task.get('StaticTotalTime')} cycles")
        return True
    else:
        print(f"   ✗ mil_to_hwx failed (code {res.returncode}):")
        print(res.stdout)
        print(res.stderr)
        return False


# ==============================================================================
# 4. Main CLI Entry Point
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Convert PyTorch FP8 (Float8_e4m3fn) model to CoreML .mlpackage, MIL, and ANE HWX")
    parser.add_argument("--batch", type=int, default=1, help="Batch size (default: 1)")
    parser.add_argument("--channels", type=int, default=64, help="Channel dimension (default: 64)")
    parser.add_argument("--size", type=int, default=256, help="Spatial height/width (default: 256)")
    parser.add_argument("--kernel", type=int, default=3, help="Kernel size (default: 3)")
    parser.add_argument("--layers", type=int, default=2, help="Number of chained conv layers (default: 2)")
    parser.add_argument("--scale", type=float, default=0.0625, help="FP8 scale factor (default: 0.0625 = 2^-4)")
    parser.add_argument("--output", type=str, default="pytorch_fp8_conv", help="Output base path")
    parser.add_argument("--check", action="store_true", help="Run PyTorch forward verification pass")
    parser.add_argument("--dump-mil", action="store_true", help="Print the compiled model.mil program text")
    parser.add_argument("--hwx", action="store_true", help="Compile to ANE HWX binary using mil_to_hwx")
    parser.add_argument("--arch", type=str, default="h18p", help="Target ANE architecture for HWX (default: h18p for iPhone 17 Pro)")

    args = parser.parse_args()

    print("=================================================================")
    print("      PyTorch FP8 (Float8_e4m3fn) to CoreML MIL & HWX Converter  ")
    print("=================================================================")
    print(f"Architecture : Conv2D [{args.channels} -> {args.channels}], K={args.kernel}x{args.kernel}, L={args.layers}")
    print(f"Input Tensor : [{args.batch}, {args.channels}, {args.size}, {args.size}] FP16")
    print(f"Precision    : FP8 (Float8_e4m3fn), Scale: {args.scale}")
    print(f"PyTorch ver  : {torch.__version__}")
    print(f"CoreMLTools  : {ct.__version__}")
    print()

    # 1. Instantiate PyTorch Model
    print("🛠️  [Step 1/5] Constructing PyTorch FP8 Model...")
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

    if args.check:
        print("🔍 Running PyTorch forward pass...")
        test_in = torch.randn(args.batch, args.channels, min(args.size, 32), min(args.size, 32), dtype=torch.float16)
        with torch.no_grad():
            test_out = model(test_in)
        print(f"   PyTorch test output: {list(test_out.shape)} {test_out.dtype}, mean={test_out.mean():.4f}, std={test_out.std():.4f}")

    # 2. Build CoreML Model
    print("\n📐 [Step 2/5] Synthesizing CoreML Model with constexpr_lut_to_dense...")
    mlmodel = build_coreml_model_from_pytorch(model, batch=args.batch, spatial=args.size)
    print("   Built MIL specification with FP8 tensor constants and operations.")

    # 3. Export .mlpackage
    print("\n📦 [Step 3/5] Packaging to .mlpackage...")
    pkg_path = args.output if args.output.endswith(".mlpackage") else f"{args.output}.mlpackage"
    mlmodel.save(pkg_path)
    print(f"   Saved CoreML package: {pkg_path}")

    # 4. Compile with coremlcompiler
    print("\n⚡ [Step 4/5] Compiling package using coremlcompiler...")
    compiled_path = compile_with_coremlcompiler(pkg_path, output_parent=".")
    print(f"   Compiled binary model: {compiled_path}")

    mil_file = os.path.join(compiled_path, "model.mil")
    if os.path.exists(mil_file) and (args.dump_mil or args.layers <= 3):
        print("\n------------------- Disassembled model.mil -------------------")
        with open(mil_file, "r") as f:
            lines = f.readlines()
        for line in lines[:30]:
            print(line, end="")
        if len(lines) > 30:
            print(f"... [{len(lines) - 30} more lines in {mil_file}] ...")
        print("--------------------------------------------------------------")

    # 5. Optional HWX Compilation
    if args.hwx:
        print(f"\n🚀 [Step 5/5] Compiling MIL to ANE HWX for target {args.arch}...")
        compile_with_mil_to_hwx(compiled_path, arch=args.arch)

    print("\n✅ Conversion completed successfully!")
    print(f"   Deployable artifact: {compiled_path}")


if __name__ == "__main__":
    main()
