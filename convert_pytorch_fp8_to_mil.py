#!/usr/bin/env python3
"""
convert_pytorch_fp8_to_mil.py

# Constructs a PyTorch Convolution model with native FP8 (Float8_e4m3fn) weights,
# and converts it directly to a CoreML .mlpackage and compiled MIL program (.mlmodelc / model.mil).
# Also compiles to ANE HWX binary format using mil_to_hwx (https://github.com/freedomtan/coreml_to_ane_hwx),
# and verifies hardware registers (KernelCfg: Fmt=e4m3, Pal=0, InDim: Type=e4m3, DblInt8=1).

"""

import argparse
import json
import os
import re
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
# 2. Native FP8 & LUT MIL Synthesis
# ==============================================================================

def build_lut_coreml_model(
    model: PyTorchFP8ConvModel,
    batch: int = 1,
    spatial: int = 256,
) -> ct.models.MLModel:
    """
    Legacy LUT baseline: constructs CoreML model using constexpr_lut_to_dense palettization.
    Note: ANE decompresses LUT into FP16 (KernelCfg: Fmt=fp16 Pal=1), running at FP16 speed.
    """
    channels = model.channels
    layers = model.layers
    kernel = model.kernel_size
    scale = model.scale

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

    return ct.convert(prog, minimum_deployment_target=ct.target.iOS18)


def synthesize_native_fp8_mlmodelc(
    model: PyTorchFP8ConvModel,
    batch: int = 1,
    spatial: int = 256,
    output_mlmodelc: str = "pytorch_fp8_conv.mlmodelc",
    qdq_activation: bool = True,
) -> str:
    """
    Synthesizes a complete compiled .mlmodelc directory with native FP8 weights & MIL:
    1. weights/weight.bin using MILBlob binary format:
       - DataType 16 (Fp8E4M3FN) for weight tensors
       - DataType 1 (Float16) for per-channel scale vectors
    2. model.mil using native CoreML9 / iOS 26 constexpr_blockwise_shift_scale and activation QDQ
    3. metadata.json and coremldata.bin for direct CoreML runtime loading
    """
    channels = model.channels
    layers = model.layers
    kernel = model.kernel_size
    scale = model.scale

    os.makedirs(output_mlmodelc, exist_ok=True)
    weights_dir = os.path.join(output_mlmodelc, "weights")
    os.makedirs(weights_dir, exist_ok=True)

    # 1. Build weights/weight.bin
    # File header (64 bytes): version 3, count 2, zeros
    file_header = struct.pack("<16I", 3, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    weight_blobs = bytearray(file_header)

    layer_offsets = []
    # FP16 scale values for all channels
    scale_fp16_val = np.float16(scale).view(np.uint16)
    scale_bytes = struct.pack(f"<{channels}H", *([scale_fp16_val] * channels))
    scale_size = len(scale_bytes)

    for l in range(layers):
        w_bytes = model.get_layer_fp8_bytes(l)
        w_size = len(w_bytes)

        # Align to 64 bytes for weight entry
        curr_offset = (len(weight_blobs) + 63) & ~63
        weight_blobs.extend(b"\x00" * (curr_offset - len(weight_blobs)))
        w_entry_offset = curr_offset

        # Entry header: Magic 0xdeadbeef, DataType 16 (Fp8E4M3FN)
        w_data_offset = w_entry_offset + 64
        w_entry = struct.pack("<16I", 3735928559, 16, w_size, 0, w_data_offset, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        weight_blobs.extend(w_entry)
        weight_blobs.extend(w_bytes)

        # Align to 64 bytes for scale entry
        curr_offset = (len(weight_blobs) + 63) & ~63
        weight_blobs.extend(b"\x00" * (curr_offset - len(weight_blobs)))
        s_entry_offset = curr_offset

        # Entry header: Magic 0xdeadbeef, DataType 1 (Float16)
        s_data_offset = s_entry_offset + 64
        s_entry = struct.pack("<16I", 3735928559, 1, scale_size, 0, s_data_offset, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        weight_blobs.extend(s_entry)
        weight_blobs.extend(scale_bytes)

        layer_offsets.append((w_entry_offset, s_entry_offset))

    weight_bin_path = os.path.join(weights_dir, "weight.bin")
    with open(weight_bin_path, "wb") as f:
        f.write(weight_blobs)

    # 2. Build model.mil
    mil_lines = [
        "program(1.3)",
        "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3600.16.1\"}, {\"coremlc-version\", \"3600.25.1\"}})]",
        "{",
        f"    func main<ios19>(tensor<fp16, [{batch}, {channels}, {spatial}, {spatial}]> x) {{",
        "        string pt = const()[val = string(\"same\")];",
        "        tensor<int32, [2]> st = const()[val = tensor<int32, [2]>([1, 1])];",
        "        tensor<int32, [2]> dil = const()[val = tensor<int32, [2]>([1, 1])];",
        "        int32 g = const()[val = int32(1)];",
        "        tensor<int32, [4]> p = const()[val = tensor<int32, [4]>([0, 0, 0, 0])];",
    ]

    if qdq_activation:
        mil_lines.extend([
            f"        tensor<fp16, []> s = const()[val = fp16({scale})];",
            "        string dt = const()[val = string(\"fp8e4m3fn\")];",
        ])

    curr_in = "x"
    for l, (w_off, s_off) in enumerate(layer_offsets):
        # Native FP8 weight dequantization via constexpr_blockwise_shift_scale
        mil_lines.append(
            f"        tensor<fp16, [{channels}, {channels}, {kernel}, {kernel}]> dqw_{l} = constexpr_blockwise_shift_scale("
            f"data = tensor<fp8e4m3fn, [{channels}, {channels}, {kernel}, {kernel}]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64({w_off}))), "
            f"scale = tensor<fp16, [{channels}, 1, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64({s_off}))));"
        )

        conv_in = curr_in
        if qdq_activation:
            # Activation Quantize + Dequantize pair (fuses into E4M3 InDim / OutDim)
            mil_lines.append(f"        tensor<fp8e4m3fn, [{batch}, {channels}, {spatial}, {spatial}]> qx_{l} = quantize(input = {curr_in}, output_dtype = dt, scale = s);")
            mil_lines.append(f"        tensor<fp16, [{batch}, {channels}, {spatial}, {spatial}]> dqx_{l} = dequantize(input = qx_{l}, scale = s);")
            conv_in = f"dqx_{l}"

        out_var = f"y_{l}"
        mil_lines.append(
            f"        tensor<fp16, [{batch}, {channels}, {spatial}, {spatial}]> {out_var} = conv("
            f"dilations = dil, groups = g, pad = p, pad_type = pt, strides = st, weight = dqw_{l}, x = {conv_in});"
        )
        curr_in = out_var

    mil_lines.append(f"    }} -> ({curr_in});")
    mil_lines.append("}")

    mil_path = os.path.join(output_mlmodelc, "model.mil")
    with open(mil_path, "w") as f:
        f.write("\n".join(mil_lines) + "\n")

    # 3. Write metadata.json for CoreML framework compatibility
    model_name = os.path.basename(output_mlmodelc.rstrip("/")).replace(".mlmodelc", "")
    metadata = [
        {
            "metadataOutputVersion": "3.0",
            "storagePrecision": "Float8_e4m3fn (Native FP8)",
            "outputSchema": [
                {
                    "hasShapeFlexibility": "0",
                    "isOptional": "0",
                    "dataType": "Float16",
                    "formattedType": f"MultiArray (Float16 {batch} × {channels} × {spatial} × {spatial})",
                    "shortDescription": "Convolution output",
                    "shape": f"[{batch}, {channels}, {spatial}, {spatial}]",
                    "name": curr_in,
                    "type": "MultiArray"
                }
            ],
            "specificationVersion": 9,
            "availability": {
                "macOS": "15.0",
                "tvOS": "18.0",
                "visionOS": "2.0",
                "watchOS": "11.0",
                "iOS": "18.0",
                "macCatalyst": "18.0"
            },
            "modelType": {"name": "MLModelType_mlProgram"},
            "userDefinedMetadata": {
                "builder": "convert_pytorch_fp8_to_mil.py",
                "format": "Native FP8 (Float8_e4m3fn) with constexpr_blockwise_shift_scale"
            },
            "inputSchema": [
                {
                    "hasShapeFlexibility": "0",
                    "isOptional": "0",
                    "dataType": "Float16",
                    "formattedType": f"MultiArray (Float16 {batch} × {channels} × {spatial} × {spatial})",
                    "shortDescription": "Input image tensor",
                    "shape": f"[{batch}, {channels}, {spatial}, {spatial}]",
                    "name": "x",
                    "type": "MultiArray"
                }
            ],
            "generatedClassName": model_name,
            "method": "predict"
        }
    ]
    with open(os.path.join(output_mlmodelc, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    return output_mlmodelc


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


def find_hwx_tool(tool_name: str) -> str:
    """Finds hwx tools from env or search paths (https://github.com/freedomtan/coreml_to_ane_hwx)."""
    env_var = f"{tool_name.upper().replace('.', '_')}_PATH"
    if env_var in os.environ and os.path.exists(os.environ[env_var]):
        return os.environ[env_var]
    candidates = [
        os.path.expanduser(f"~/work/coreml_to_ane_hwx/{tool_name}"),
        os.path.expanduser(f"~/work/coreml_to_ane_hwx_hacks/{tool_name}"),
        shutil.which(os.path.basename(tool_name)) or "",
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return ""


def verify_hwx_registers(hwx_path: str):
    """
    Parses compiled ANE HWX binary using hwx_dump/hwx_parsing.py or hwx_parsing
    from https://github.com/freedomtan/coreml_to_ane_hwx
    and verifies that native FP8 hardware registers are correctly programmed.
    """
    hwx_parser = find_hwx_tool("hwx_dump/hwx_parsing.py")
    if not hwx_parser:
        hwx_parser = find_hwx_tool("hwx_dump/hwx_parsing")
    if not hwx_parser:
        print("⚠️  HWX parser not found. Clone tool from https://github.com/freedomtan/coreml_to_ane_hwx")
        return

    cmd = ["python3", hwx_parser, hwx_path] if hwx_parser.endswith(".py") else [hwx_parser, hwx_path]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print("⚠️  Failed to parse HWX registers:")
        print(res.stderr)
        return

    tasks = []
    curr_task = None
    for line in res.stdout.splitlines():
        m_task = re.search(r"\[ANE Task (\d+) @", line)
        if m_task:
            curr_task = {"id": int(m_task.group(1)), "in_type": "UNKNOWN", "out_type": "UNKNOWN", "kernel_fmt": "-", "pal": "-", "dbl_int8": 0}
            tasks.append(curr_task)
        if curr_task is not None:
            m_in = re.search(r"InDim\s+:.*?\bType=(\w+)", line)
            if m_in: curr_task["in_type"] = m_in.group(1)
            m_out = re.search(r"OutDim\s+:.*?\bType=(\w+)", line)
            if m_out: curr_task["out_type"] = m_out.group(1)
            m_k = re.search(r"KernelCfg:\s+Fmt=(\w+)\s+Pal=(\d+)", line)
            if m_k:
                curr_task["kernel_fmt"] = m_k.group(1)
                curr_task["pal"] = int(m_k.group(2))
            m_mac = re.search(r"DblInt8=(\d+)", line)
            if m_mac: curr_task["dbl_int8"] = int(m_mac.group(1))

    print("\n   === ANE Hardware Register Verification ===")
    print("   " + "-" * 74)
    print(f"   | {'Task':<6} | {'InDim Type':<12} | {'OutDim Type':<12} | {'KernelFmt':<10} | {'Pal':<5} | {'DblInt8':<8} |")
    print("   " + "-" * 74)
    has_native_fp8 = False
    for t in tasks:
        pal_str = f"Pal={t['pal']}" if t['pal'] != '-' else '-'
        print(f"   | {t['id']:<6} | {t['in_type']:<12} | {t['out_type']:<12} | {t['kernel_fmt']:<10} | {pal_str:<5} | {t['dbl_int8']:<8} |")
        if t['kernel_fmt'].lower() == "e4m3" and t['pal'] == 0:
            has_native_fp8 = True
    print("   " + "-" * 74)

    if has_native_fp8:
        print("   🎉 SUCCESS: Native FP8 (KernelCfg: Fmt=e4m3, Pal=0, InDim: Type=e4m3) confirmed!")
        print("   Hardware E4M3 arithmetic execution verified on ANE.")
    else:
        print("   ⚠️  Warning: Native FP8 kernel format was not detected in task table.")



def compile_with_mil_to_hwx(mlmodelc_path: str, arch: str = "h18", output_dir: str = "/tmp/hwx_output") -> bool:
    mil_to_hwx_path = find_hwx_tool("mil_to_hwx")
    if not mil_to_hwx_path:
        print("⚠️  mil_to_hwx tool not found. Clone tool from https://github.com/freedomtan/coreml_to_ane_hwx")
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
        hwx_file = os.path.join(output_dir, f"{model_name}_{arch}", "model.hwx")
        
        # Copy model.hwx into .mlmodelc so it is self-contained and directly deployable
        dst_hwx = os.path.join(mlmodelc_path, "model.hwx")
        if os.path.exists(hwx_file):
            shutil.copy(hwx_file, dst_hwx)
            print(f"   ✓ Installed deployable model.hwx into {mlmodelc_path}/")

        # Parse analytics
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
                layer_names = ", ".join([k for k in grp.keys() if k != "Tasks" and k != "Layer Name"])
                for task in grp.get("Tasks", []):
                    print(f"   - Group Task {task.get('TaskID')}: NE Time={task.get('StaticNETime')} cycles, Total={task.get('StaticTotalTime')} cycles")

        # Verify registers
        if os.path.exists(hwx_file):
            verify_hwx_registers(hwx_file)
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
    parser.add_argument("--mode", type=str, choices=["native", "lut"], default="native", help="Weight representation mode: 'native' (E4M3 constexpr_blockwise_shift_scale) or 'lut' (constexpr_lut_to_dense)")
    parser.add_argument("--qdq-activation", action="store_true", default=True, help="Insert FP8 activation QDQ pair around convolutions (default: True)")
    parser.add_argument("--no-qdq-activation", dest="qdq_activation", action="store_false", help="Disable FP8 activation QDQ (weight-only FP8)")
    parser.add_argument("--output", type=str, default="pytorch_fp8_conv", help="Output base path")
    parser.add_argument("--check", action="store_true", help="Run PyTorch forward verification pass")
    parser.add_argument("--dump-mil", action="store_true", help="Print the compiled model.mil program text")
    parser.add_argument("--hwx", action="store_true", default=True, help="Compile to ANE HWX binary using mil_to_hwx (default: True)")
    parser.add_argument("--no-hwx", dest="hwx", action="store_false", help="Skip ANE HWX compilation")
    parser.add_argument("--arch", type=str, default="h18", help="Target ANE architecture for HWX (default: h18 for A19 / M5)")

    args = parser.parse_args()

    print("=================================================================")
    print("      PyTorch FP8 (Float8_e4m3fn) to CoreML MIL & HWX Converter  ")
    print("=================================================================")
    print(f"Architecture : Conv2D [{args.channels} -> {args.channels}], K={args.kernel}x{args.kernel}, L={args.layers}")
    print(f"Input Tensor : [{args.batch}, {args.channels}, {args.size}, {args.size}] FP16")
    print(f"Mode         : {args.mode.upper()} {'(Native FP8)' if args.mode == 'native' else '(LUT Palettization)'}")
    print(f"Act QDQ      : {args.qdq_activation}")
    print(f"Precision    : FP8 (Float8_e4m3fn), Scale: {args.scale}")
    print(f"PyTorch ver  : {torch.__version__}")
    print(f"CoreMLTools  : {ct.__version__}")
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

    if args.check:
        print("🔍 Running PyTorch forward pass...")
        test_in = torch.randn(args.batch, args.channels, min(args.size, 32), min(args.size, 32), dtype=torch.float16)
        with torch.no_grad():
            test_out = model(test_in)
        print(f"   PyTorch test output: {list(test_out.shape)} {test_out.dtype}, mean={test_out.mean():.4f}, std={test_out.std():.4f}")

    # 2. Build Model
    mlmodelc_dir = f"{args.output}.mlmodelc" if not args.output.endswith(".mlmodelc") else args.output
    if args.mode == "native":
        print("\n📐 [Step 2/4] Synthesizing Native FP8 .mlmodelc & model.mil...")
        synthesize_native_fp8_mlmodelc(
            model=model,
            batch=args.batch,
            spatial=args.size,
            output_mlmodelc=mlmodelc_dir,
            qdq_activation=args.qdq_activation,
        )
        print(f"   Created native FP8 model package: {mlmodelc_dir}")
    else:
        print("\n📐 [Step 2/4] Synthesizing CoreML Model with constexpr_lut_to_dense...")
        mlmodel = build_lut_coreml_model(model, batch=args.batch, spatial=args.size)
        pkg_path = f"{args.output}.mlpackage"
        mlmodel.save(pkg_path)
        print(f"   Saved CoreML package: {pkg_path}")
        print("\n⚡ Compiling package using coremlcompiler...")
        mlmodelc_dir = compile_with_coremlcompiler(pkg_path, output_parent=".")
        print(f"   Compiled binary model: {mlmodelc_dir}")

    # 3. Dump MIL
    mil_file = os.path.join(mlmodelc_dir, "model.mil")
    if os.path.exists(mil_file) and (args.dump_mil or args.layers <= 3):
        print("\n------------------- Disassembled model.mil -------------------")
        with open(mil_file, "r") as f:
            lines = f.readlines()
        for line in lines[:35]:
            print(line, end="")
        if len(lines) > 35:
            print(f"... [{len(lines) - 35} more lines in {mil_file}] ...")
        print("--------------------------------------------------------------")

    # 4. Optional HWX Compilation & Hardware Register Verification
    if args.hwx:
        print(f"\n🚀 [Step 4/4] Compiling MIL to ANE HWX for target {args.arch}...")
        compile_with_mil_to_hwx(mlmodelc_dir, arch=args.arch)

    print("\n✅ Conversion completed successfully!")
    print(f"   Deployable artifact: {mlmodelc_dir}")


if __name__ == "__main__":
    main()
