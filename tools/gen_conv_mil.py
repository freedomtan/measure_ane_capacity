#!/usr/bin/env python3
"""Generate CoreML .mlpackage models for ANE convolution capacity benchmarking.

The graph is authored directly in MIL (Model Intermediate Language) via
coremltools' Builder, mirroring the MPSGraph workload in measure_conv_universal.m:
L chained 3x3 convolutions over an [B, Ci, H, W] tensor, NCHW layout, SAME padding.

H17 and later silicon implements hardware zero-skipping and lossless zero-
compression, so zero-valued tensors bypass the MAC arrays and inflate measured
TOPS. Two weight initializations are available:

  --weights dense  (default) random-sign +/-0.03125, RMS-scaled so activation
                   magnitude stays roughly constant across L layers.
  --weights repeat the tiled +/-0.0625, +/-0.03125 pattern used by
                   fillNonZeroData() in the Objective-C/MPSGraph benchmarks.

Note that `repeat` is degenerate: the alternating pattern cancels exactly under
the convolution reduction, so layer 1 outputs all zeros and every later layer
consumes a zero tensor -- exactly the condition zero-skipping detects. Use it
only for bit-comparability with the MPSGraph binaries.
"""

import argparse
import os
import shutil
import sys

import numpy as np

try:
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types
except ImportError:
    sys.exit("error: coremltools is required (pip3 install coremltools)")

# Same pattern as fillNonZeroData() for MPSDataTypeFloat16: fp16 bit patterns
# 0x2C00, 0xAC00, 0x2800, 0xA800.
FP16_PATTERN = np.array([0.0625, -0.0625, 0.03125, -0.03125], dtype=np.float16)

# Fixed seed so generated models are reproducible.
WEIGHT_SEED = 0x5EED

# 1/32, exactly representable in fp16 (0x2800). With random signs over a
# reduction of Ci*K*K terms the per-layer gain is sqrt(Ci*K*K)/32, which for
# Ci=128, K=3 is ~1.06 -- activations stay in fp16 range across 20+ layers.
DENSE_MAGNITUDE = np.float16(0.03125)


def fill_repeat(shape):
    """Tile FP16_PATTERN across `shape` in C order, as the ObjC code does."""
    count = int(np.prod(shape))
    flat = np.resize(FP16_PATTERN, count)
    return flat.reshape(shape)


def fill_dense(shape):
    """Random-sign constant-magnitude weights: non-zero and non-cancelling."""
    rng = np.random.default_rng(WEIGHT_SEED)
    signs = rng.integers(0, 2, size=shape).astype(np.float16) * 2 - 1
    return (signs * DENSE_MAGNITUDE).astype(np.float16)


def build_program(B, Ci, Co, H, W, K, L, opset, weight_mode):
    weights = fill_dense((Co, Ci, K, K)) if weight_mode == "dense" \
        else fill_repeat((Co, Ci, K, K))

    @mb.program(
        input_specs=[mb.TensorSpec(shape=(B, Ci, H, W), dtype=types.fp16)],
        opset_version=opset,
    )
    def prog(x):
        # One shared weight constant across all layers, as in the MPSGraph version.
        w = mb.const(val=weights, name="weights")
        for i in range(L):
            x = mb.conv(
                x=x,
                weight=w,
                strides=[1, 1],
                pad_type="same",
                dilations=[1, 1],
                groups=1,
                name=f"conv_{i}",
            )
        return x

    return prog


def count_conv_ops(mlmodel):
    """Count conv ops in the converted model's MIL program."""
    program = mlmodel.get_spec().mlProgram
    total = 0
    for func in program.functions.values():
        for op in func.block_specializations[func.opset].operations:
            if op.type == "conv":
                total += 1
    return total


def mil_text(mlmodel):
    program = mlmodel.get_spec().mlProgram
    lines = []
    for name, func in program.functions.items():
        block = func.block_specializations[func.opset]
        lines.append(f"function {name} (opset {func.opset}):")
        for op in block.operations:
            inputs = ", ".join(
                f"{k}={[a.name for a in v.arguments]}" for k, v in op.inputs.items()
            )
            outputs = ", ".join(o.name for o in op.outputs)
            lines.append(f"  {outputs} = {op.type}({inputs})")
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(
        description="Generate MIL-authored CoreML conv models for ANE benchmarking"
    )
    p.add_argument("--batch", type=int, default=1, help="batch dimension B")
    p.add_argument("--size", type=int, default=256, help="spatial height and width H=W")
    p.add_argument("--channels", type=int, default=128, help="input/output channels Ci=Co")
    p.add_argument("--kernel", type=int, default=3, help="kernel size K")
    p.add_argument("--layers", type=int, default=20, help="number of chained conv layers")
    p.add_argument("--variant", default="fp16", choices=["fp16"], help="precision variant")
    p.add_argument(
        "--weights",
        default="dense",
        choices=["dense", "repeat"],
        help="weight init: dense (random-sign, non-cancelling) or repeat "
        "(tiled pattern matching the MPSGraph binaries; cancels to zero)",
    )
    p.add_argument("--out", default="models", help="output directory")
    p.add_argument(
        "--no-default-copy",
        action="store_true",
        help="skip writing the unsuffixed conv_<variant>.mlpackage copy",
    )
    args = p.parse_args()

    B, H, W = args.batch, args.size, args.size
    Ci = Co = args.channels
    K, L = args.kernel, args.layers

    opset = ct.target.iOS18
    print(
        f"Building MIL program: [{B}, {Ci}, {H}, {W}] -> {L}x conv{K}x{K} "
        f"(Co={Co}), fp16, NCHW/SAME, weights={args.weights}"
    )
    if args.weights == "repeat":
        print(
            "warning: 'repeat' weights cancel exactly under the conv reduction; "
            "layers 2..L will consume an all-zero tensor."
        )
    prog = build_program(B, Ci, Co, H, W, K, L, opset, args.weights)

    # FP16 input and output types: an FP32 boundary would force CoreML to cast a
    # multi-MB tensor on the CPU every prediction, swamping the measurement.
    mlmodel = ct.convert(
        prog,
        minimum_deployment_target=opset,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )

    # Guard against an optimization pass silently collapsing the chain: we must
    # not report TOPS for a graph we did not author.
    found = count_conv_ops(mlmodel)
    if found != L:
        sys.exit(
            f"error: expected {L} conv ops after conversion, found {found}. "
            "A coremltools pass altered the graph; TOPS would be wrong."
        )
    print(f"Verified {found}/{L} conv ops survived conversion")

    os.makedirs(args.out, exist_ok=True)
    # Only the non-default weight mode is tagged, so the common case reads cleanly.
    tag = "" if args.weights == "dense" else f"_{args.weights}"
    stem = f"conv_{args.variant}{tag}_B{B}_C{Ci}_H{H}_K{K}_L{L}"
    pkg = os.path.join(args.out, stem + ".mlpackage")
    if os.path.exists(pkg):
        shutil.rmtree(pkg)
    mlmodel.save(pkg)

    mil_path = os.path.join(args.out, stem + ".mil.txt")
    with open(mil_path, "w") as f:
        f.write(mil_text(mlmodel))

    print(f"Wrote {pkg}")
    print(f"Wrote {mil_path}")

    # Stable unsuffixed name so measure_conv_coreml runs with no arguments.
    if not args.no_default_copy:
        default_pkg = os.path.join(args.out, f"conv_{args.variant}{tag}.mlpackage")
        if os.path.exists(default_pkg):
            shutil.rmtree(default_pkg)
        shutil.copytree(pkg, default_pkg)
        print(f"Wrote {default_pkg}")


if __name__ == "__main__":
    main()
