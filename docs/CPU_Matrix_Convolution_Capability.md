# CPU Matrix Multiplication / Convolution Capability

**Scope:** this repo's ANE/GPU benchmarks (`measure_matmul_universal.m`, `measure_conv_universal.m`,
`measure_conv_coreml.m`) have no CPU baseline. This investigates two possible routes to one —
MPSGraph's private `CPUDevice` SPI, and CoreML/MIL's public `MLComputeUnitsCPUOnly` — before
reaching for lower-level APIs (Accelerate BLAS/BNNS, raw NEON).

**Environment:** macOS 27.0 / Xcode 27, Apple M4 Pro (same unit as the README's baseline numbers).

## Finding 1 — `+[MPSGraphDevice CPUDevice]` exists but crashes unconditionally on compile

`MPSGraphDevice` ships a private `+CPUDevice` class method, parallel to the `+ANEDevice` method
`measure_matmul_universal.m`/`measure_conv_universal.m` already used (confirmed via a cached
header dump: `/Users/freedom/work/mil/MPSGraphDevice.h`):

```objc
/* class methods */
+[MPSGraphDevice deviceWithMTLDevice:];
+[MPSGraphDevice CPUDevice];
+[MPSGraphDevice ANEDevice];
```

Both `measure_matmul_universal.m` and `measure_conv_universal.m` were extended with a `kTargetCPU`
option that builds the graph identically to the GPU/ANE paths but passes `[MPSGraphDevice CPUDevice]`
as the compile-time device. Running either benchmark against this target crashes, every time,
with the identical stack trace regardless of op (matmul or conv) or optimization level:

```
*** Terminating app due to uncaught exception 'NSInvalidArgumentException', reason:
'*** -[__NSCFConstantString stringByAppendingString:]: nil argument'
*** First throw call stack:
...
2   Foundation                      NSCreateMapTable + 0
3   MetalPerformanceShadersGraph    -[MPSGraphExecutable applyOptimizationPassesWithDevice:module:
                                      compilationID:compilationDescriptor:perEntryPointFuncOpMLIRName:] + 7980
4   MetalPerformanceShadersGraph    -[MPSGraphExecutable specializedModuleWithDevice:shapedEntryPoints:
                                      compilationDescriptor:error:] (block) + 3692
5   MetalPerformanceShadersGraph    -[MPSGraphExecutable specializedModuleWithDevice:shapedEntryPoints:
                                      compilationDescriptor:error:] + 4808
6   MetalPerformanceShadersGraph    -[MPSGraphExecutable specializeWithDevice:shapedEntryPoints:
                                      compilationDescriptor:error:] (block) + 72
7-14 libdispatch.dylib / libsystem_pthread.dylib (background GCD worker thread)
```

**Not shape-specific, not optimization-level-specific.** Tried `MPSGraphOptimizationLevel0` to skip
whatever `applyOptimizationPassesWithDevice:` normally does — same crash, same offset (`+7980`),
confirming the method runs unconditionally during specialization regardless of the requested
optimization level. Same exact stack for both the matmul and the convolution benchmark.

**Likely root cause (not confirmed by disassembly): a nil `metalDeviceName`.** `MPSGraphDevice`'s
`metalDevice`/`metalDeviceName` properties are presumably nil for a `CPUDevice` instance (it has no
backing `MTLDevice`), and something inside this compile/specialization pass builds a cache key or
debug label by `-stringByAppendingString:` against that name, which throws instead of handling the
nil case. This is a hypothesis, not verified via disassembly here — would need the same
`ipsw macho disass` treatment the main MXU investigation doc used on `MPSNDArray`/`MetalPerformanceShadersGraph`.

**Not catchable in-process.** The exception is thrown from a background GCD worker thread inside
`compileWithDevice:`'s own internals (`_dispatch_lane_invoke`/`_dispatch_workloop_worker_thread`),
not the calling thread — wrapping the call in `@try`/`@catch` on the caller's thread does nothing;
the whole process still aborts. Both benchmarks were restructured with a `--target gpu|ane|cpu` flag
so each target runs as its own subprocess, isolating the CPU crash from the GPU/ANE results.

**Conclusion: this SPI, on this OS/MPSGraph build, is not a usable path to a CPU baseline.**
Confirmed results on this M4 Pro, GPU/ANE unaffected by the refactor:

| Workload | GPU FP16 | ANE FP16 | ANE INT8 | MPSGraph CPU |
|---|---|---|---|---|
| Matmul 1024×1024×1024, L=20 | 6.13–6.30 TOPS | 15.68–16.58 TOPS | 16.58 TOPS | crashes |
| Conv [1,128,256,256]/3×3, L=20 | 9.92 TOPS | 19.10 TOPS | 37.21 TOPS | crashes |

## Finding 2 — CoreML/MIL's `MLComputeUnitsCPUOnly` works cleanly and gives a real number

`measure_conv_coreml.m` already supported `--units cpu` (`MLComputeUnitsCPUOnly`) from the
original CoreML/MIL benchmark work — it was simply never run against that target before. Running
it:

```
$ ./measure_conv_coreml --units cpu
CoreML Conv2D Capacity Benchmark (FP16)
Model: built at runtime by MILSpecBuilder (weights=dense)
Input: dense (random-sign)
Workload: [1, 128, 256, 256] x 20 conv3x3 (Co=128), 386.55 GOPs/pass, 20 iterations
[CoreML CPU FP16] Compile: 45.95 ms, Load: 465.55 ms, Avg: 125.26 ms, Speed: 3.0860 TOPS (8.0 FPS)
```

No crash, no code change needed — just the first time anyone pointed this tool at `cpu`. **3.09
TOPS is now this repo's first real CPU convolution baseline**, ~3.2x slower than GPU FP16 (9.92
TOPS) and ~6.2x slower than ANE FP16 (19.10 TOPS) on the same chip, same workload shape.

## Finding 3 — the same bug reproduces on iOS (A18 Pro, A20 Pro), but at a different pipeline stage

Ported the macOS CLI's `kTargetCPU` path into `ANECapacityApp` as `NativeCPUMatmulConvProbe.swift`
(`--native-cpu-matmul` / `--native-cpu-conv`), using the identical `+[MPSGraphDevice CPUDevice]`
SPI call via the same raw-`IMP` pattern used throughout this app's other probes. Tested on both
physical devices available: an iPhone 16 Pro (A18 Pro) and an iPhone 18 Pro (A20 Pro).

**First attempt, full-size shapes (matmul 1024³×20, matching the macOS CLI): no crash, no output,
no completion for several minutes.** Forced `setvbuf(stdout, nil, _IONBF, 0)` to rule out output
buffering as the explanation — still nothing printed, including the very first `print()` call,
until the process was manually terminated. This was ambiguous: either a true hang, or just an
extremely slow compile with nothing to show for it in between.

**Shrinking the shapes resolved the ambiguity: it's slow, not hung, and the crash moves to a
different stage than on macOS.** At trivial sizes (64×64×64×1 matmul, 32×32×8ch 3×3×1 conv), both
devices reliably reproduce a crash — but **the crash is no longer inside the compile call**.
`-[MPSGraphExecutable compileWithDevice:...]` returns a real, non-nil executable; the identical
`NSInvalidArgumentException`/`-stringByAppendingString:: nil argument` crash instead happens
during `-[MPSGraphExecutable runWithMTLCommandQueue:inputsArray:resultsArray:executionDescriptor:]`:

```
[CPUProbe] Device: Apple A18 Pro GPU, targeting MPSGraphDevice.CPUDevice for matmul
[CPUProbe] Compiling matmul graph for CPUDevice...
[CPUProbe] Compile returned without crashing: <MPSGraphExecutable: 0x11a6c5400>
*** Terminating app due to uncaught exception 'NSInvalidArgumentException', reason:
'*** -[__NSCFConstantString stringByAppendingString:]: nil argument'
```

Identical on all four combinations tried (A18 Pro × {matmul, conv}, A20 Pro × {matmul, conv}) —
same exception class, same message, same "compile succeeds, run crashes" ordering, differing only
in raw unsymbolicated addresses (no on-device dSYM to resolve them to function names, unlike the
macOS case where the crash reporter had full symbols).

**Later confirmed on a third chip, an iPhone 17 Pro (A19 Pro):** identical crash, identical
ordering, for both matmul and conv:

```
[CPUProbe] Device: Apple A19 Pro GPU, targeting MPSGraphDevice.CPUDevice for matmul
[CPUProbe] Compiling matmul graph for CPUDevice...
[CPUProbe] Compile returned without crashing: <MPSGraphExecutable: 0x10c6ed400>
*** Terminating app due to uncaught exception 'NSInvalidArgumentException', reason:
'*** -[__NSCFConstantString stringByAppendingString:]: nil argument'
```

**Interpretation:** this is very likely the same underlying defect as Finding 1 (something
building a cache key or debug label off a nil `metalDeviceName`-shaped string for a `CPUDevice`
instance), but iOS's MPSGraph build apparently defers whatever triggers it from compile-time to
run-time, and/or the macOS crash's `applyOptimizationPassesWithDevice:` code path legitimately
takes much longer at full problem size on these phones' CPUs before failing — the full-size,
several-minutes-with-no-output run was never confirmed to be either "still compiling" or "would
crash eventually," since it was terminated rather than left to run further. Likely compounds with
the fact that MPSGraph's CPU "device" has to synthesize working code on a path presumably never
performance-tuned or perhaps even regularly exercised by Apple's own internal callers (Finding 2's
CoreML/MIL route being the one that actually works).

**Net result: `+[MPSGraphDevice CPUDevice]` is confirmed non-functional across all four physical
chips tested in this repo (M4 Pro, A18 Pro, A19 Pro, A20 Pro) and both benchmark shapes (matmul,
conv).** Finding 2's CoreML/MIL route remains the only working CPU path.

## Finding 4 — resolved: `MILSpecBuilder.mm` now has a `matmul` chain builder, and `measure_matmul_coreml.m` gives a real CPU matmul number

Added `MILMatmulChainConfig`/`MILBuildMatmulChainSpec`/`MILMatmulChainText` to `MILSpecBuilder.mm`
(FP16 only — no quantized variant yet, unlike the conv chain), and a new `measure_matmul_coreml.m`
CLI matching `measure_conv_coreml.m`'s `--units`/`--plan`/`--check`/`--dump-mil`/`--save`
conventions.

**One real surprise building it:** `matmul`'s `transpose_x`/`transpose_y` parameters are documented
as "optional, default `False`" — true only at coremltools' Python `mb.matmul()` builder level. The
raw compiled `MLProgram` rejects the op outright without them: `Required param 'transpose_y' is
missing`. Fixed with an explicit shared `no_transpose` `BOOL` const bound to both parameters.

Verified correct with a standalone test harness (hand-computed a 2×2 single-layer matmul against
the model's own deterministic weight-generation seed; result matched exactly, max diff `0.0`) before
trusting it enough to benchmark. Then ran the real workload (`[1,1,1024,1024] x [1,1,1024,1024]`,
20 layers, matching `measure_matmul_universal.m`'s shapes) across all three CoreML compute units on
this M4 Pro, with `--check` confirming finite, non-degenerate output on each:

```
[CoreML ANE FP16] Compile: 122.01 ms, Load: 10606.14 ms, Avg: 2.84 ms, Speed: 15.1109 TOPS (351.8 FPS)
[CoreML GPU FP16] Compile: 122.45 ms, Load: 5978.17 ms,  Avg: 7.01 ms, Speed: 6.1261 TOPS (142.6 FPS)
[CoreML CPU FP16] Compile: 122.49 ms, Load: 2028.27 ms,  Avg: 6.49 ms, Speed: 6.6172 TOPS (154.1 FPS)
```

`--plan` confirms the ANE run genuinely places all 20/20 `matmul` ops on the Neural Engine, not a
silent CPU/GPU fallback. **6.62 TOPS is this repo's first real CPU matmul baseline** — and,
surprisingly, CoreML's CPU matmul here is *faster* than its GPU matmul (6.62 vs 6.13 TOPS), unlike
the conv case where CPU (3.09 TOPS) was clearly slowest. Not investigated further; plausibly matmul
being a much more regular, cache-friendly access pattern than a strided 3×3/128-channel convolution,
favoring whatever CPU GEMM path CoreML dispatches to (Accelerate's BLAS/AMX path is the leading
guess, not confirmed).

Updated results table (both ops, all three CoreML compute units, this M4 Pro):

| Workload | ANE FP16 | GPU FP16 | CPU FP16 |
|---|---|---|---|
| Matmul 1024³×L20 | 15.11 TOPS | 6.13 TOPS | **6.62 TOPS** |
| Conv 256²/128ch/3×3×L20 | — | — | 3.09 TOPS |

(ANE/GPU conv-via-CoreML cells intentionally left blank — not re-measured in this pass; the
MPSGraph ANE/GPU numbers from the main table earlier in this doc are the trustworthy ones for
those two cells.)

## Finding 5 — the CoreML/MIL matmul path works on real iPhones too, all three chips

Ported the matmul chain builder into `ANECapacityApp`'s own, independent `MILSpecBuilder.m` — a
pure-Objective-C, zero-dependency reimplementation that hand-encodes the protobuf wire format
directly (no C++ protobuf library, which can't go into an iOS app bundle the way the repo-root
`MILSpecBuilder.mm` does). Added `MILMatmulChainConfig`/`MILBuildMatmulChainSpec`/
`MILMatmulChainInputName`/`OutputName`, mirroring the C++ version field-for-field (same MIL.proto
schema, so the same `const`/`matmul` op structure, the same `no_transpose` `BOOL` const
work-around from Finding 4). A new `NativeCoreMLMatmulProbe.swift`
(`--native-coreml-matmul`) builds the spec, compiles it via `MLModel.compileModel(at:)`, and
benchmarks all three CoreML compute units in one run — unlike the `MPSGraphDevice.CPUDevice`
probes, CoreML doesn't crash on any of them, so no per-target process isolation is needed here.

Ran on all three physical iPhones available:

```
# iPhone 16 Pro (A18 Pro)
[CoreML ANE FP16] Load: 14077.23 ms, Avg: 3.35 ms,  Speed: 12.8122 TOPS (298.3 FPS)
[CoreML GPU FP16] Load: 9336.53 ms,  Avg: 28.61 ms, Speed: 1.5013 TOPS (35.0 FPS)
[CoreML CPU FP16] Load: 2645.87 ms,  Avg: 15.60 ms, Speed: 2.7526 TOPS (64.1 FPS)

# iPhone 17 Pro (A19 Pro)
[CoreML ANE FP16] Load: 12807.51 ms, Avg: 3.34 ms,  Speed: 12.8659 TOPS (299.6 FPS)
[CoreML GPU FP16] Load: 8268.15 ms,  Avg: 6.97 ms,  Speed: 6.1613 TOPS (143.5 FPS)
[CoreML CPU FP16] Load: 2452.38 ms,  Avg: 13.40 ms, Speed: 3.2048 TOPS (74.6 FPS)

# iPhone 18 Pro (A20 Pro)
[CoreML ANE FP16] Load: 13374.08 ms, Avg: 2.40 ms,  Speed: 17.9289 TOPS (417.4 FPS)
[CoreML GPU FP16] Load: 6157.91 ms,  Avg: 4.67 ms,  Speed: 9.2017 TOPS (214.2 FPS)
[CoreML CPU FP16] Load: 1759.73 ms,  Avg: 11.05 ms, Speed: 3.8883 TOPS (90.5 FPS)
```

All nine runs (three chips × three compute units) report finite, non-degenerate output (`--check`-
equivalent output-range logging built into the probe). **First real CPU matmul numbers on actual
iPhone silicon: 2.75 TOPS (A18 Pro), 3.20 TOPS (A19 Pro), 3.89 TOPS (A20 Pro)** — a clean
monotonic increase across generations.

Notably, the M4 Pro's "CPU beats GPU at matmul" surprise from Finding 4 does **not** reproduce on
any of the three iPhones — on all three, GPU comfortably beats CPU, the ordering you'd expect
(ANE > GPU > CPU). A18 Pro's GPU number (1.50 TOPS) stands out as unusually low relative to A19 Pro
and A20 Pro's GPU numbers (6.16, 9.20 TOPS) despite A18 Pro's own ANE/CPU numbers being roughly
in line with A19 Pro's — plausibly compile/specialization overhead specific to that run (A18 Pro's
GPU load time, 9.3s, is disproportionately long relative to its own avg latency) rather than a
genuine generational throughput cliff, but not confirmed. The M4 Pro's CPU-beats-GPU result should
be treated as a Mac-specific or small-sample artifact, not a general "CoreML CPU matmul beats GPU
matmul" rule — it is the only platform of the four tested where that ordering appears.

Also confirmed on this third chip: the `MPSGraphDevice.CPUDevice` crash from Finding 3 reproduces
identically on A19 Pro, for both matmul and conv (same exception, same "compile succeeds, run
crashes" ordering) — see Finding 3's update above.

Final results table, all platforms and both ops:

| Workload | Platform | ANE FP16 | GPU FP16 | CPU FP16 |
|---|---|---|---|---|
| Matmul 1024³×L20 | M4 Pro (Mac) | 15.11 TOPS | 6.13 TOPS | 6.62 TOPS |
| Matmul 1024³×L20 | A18 Pro (iPhone 16 Pro) | 12.81 TOPS | 1.50 TOPS | 2.75 TOPS |
| Matmul 1024³×L20 | A19 Pro (iPhone 17 Pro) | 12.87 TOPS | 6.16 TOPS | 3.20 TOPS |
| Matmul 1024³×L20 | A20 Pro (iPhone 18 Pro) | 17.93 TOPS | 9.20 TOPS | 3.89 TOPS |
| Conv 256²/128ch/3×3×L20 | M4 Pro (Mac) | — | — | 3.09 TOPS |

## Finding 6 — does the CoreML CPU matmul path use AMX, SME2, or plain NEON?

(There is no SSE2 on Apple Silicon at all — that's an x86 Intel ISA. The real question is whether
the CPU GEMM reaches Apple's private AMX matrix coprocessor, ARM's newer official SME/SME2
extension, or falls back to plain NEON SIMD loops.)

**Confirmed the call path with `sample`:** ran `measure_matmul_coreml --units cpu` with a long
iteration count in the background and sampled the live process for 3 seconds. Every hot-path
stack bottoms out at:

```
main -> runBenchmark -> ... -> E5RT::Ops::BnnsCpuInferenceOperation::...
       -> BNNSGraphContextExecute_v2 (libBNNS.dylib) -> ??? (stripped, in libBNNS.dylib)
```

So CoreML's CPU compute unit, for this workload, really does go through Accelerate's BNNS
(Basic Neural Network Subroutines), not a generic/naive loop. The deepest frames inside
`libBNNS.dylib` are stripped of symbol names (`???` in the sample output), so the exact leaf
kernel wasn't directly visible this way.

**Found the real answer in `libBNNS.dylib`'s own string literals.** Disassembled the library
(the standalone, non-shared-cache copy at
`.../ios-hacking/disassm/extracted/.../vecLib.framework/Versions/A/libBNNS.dylib`, via
`otool -tv`, ~4.5M lines) and grepped for every `"KERNEL_..."` string — these are BNNS's internal
kernel-selection enum names, baked in as literal strings for logging/error messages. For
GEMM/MatMul specifically:

```
KERNEL_GEMM_NEON / KERNEL_GEMM_NEON_DYNAMIC
KERNEL_GEMM_MLA1 / MLA1_DYNAMIC
KERNEL_GEMM_MLA2 / MLA2_DYNAMIC
KERNEL_GEMM_MLA3 / MLA3_DYNAMIC
KERNEL_GEMM_SME2 / SME2_DYNAMIC
KERNEL_GEMM_INTEL64 / INTEL64_DYNAMIC        (Rosetta/x86 fallback, irrelevant here)
KERNEL_MATMUL_{NEON,MLA1,MLA2,MLA3,SME2,SME2P1,INTEL64}[_DYNAMIC]
KERNEL_SPARSE_MATMUL_{NEON,AMX2,AMX3,SME2P1}[_DYNAMIC]
```

The sparse-matmul family names its matrix-coprocessor tiers **`AMX2`/`AMX3`** explicitly; other
strings elsewhere in the binary reference `"combo_dodge_v2_amx3"`,
`"BNNS transpose2d_amx1/2/3, unsupported data types"`, and
`"dynamic_prep_and_execute_copy ... dispatching of AMX copies"`. The dense GEMM/MatMul family uses
a **different label, `MLA1`/`MLA2`/`MLA3`**, for what is almost certainly the same hardware
block under a different internal name (the numbering lines up with the sparse path's `AMX2`/`AMX3`,
and `MLA1` for an even older/first-generation tier) — not confirmed by name alone, but strongly
suggested by the parallel structure. **So: yes, BNNS has a real, named, non-NEON matrix-coprocessor
kernel tier for GEMM — it isn't just NEON SIMD loops.**

**This chip (M4 Pro) also exposes ARM's official SME/SME2 extension, confirmed via `sysctl`:**

```
hw.optional.arm.FEAT_SME: 1
hw.optional.arm.FEAT_SME2: 1
hw.optional.arm.FEAT_SME2p1: 0
hw.optional.neon: 1
```

No equivalent public sysctl exists for AMX (expected — Apple has never documented or exposed it,
consistent with the entire private-SPI theme of this repo's other findings). Given `KERNEL_GEMM_SME2`
exists as a named tier and this hardware genuinely has `FEAT_SME2`, the most likely real answer for
this specific chip is that dense FP16 GEMM here runs through the **public SME2** path rather than
the private legacy AMX tiers (`MLA1`/`MLA2`/`MLA3` plausibly exist for M1–M3-era chips that predate
SME2 silicon) — but this is an informed inference from kernel-naming conventions and hardware
capability bits, **not a direct runtime observation of which kernel actually got dispatched**.

**Closed out: confirmed SME2 is genuinely dispatched, both statically and live via `lldb`.**

Took a `sample` capture of a sustained `measure_matmul_coreml --units cpu` run (3000 iterations,
6 s sampling window) and walked the full stack past the stripped `???` frames by resolving file
offsets against the standalone `libBNNS.dylib` copy's own function-start table (scanned for
`pacibsp` prologues, since `nm`'s exported-symbol table is too sparse — most of these are static
functions with no exported name at all). The hot leaf stack bottoms out in one ~19 KB unnamed
function (`libBNNS.dylib + 0xa73ca4`, called from a small dispatcher at `+0xa7330c`, itself called
from `BNNSGraphContextExecute_v2` through a `dispatch_apply`-parallelized chain). Disassembling
that function directly shows:

```
smstart za / smstop za        -- enables the SME matrix-accumulator (ZA) state
ldr  zt0, [...]                -- loads the SME2-only ZT0 lookup-table register
luti4 { z8.h - z11.h }, zt0, ...  -- SME2 lookup-table instruction (476 occurrences)
fmopa / bfmopa                 -- outer-product accumulate into ZA (64 occurrences)
```

`ZT0` and `luti4` are **SME2-exclusive** — that register and instruction don't exist in baseline
SME1 at all — so this isn't just "SME-capable hardware, possibly running an older path"; the
compiled code itself is SME2 code.

Then confirmed this is actually reached at runtime, not just present-but-dead code: set an `lldb`
breakpoint directly on that function's address (`breakpoint set --shlib libBNNS.dylib --address
0x18f6e0ca4`), launched the real benchmark under the debugger, and hit it immediately with a full,
unbroken backtrace from CoreML's prediction call down through BNNS:

```
measure_matmul_coreml`runBenchmark -> ... -> -[MLE5Engine predictionFromFeatures:...]
  -> Espresso::E5RT::Ops::BnnsCpuInferenceOperation::ExecuteSync()
  -> BNNSGraphContextExecute_v2 (libBNNS.dylib)
  -> ___lldb_unnamed_symbol_18f2dc0c8 -> ... -> dispatch_apply_with_attr (parallelized)
  -> ___lldb_unnamed_symbol_18ee1a55c -> ___lldb_unnamed_symbol_18f6e030c (dispatcher)
  -> ___lldb_unnamed_symbol_18f6e0ca4   <- breakpoint hit here, the SME2 kernel above
```

(An earlier attempt — breaking on `__bnns_hgemm`'s own entry point and on 32 addresses chosen by
raw proximity to it — never hit, at all, across either run. That ruled those specific guesses out
but didn't answer the question; the fix was finding the real callee via `sample`-derived file
offsets and the function-start table, rather than guessing by address proximity.)

**Net result: yes, SME2 is used** for this CPU FP16 matmul workload on this M4 Pro, confirmed both
statically (SME2-exclusive instructions in the actual hot function) and dynamically (`lldb`
breakpoint hit on that exact function during a real benchmark run, with a clean backtrace up
through BNNS and CoreML). Not confirmed/out of scope: whether the private legacy AMX tiers
(`MLA1`/`MLA2`/`MLA3`) are ever selected on this chip for other shapes/dtypes, or whether older
chips without `FEAT_SME2` (pre-M4-generation) fall back to those AMX tiers instead — this was only
checked for the FP16 dense path on this specific chip.

## Finding 7 — matmul now supports INT8 and FP8, matching the conv chain's QDQ pattern, on all three CoreML compute units

Extended `MILMatmulChainConfig` with the same `precision` field the conv chain already has, and
mirrored its QDQ (quantize/dequantize) pattern exactly for `matmul`:

- **INT8**: weights stored via `constexpr_blockwise_shift_scale` (INT8 inline bytes + FP16
  per-channel scale), activations `quantize`d to INT8 and `dequantize`d back to FP16 around each
  `matmul`, same `act_scale = 1/32` constant the conv chain uses.
- **FP8**: weights stored via `constexpr_lut_to_dense` (FP8-E4M3 inline indices + the same
  256-entry IEEE-Float8-to-FP16 lookup table the conv chain uses), dequantized once to FP16 — like
  the conv chain, activations are **not** quantized for FP8, only the weights.

Added `--precision fp16|int8|fp8` to `measure_matmul_coreml.m`, matching `measure_conv_coreml.m`'s
flag exactly. Ran all 9 combinations (3 precisions × ANE/GPU/CPU) on this M4 Pro, `--check`-verified
finite/non-degenerate output on every one:

| Precision | ANE | GPU | CPU |
|---|---|---|---|
| FP16 | 15.11 TOPS | 6.13 TOPS | 6.62 TOPS |
| INT8 | 18.51 TOPS | 6.17 TOPS | 4.89 TOPS |
| FP8  | 15.94 TOPS | 6.51 TOPS | 6.63 TOPS |

**FP8's output is numerically identical to FP16's** (same sample values down to the printed
digits) — expected, not a bug: like the conv chain, FP8 here only changes how the *weights* are
stored on disk (`constexpr_lut_to_dense` dequantizes them to FP16 immediately), and the actual
`matmul` still executes in FP16. This measures "FP8-compressed-weights overhead," not a native
FP8 compute path — same caveat the conv chain's FP8 mode has always had, now true for matmul too.

**INT8 is the only precision where ANE pulls meaningfully ahead of FP16/FP8** (18.51 vs ~15.5
TOPS) — consistent with INT8 being the one case here that changes the actual *activation* dtype
flowing through the op, not just the weight storage format, giving the ANE's integer path
something real to do. INT8 CPU is the only precision where CPU meaningfully regresses (4.89 vs
~6.6 TOPS for FP16/FP8) — plausibly the `quantize`/`dequantize` ops around every layer adding
real per-layer overhead on CPU that the other compute units absorb better; not investigated
further.

## Finding 8 — INT8/FP8 matmul ported to all three iPhones; the INT8-CPU-regression pattern is Mac-specific

Ported the same QDQ pattern into `ANECapacityApp`'s independent `MILSpecBuilder.m` (new
`generateWeightsMatmulINT8`/`generateWeightsMatmulFP8`, and precision branching in
`MILBuildMatmulChainSpec` mirroring the root `.mm`'s exactly), and extended
`NativeCoreMLMatmulProbe.swift` to loop over all three precisions. One Swift/ObjC-bridging
surprise along the way: `MILPrecisionFP16`/`INT8`/`FP8` import into Swift not as the usual
lowercased-prefix-stripped `.fp16`/`.int8`/`.fp8` (which is what `MILWeightModeDense` becomes —
`.dense`), but as **`.FP16`/`.INT8`/`.FP8`**, fully uppercase — plausibly because Swift's import
heuristic, when the remaining suffix is *entirely* uppercase letters/digits, treats it as an
all-caps acronym and leaves the case alone rather than lowercasing it the way it does for a
mixed-case remainder like `Dense`. Confirmed empirically via a `swiftc -typecheck` throwaway
rather than guessed.

Ran all 27 combinations (3 chips × 3 precisions × ANE/GPU/CPU) across the three iPhones, all
finite/non-degenerate:

| Precision | A18 Pro ANE | A18 Pro GPU | A18 Pro CPU | A19 Pro ANE | A19 Pro GPU | A19 Pro CPU | A20 Pro ANE | A20 Pro GPU | A20 Pro CPU |
|---|---|---|---|---|---|---|---|---|---|
| FP16 | 13.01 | 1.51 | 3.13 | 13.02 | 6.35 | 3.19 | 16.96 | 9.28 | 4.04 |
| INT8 | 17.69 | 1.50 | 3.00 | 20.43 | 6.17 | 2.93 | 21.06 | 9.23 | 3.55 |
| FP8  | 12.76 | 1.58 | 3.16 | 13.04 | 5.96 | 3.32 | 18.16 | 8.82 | 4.06 |

(All TOPS. GPU numbers are flat across precision on each chip, as expected — GPU compute doesn't
change between FP16/INT8/FP8 here any more than it did on the Mac, since the dequantize-then-FP16
pattern means the GPU's actual matmul is always FP16 regardless of precision label.)

**The headline cross-platform finding: the M4 Pro's INT8-CPU-regression result (4.89 vs ~6.6 TOPS
for FP16/FP8) does not reproduce on any of the three iPhones.** CPU INT8 is close to CPU FP16/FP8
on every phone — a small dip (A18 Pro: 3.13→3.00, A19 Pro: 3.19→2.93) or a mild dip (A20 Pro:
4.04→3.55), nothing like the Mac's ~26% drop. Combined with Finding 5's earlier observation that
the Mac's "CPU beats GPU at matmul" result also doesn't reproduce on iPhones, this is now the
**second** CPU-matmul anomaly specific to the M4 Pro/macOS CoreML CPU path rather than a property
of the underlying arithmetic — consistent with Finding 6's finding that BNNS's CPU GEMM dispatch
has multiple real hardware-tiered kernels (`MLA1/2/3`, `SME2`) whose selection logic could very
plausibly behave differently for the QDQ-wrapped INT8 case specifically on macOS vs iOS, though
this wasn't traced further.

**ANE INT8 pulling ahead of FP16/FP8 reproduces cleanly on every chip** (consistent with Finding
7's Mac result and the explanation given there — INT8 is the only precision that changes the
actual activation dtype flowing through the op, not just the weight storage format).

## Not pursued here

- Disassembling `applyOptimizationPassesWithDevice:` to confirm the nil-`metalDeviceName`
  hypothesis in Finding 1 (would reuse the `ipsw`/Xcode-cached-dylib techniques from
  `GPU_MXU_And_Convolution_Dispatch_Investigation.md`).
- Accelerate (BLAS `cblas_sgemm` / BNNS convolution) as a lower-level alternative CPU baseline,
  independent of both MPSGraph and CoreML — explicitly deferred per discussion in favor of trying
  the two existing frameworks first.
- Confirming whether the legacy AMX tiers (`MLA1`/`MLA2`/`MLA3`) are ever actually selected on this
  chip (vs. always SME2) for other shapes/dtypes, or on older chips lacking `FEAT_SME2` — Finding 6
  confirmed SME2 dispatch for the FP16 dense path on this M4 Pro, but didn't check the full matrix
  of shapes/precisions/chips.
