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

## Not pursued here

- Disassembling `applyOptimizationPassesWithDevice:` to confirm the nil-`metalDeviceName`
  hypothesis in Finding 1 (would reuse the `ipsw`/Xcode-cached-dylib techniques from
  `GPU_MXU_And_Convolution_Dispatch_Investigation.md`).
- Accelerate (BLAS `cblas_sgemm` / BNNS convolution) as a lower-level alternative CPU baseline,
  independent of both MPSGraph and CoreML — explicitly deferred per discussion in favor of trying
  the two existing frameworks first.
- AMX detection/dispatch investigation (whether Accelerate's CPU GEMM actually reaches Apple
  Silicon's CPU-side matrix coprocessor) — out of scope until a CPU GEMM path exists at all.
