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

## Open: no CoreML/MIL matmul equivalent exists yet

`MILSpecBuilder.mm` (the in-process MIL-program builder behind `measure_conv_coreml.m`) only emits
`conv` ops — there is no `measure_matmul_coreml.m` and no `matmul`/`linear` op support in
`MILSpecBuilder.mm`. Building one would be new work of comparable size to the existing conv
builder (op descriptor, shape plumbing, `--units`/`--plan`/`--check` CLI surface matching
`measure_conv_coreml.m`'s conventions), not attempted in this pass.

## Not pursued here

- Disassembling `applyOptimizationPassesWithDevice:` to confirm the nil-`metalDeviceName`
  hypothesis in Finding 1 (would reuse the `ipsw`/Xcode-cached-dylib techniques from
  `GPU_MXU_And_Convolution_Dispatch_Investigation.md`).
- Accelerate (BLAS `cblas_sgemm` / BNNS convolution) as a lower-level alternative CPU baseline,
  independent of both MPSGraph and CoreML — explicitly deferred per discussion in favor of trying
  the two existing frameworks first.
- AMX detection/dispatch investigation (whether Accelerate's CPU GEMM actually reaches Apple
  Silicon's CPU-side matrix coprocessor) — out of scope until a CPU GEMM path exists at all.
