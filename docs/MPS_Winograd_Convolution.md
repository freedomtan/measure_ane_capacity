# Winograd Convolution in MetalPerformanceShaders

**Scope:** a side-investigation that grew out of
`GPU_MXU_And_Convolution_Dispatch_Investigation.md`'s Finding 7, where
`MPSNDArrayConvolutionDeviceBehaviorA18::IsWinogradSupported` and
`getWinogradA18PrecisionMode` turned up as two of the eleven call sites of
`_MPSForceMatrixUnit`/`_MPSDisableMatrixUnit`. This document traces those two
functions fully, plus the broader landscape of Winograd implementations
discovered across Apple's stack while doing so. Same spirit as the other two
documents in this directory — purely exploratory, unrelated to this repo's actual
ANE FP8 QDQ benchmarking work.

**Tooling/environment:** identical to the companion documents — `ipsw` against the
live system `dyld_shared_cache_arm64e` (macOS 27 / Xcode 27), plus the pre-dumped
`symaddr_27_0_1.txt` whole-cache symbol listing for fast lookups.

## Background: what Winograd convolution is, briefly

Winograd's minimal-filtering algorithm (Lavin & Gray, 2015, building on Winograd's
1980 convolution theorem) computes small convolutions (classically 3×3) using
fewer multiplications than direct convolution, at the cost of extra additions and
numerically-sensitive transform steps. It's a well-known fast-convolution
technique, not something Apple invented — the question here is specifically how
*Apple's* implementation decides when to use it and at what numeric precision.

## Finding 1 — eligibility gate: `MPSNDArrayConvolutionDeviceBehaviorA18::IsWinogradSupported`

Address: `0x190cb8bc0`, `MPSNDArray.framework` (a sub-framework of
`MetalPerformanceShaders.framework`). A sibling `A14` version exists at
`0x190c98fbc`, fully decompiled below in Finding 1b; confirmed via the sweep in
the companion doc's Finding 7 that it does **not** call either `_MPSForceMatrixUnit`
or `_MPSDisableMatrixUnit`, unlike the `A18` version).

```cpp
// __ZNK38MPSNDArrayConvolutionDeviceBehaviorA1819IsWinogradSupportedEP28NDArrayConvolutionEncodeData
// MPSNDArray.framework, 0x190cb8bc0

bool MPSNDArrayConvolutionDeviceBehaviorA18::IsWinogradSupported(
    NDArrayConvolutionEncodeData *data) const
{
    // --- Stage 1: manual disable override, early-out ---
    if (_MPSDisableMatrixUnit()) {
        os_log(logHandle, "MXU explicitly disabled. Falling back to %s", "TEC...");
                                           // string @ 0x190d87000+0x10f, truncated
                                           // here as "...isabled. Falling back to
                                           // TEC implementation." — TEC is another
                                           // undefined acronym (see Finding 4);
                                           // not traced further
        return false;
    }

    // --- Stage 2: dtype-kind filter ---
    // Confirmed from the raw disassembly (0x190cb8c40/c50/c60): the input and
    // weight dtypes are read via the same getter (bl 0x198013fe0, index 0 and
    // 1 respectively), but the OUTPUT dtype is read via a genuinely different
    // getter (bl 0x198014280, index 0 again but a different accessor function
    // entirely) — i.e. this isn't three symmetric getDType(data, N) calls, the
    // third one goes through separate accessor code. (The A14 version in
    // Finding 1b independently exhibits the same input/weight-vs-output getter
    // split, so this is a real, consistent pattern across generations, not a
    // one-off.)
    MPSDataType inputDtype  = getDType(data, 0);
    MPSDataType weightDtype = getDType(data, 1);
    MPSDataType outputDtype = getDTypeViaOutputAccessor(data);
    if (!(inputDtype & 0x10000000))  return false;   // MPSDataTypeFloatBit
    if (!(weightDtype & 0x10000000)) return false;
    if (!(outputDtype & 0x10000000)) return false;
    // further masks each dtype against 0xfff0 and requires nonzero — appears to
    // reject some sub-category of float encodings (exact semantics not pinned
    // down), on top of the bare float-kind check

    // --- Stage 3: secondary eligibility helper, not traced ---
    bool otherEligible = SomeHelper(this, data);           // bl 0x190c98ba0
    if (!otherEligible) return false;

    // --- Stage 4: precision mode must be nonzero (Finding 2) ---
    int precisionMode = getWinogradA18PrecisionMode(device, inputDtype,
                                                      weightDtype, outputDtype);
    if (precisionMode == 0) return false;

    // --- Stage 5: a three-tier channel-count gate, cost model only in Tier 3 ---
    // Ci = data->0x20 ("in" channel-ish dim), Co = data->0x18 ("out" channel-ish
    // dim). Addresses are from the raw disassembly, 0x190cb8e38–0x190cb8f5c.

    // Tier 1 — reject outright (0x190cb8e3c/0x190cb8e4c, b.lo to the common exit
    // at 0x190cb8f50 with w8 left at its default 0/false):
    if (Ci < 16 || Co < 16) return false;

    // Tier 2 — auto-accept, cost model skipped entirely (0x190cb8e50–0x190cb8e5c:
    // cmp Ci,#0x40; b.hs skips this; otherwise mov w8,#1; b to exit with w8=1):
    if (Ci < 64 /* and we didn't fail Tier 1 */) return true;

    // Tier 3 — Ci >= 64: the actual floating-point cost model, 0x190cb8e60–
    // 0x190cb8f2c. Reconstructed directly from the constant pool (every
    // constant below was decoded bit-for-bit from its mov/movk immediate):
    //   0x3f666666 = 0.9f      0x44800000 = 1024.0f      0x2c (fmov #0.5) = 0.5f
    //   0x4319999a = 153.6f    (this and 1024.0f were previously mis-stated as
    //                           "100.0" with no second constant at all — corrected)
    //   1578.0f  (0x44c54000 — a fifth constant, not previously identified)
    //
    // deviceByte1 = *(uint8_t*)(deviceRelative(+0x5c8))   // ldrb @ 0x190cb8ea0
    // deviceByte2 = same per-device field, read again and fed through a
    //               multiply-high integer-divide-by-~10 idiom:
    //               (deviceByte2 * 0x199a + 0xe66a) >> 16   // 0x190cb8ef4-f00
    //
    // directCost    = 9 * Co * Ci * KH * KW * OutSize        // x13,x16 products,
    //                                                          // 0x190cb8e64-e90
    // directTime    = directCost / (deviceByte1 * 512 * 0.9)  // 0x190cb8ea4-ec0
    //
    // winogradCost  = 9*Co*Ci*elemByteSize(weightDtype)
    //                 + 16*Co*Ci*(precisionMode==2 ? 2 : 4)   // 0x190cb8ec4-edc;
    //                                                          // the ?2:4 term
    //                                                          // means the fp19
    //                                                          // path (mode 1)
    //                                                          // is modeled as
    //                                                          // 2x costlier per
    //                                                          // tile than the
    //                                                          // fast FP16/BF16
    //                                                          // MXU path (mode 2)
    // winogradTime  = winogradCost * 1578.0
    //                 / ((deviceByte2/10) * 153.6 * 1024.0)    // 0x190cb8ee0-f24
    //
    // Winograd must beat HALF the direct-method estimate, not just edge it out:
    if (winogradTime > directTime * 0.5) return false;   // fcmp/b.gt @ 0x190cb8f30/f34

    // --- Stage 6: a second, independent size check even after Tier 3 passes ---
    // 0x190cb8f38-f4c: requires Co >= 64 (redundant with Tier 3's Ci>=64 gate —
    // note this checks Co, the OTHER channel dim) AND (KH * KW) > 0xc03 (3075).
    // That threshold is far larger than any real 3x3/5x5 kernel area, so either
    // KH/KW here are not literal kernel dimensions or "3075" gates something
    // other than kernel footprint — not resolved; see Open Threads.
    if (Co < 64) return false;
    return (KH * KW) > 3075;
}
```

**The dtype-kind filter is necessary but nowhere near sufficient.** It rules out
integer/quantized convolutions entirely (those run through a wholly separate
`IsQuantizationSupported`/affine-quantized convolution path — confirmed in the
companion doc's Finding 7 sweep to never call either override). But passing it
only gets a shape into the precision-mode and cost-model gauntlet below; it does
not by itself determine whether Winograd actually runs.

## Finding 1b — the `A14` version: no MXU override machinery at all

`MPSNDArrayConvolutionDeviceBehaviorA14::IsWinogradSupported` (`0x190c98fbc`–
`0x190c99304`, 210 instructions) is structurally quite different from the `A18`
version above — simpler, and critically, it never calls either override stub
(confirmed by the boundary-precise sweep in the companion document's Finding 7,
and visible directly in this disassembly: no reference to `0x1980f1fd0` or
`0x1980f1fc0` anywhere in the function).

```cpp
// __ZNK38MPSNDArrayConvolutionDeviceBehaviorA1419IsWinogradSupportedEP28NDArrayConvolutionEncodeData
// MPSNDArray.framework, 0x190c98fbc

bool MPSNDArrayConvolutionDeviceBehaviorA14::IsWinogradSupported(
    NDArrayConvolutionEncodeData *data) const
{
    // --- Stage 1: reject grouped/multi-something convolutions ---
    if (data->someCountField >= 2) {
        log("...");           // string @ 0x190d85000+0x88b
        return false;
    }

    MPSDataType in     = getDType(device, index0);     // via 0x198013fe0
    MPSDataType weight = getDType(device, index1);
    MPSDataType out    = getDType(device, index2);      // via 0x198014280 (different getter)

    // --- Stage 2: reject any 8-bit dtype (Int8/UInt8/Bool-shaped), via an
    //     OR-then-compare trick that's kind-agnostic — NOT a float-kind
    //     check like A18's. A14 would actually accept Int16/Int32/etc. here,
    //     as long as the size field isn't 8. ---
    if (sizeFieldIs8(in) || sizeFieldIs8(weight) || sizeFieldIs8(out)) {
        log("...");          // @ 0x190d85000+0x97c
        return false;
    }

    // --- Stage 3: a mode-specific extra check ---
    if (data->modeByte == 1) {
        if (data->dimA * data->dimB < 5) {
            log("...");       // @ 0x190d85000+0x912
            return false;
        }
    }

    // --- Stage 4: a plugin-compatibility helper — DIAGNOSTIC ONLY ---
    // Calls two PAC-signed virtual methods on the input/weight dtype objects
    // (likely "isFloatingPoint"-shaped queries), then (if both pass) calls a
    // separate helper (0x190c7be84 — plausibly IsPluiginWinogradCompatible,
    // the exact symbol found by string search below in Finding 4). If that
    // returns nonzero, this logs a diagnostic warning (with a literal source
    // line number, 0x131 = 305, baked into the log call) but does NOT reject
    // — execution continues regardless of the result.

    // --- Stage 5: shape/dimension thresholds ---
    // Gathers five dimension values (kernel/stride/dilation-ish) via a
    // repeated helper call (0x190c98f44), with slightly different struct
    // offsets depending on whether someCountField is 0 or 1.
    if (kernelDim < 16) {                    log("..."); return false; }  // @ +0xa63
    if (weightDim <= (someFlag ? 4 : 8)) {   log("..."); return false; }  // @ +0xaaa (reused)
    if (dimA * dimB * dimC <= 15) {          log("..."); return false; }  // @ +0xaaa
    return true;
}
```

Two things stand out next to the `A18` version in Finding 1:

- **No precision-mode selection whatsoever.** `A14` returns a flat boolean; there
  is no equivalent of `getWinogradA18PrecisionMode`, no `_MPSForceMatrixUnit`/
  `_MPSDisableMatrixUnit` consultation, no fp19-intermediate decision. This
  confirms the entire override/precision-mode apparatus documented in Finding 2
  is genuinely new, `A18`-specific infrastructure — consistent with `A14`
  predating MXU hardware entirely (it's below even the real G17 hardware cutoff
  established in the companion `MTLDevice_supportsFP8_Reverse_Engineering_Report.md`
  document).
- **A different dtype-eligibility philosophy, not just a missing feature.** `A14`
  rejects dtypes via a kind-agnostic "is the size field 8" check — which would
  reject `Int8`/`UInt8`/`Bool`-shaped types but *accept* something like `Int16`
  or `Int32` if it got this far — whereas `A18` positively requires the
  float-kind bit (`0x10000000`) on all three dtypes. The two generations don't
  just differ in how sophisticated their precision handling is; they disagree on
  what counts as an eligible dtype in the first place.

## Finding 2 — `getWinogradA18PrecisionMode`, fully decompiled

Address: `0x190cb8fc8`, same framework. Takes `(MPSKernel *kernel, MPSDataType in,
MPSDataType weight, MPSDataType out)` and returns an integer precision mode (`0`,
`1`, or `2` observed). This is the function that, in the companion document, was
initially misread as checking an `Int32` dtype triple — that was a cross-contamination
from a *different* function (`isMXUDataTypeSupported`, a matmul-path function)
being read in parallel; corrected here.

```cpp
// __Z27getWinogradA18PrecisionModeP9MPSKernel11MPSDataTypeS1_S1_
// MPSNDArray.framework, 0x190cb8fc8

int getWinogradA18PrecisionMode(MPSKernel *kernel, MPSDataType in,
                                  MPSDataType weight, MPSDataType out)
{
    uint8_t caps = *(uint8_t *)(deviceRelative(kernel, +0xf20_slot));
                                        // genuine per-device ivar byte read —
                                        // same GOT-ivar-offset-holder pattern
                                        // used throughout this investigation for
                                        // real device-object fields, not a
                                        // process-wide cache

    if (caps & 0x42) {                  // bits 1 and 6
        if (in == weight && weight == out && in == MPSDataTypeFloat16) {
                                        // 0x10000010
            os_log(logHandle, "allowReducedPrecision flag is set. "
                               "Running A18 MXU Winograd...");  // @ 0x190d87e86-ish
            return 2;
        }
        if (in == weight && weight == out && in == MPSDataTypeBFloat16) {
                                        // 0x90000010 = AlternateEncodingBit|FloatBit|16
            os_log(logHandle, "...ision flag is set. Running A18 MXU...");
                                        // @ 0x190d87ed9-ish, same wording pattern
            return 2;
        }
        // any other dtype combination falls through even with caps bit set
    }

    // --- three independent manual overrides, checked in this order ---
    if (cachedFlag_0xec0) {             // dispatch_once-cached, own predicate
                                        // @ page 0x1e4748000+0xee0; source env
                                        // var name not identified
        os_log(logHandle, "... flag is set. Running A18 MXU...");
        return 2;
    }
    if (cachedFlag_0xec1) {             // separate dispatch_once-cached flag,
                                        // predicate @ +0xee8
        os_log(logHandle, "MPSNDARRAY_WINOGRAD_FP19_INTERMEDIATE is set. "
                           "Running A18 MXU Winograd...");   // fully confirmed
                                                               // env var name
        return 1;
    }
    if (_MPSForceMatrixUnit()) {        // MPSNDARRAY_FORCE_MXU
        os_log(logHandle, "FORCE_MXU is set to 1. Forcing A18 MXU Winograd...");
        return 1;
    }

    // --- final check: a SECOND, different per-device capability byte ---
    uint8_t caps2 = *(uint8_t *)(deviceRelative(kernel, +0x90_slot));
    if (!(caps2 & 0x80)) {              // bit 7
        return 0;                        // the true default: Winograd precision
                                          // unavailable
    }
    os_log(logHandle, "Running A18 MXU Winograd using fp19 precision. "
                       "AllowConvertingOperandsFromFP32ToFP19 flag is set.");
    return 1;
}
```

**So the dtype restriction is narrower than "any float type," but not in the way
first assumed.** `Float16`-only and `BFloat16`-only triples get a fast,
hardware-capability-gated path straight to mode 2. Every other float combination
— including plain `Float32`, and any mixed-precision combination — skips that
fast path entirely and depends on: three independent manual overrides (one of
which, `MPSNDARRAY_WINOGRAD_FP19_INTERMEDIATE`, is now a confirmed real env var
name; `_MPSForceMatrixUnit`/`MPSNDARRAY_FORCE_MXU` is a second), or, failing all
three, a *second* and different per-device capability bit (bit 7) that gates
whether FP32 operands are allowed to be converted to the FP19 intermediate format
for the Winograd transform. If none of those five conditions hold, the result is
mode `0`, and `IsWinogradSupported` returns `false`.

**On "fp19":** this is the format behind the string fragment found earlier
(`"...d using fp19 precision."`) — a reduced-but-wider-than-fp16 internal
accumulator format Apple's GPUs use for Winograd's numerically-sensitive
transform-domain intermediate products. It is cheaper than full fp32 while
carrying more precision than fp16, which is exactly the trade-off Winograd's
transform step benefits from (the forward/inverse transforms involve small
constant-coefficient linear combinations that can amplify rounding error at fp16).
No further detail on the exact bit layout of "fp19" was found or pursued.

## Finding 3 — a separate, dedicated precision override: `MPSWinogradForceF16`

Found via string search (`_MPSIsReducedPrecision`'s sibling family — see the
companion document's Finding 7 — includes several similarly-shaped functions;
this one is Winograd-specific rather than generic). Not fully decompiled in this
document, but its existence is confirmed via the mangled block-helper symbol
`____ZL19MPSWinogradForceF16v_block_invoke` in `libobjc.A.dylib`'s string table —
i.e. a `dispatch_once`-driven block, same shape as every other override function
in this investigation. This gives Winograd *another* manual knob, independent of
the four checks inside `getWinogradA18PrecisionMode`, specifically for forcing
FP16 instead of whatever precision mode would otherwise be selected. The relationship
between this override and the four checks above (does it short-circuit before them,
override their result, or something else) was not traced.

## Finding 4 — multiple, structurally distinct Winograd implementations coexist

A string sweep across `MetalPerformanceShaders`/`MPSNDArray`/`MPSNeuralNetwork`
turned up evidence of at least four separate Winograd code paths in Apple's stack,
built at different times with no apparent code sharing:

1. **`MPSNDArray`'s current "Plugin" Winograd** — the generation-specific
   (`A14`/`A18`) implementation this document traces. Named via
   `__Z27IsPluiginWinogradCompatibleP28NDArrayConvolutionEncodeData` — yes,
   genuinely misspelled "Pluigin" in Apple's own mangled symbol, not a transcription
   error introduced here.
2. **An "AMD"-named fallback** (`ndArrayWinogradAMD_NCHW`), confirmed as a real,
   intentional fallback via the log string `"Plugin Winograd not efficient for
   small images. Falling back to MPS AMD implementation"`. Nothing found in this
   investigation suggests any relation to AMD GPUs specifically — the name's
   origin wasn't traced.
3. **An `A14`-specific direct-convolution fallback** for data/weight format
   combinations the A14 Winograd kernels don't support:
   `"A14 Winograd Convolution: Only NHWC and NCHW data format and HWIO weights
   format supported by A14 direct kernels. Falling back to A14 direct
   implementation."`
4. **A completely separate classic implementation** in the older
   `MPSNeuralNetwork.framework`: `cnnConvWinograd_2x2_3x3_32x32_256` — the
   textbook F(2,3) minimal Winograd algorithm (2×2 output tile, 3×3 filter,
   named for its tiling block size `32x32` and presumably a `256`-channel
   specialization). This framework and `MPSNDArray` are historically distinct
   APIs (`MPSCNNConvolution` vs. `MPSNDArray`-based convolution), and nothing in
   this investigation found any connection between the two Winograd
   implementations.
5. **A "TEC" fallback** — found in `IsWinogradSupported`'s `_MPSDisableMatrixUnit`
   early-out log message ("Falling back to TEC implementation"). TEC's expansion
   is undefined anywhere in this cache, same situation as "MXU" in the companion
   document — likely another GPU execution-unit acronym, not confirmed.

## Finding 5 — a wholly unrelated fifth implementation: the ANE/CoreML MLIR compiler

Separately from everything above — found via the same broad string sweep, purely
by coincidence of search terms — the ANE compiler stack has its **own**,
completely independent Winograd implementation, as a first-class MLIR operation:

```
polylang::anehlo::WinogradOp                       (the MLIR op itself)
mlir::detail::WinogradInputTransformOpGenericAdaptorBase
mlir::detail::WinogradFilterTransformOpGenericAdaptorBase
LLIR::AneHloEmitter::EmitWinograd(mlir::Block*)     (the lowering/emission code)
```

This lives in CoreML's HLO-based ANE compiler ("AICode"/"ODIE compiler", in the
terminology from the companion `MTLDevice_supportsFP8_Reverse_Engineering_Report.md`
document), not in `MetalPerformanceShaders` at all. It shares no code, and
presumably no implementation details, with anything in Findings 1–4 above. This is
the one finding in this document that's actually adjacent to this repo's real
purpose: this repo's ANE FP8 QDQ benchmarking work would, if it ever touched a
Winograd-eligible convolution, go through *this* implementation exclusively — the
`MPSNDArray`/GPU-side machinery documented above is entirely unreachable from the
ANE compilation path.

## Open threads (not pursued further)

- The exact masking semantics of `IsWinogradSupported`'s `& 0xfff0` dtype checks
  (beyond the bare float-kind bit) were not pinned down.
- The secondary eligibility helper at `0x190c98ba0`, called from
  `IsWinogradSupported`, was not decompiled.
- The exact floating-point cost-model formula (the FLOP-ratio estimate, now
  confirmed to use constants `0.9`/`1024.0`/`153.6`/`0.5` plus an unidentified
  integer multiply-high scaling of a per-device byte field — see Finding 1,
  Stage 5) was read from the instruction sequence but not translated into a
  clean closed-form expression.
- The three manual override flags in `getWinogradA18PrecisionMode` were only
  fully named in one case (`MPSNDARRAY_WINOGRAD_FP19_INTERMEDIATE`); the first
  ("allowReducedPrecision"-logged) override's actual env var name/source was not
  traced back past its log message.
- `MPSWinogradForceF16` (Finding 3) was found but not decompiled; its precedence
  relative to `getWinogradA18PrecisionMode`'s own four-stage check was not
  determined.
- What "TEC" stands for (Finding 4, item 5) is unknown — same category of
  unresolved acronym as "MXU" itself in the companion document.
- None of the actual compiled Metal kernel bodies (whatever backs the "Plugin"
  A18 Winograd kernels) were disassembled — this document is scoped to the
  CPU-side eligibility/dispatch logic only, consistent with every other document
  in this series.

None of these affect this repo's actual ANE FP8 QDQ work, which (per Finding 5)
would never touch any of the GPU-side machinery documented here regardless.
