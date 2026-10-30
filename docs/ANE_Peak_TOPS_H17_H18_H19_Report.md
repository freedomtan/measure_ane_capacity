# ANE Peak TOPS Across H17 / H18 / H19 — FP8's Generation Lock Is One Chip Earlier Than the GPU's

Companion to `docs/GPU_Peak_TOPS_A18_A19_A20_Report.md`. Same three devices,
same app, same sweep (`--ane --mps --all --sweep matmulDimensions`, 128³–4096³,
L=20, `--autorun`), same day — but target `.ane` instead of `.gpu`. The ANE
engine doesn't expose a raw matmul primitive; `ANECapacityEngine.runSingle`
maps GEMM onto ANE as a 1×1 convolution (`[B,K,H,W]×[N,K,1,1]→[B,N,H,W]`,
documented at the top of that branch in `ANECapacityEngine.swift`) and lets
MPSGraph's ANE compilation path (`preferredDevice = 2`) decide per-precision
whether it can actually compile to the Neural Engine.

| Device | Chip | ANE generation |
|---|---|---|
| iPhone 16 Pro | A18 Pro | H17 |
| iPhone 17 Pro | A19 Pro | H18 |
| iPhone 18 Pro | A20 Pro | H19 |

(ANE generation number = A-series number minus 1 — an existing, already-documented
mapping in this repo, not something this report establishes.)

## The headline difference from the GPU story

The GPU's FP8 hardware cutoff is A20 Pro only (`Apple11` GPU family — see the
companion report). **The ANE's FP8 cutoff is one generation earlier**: H18 and
H19 (iPhone 17 Pro **and** iPhone 18 Pro) both compile FP8 natively to the
Neural Engine. Only iPhone 16 Pro (H17) can't — and unlike the GPU's hard
crash, the ANE failure is silent and non-fatal: MPSGraph just declines to
emit an ANE microcode bundle and the op runs on GPU instead, with no error,
exception, or crash of any kind. The tell is purely in the console log:

```
[MPS]    [PMU] Locating compiled ANE microcode bundle...
[MPS]    [PMU Note] No temporary ANE bundle emitted by MPSGraph on this runtime/device.
```

vs. the success case:

```
[MPS]    [PMU] Locating compiled ANE microcode bundle...
[MPS]    [PMU] Found ANE microcode bundle: mpsgraph-<pid>-<timestamp>-<hash>
[MPS]    └─ PMU: Compute=+..., Stalls=+..., DMA=+... MB, ALU Saturation=...%, Clock=... GHz
```

On iPhone 16 Pro, every one of the 6 FP8 sweep steps printed the "No temporary
ANE bundle" line; on iPhone 17 Pro and iPhone 18 Pro, every FP8 (and FP16, and
INT8) step printed a real PMU bundle with real compute/stall/DMA counters. The
measured numbers below bear this out directly: iPhone 16 Pro's "ANE FP8"
throughput is numerically indistinguishable from its GPU FP8 fallback numbers
in the companion report (e.g. 1.73 TOPS at 4096³, 1.58 TOPS at 1024³ — same
figures, because it's literally the same GPU path underneath).

## Peak ANE TOPS, Matrix Dimension Sweep (128³–4096³, L=20, all three precisions in one run)

**iPhone 16 Pro (H17):**

| Size | FP16 | INT8 | FP8-E4M3 |
|---|---|---|---|
| 128³ | 0.08 | 0.12 | 0.05 |
| 256³ | 0.89 | 0.58 | 0.24 |
| 512³ | 3.59 | 4.22 | 1.19 |
| 1024³ | 12.04 | 9.02 | 1.58 |
| 2048³ | **13.03** | **19.68** | **1.73** |
| 4096³ | 6.96 | 17.62 | 1.48 |

<img src="images/iphone16pro_ane_fp16_fp8_int8_figures.png" width="320" alt="iPhone 16 Pro — ANE FP16/FP8/INT8 throughput vs matrix size">

*FP8 (teal) is pinned far below FP16 and INT8 across the whole sweep — the visual signature of "silently running on GPU, not ANE."*

**iPhone 17 Pro (H18):**

| Size | FP16 | INT8 | FP8-E4M3 |
|---|---|---|---|
| 128³ | 0.12 | 0.11 | 0.12 |
| 256³ | 0.63 | 0.55 | 0.60 |
| 512³ | 3.33 | 3.64 | 3.86 |
| 1024³ | **14.42** | 12.24 | 12.95 |
| 2048³ | 13.78 | **18.46** | 18.33 |
| 4096³ | 8.93 | 15.30 | 16.75 |

<img src="images/iphone17pro_ane_fp16_fp8_int8_figures.png" width="320" alt="iPhone 17 Pro — ANE FP16/FP8/INT8 throughput vs matrix size">

*FP8 now tracks INT8 almost exactly, both clearly above FP16 — real native ANE hardware acceleration, one generation earlier than the GPU gets it.*

**iPhone 18 Pro (H19):**

| Size | FP16 | INT8 | FP8-E4M3 |
|---|---|---|---|
| 128³ | 0.07 | 0.09 | 0.08 |
| 256³ | 0.72 | 0.56 | 0.62 |
| 512³ | 3.63 | 3.89 | 3.65 |
| 1024³ | 18.97 | 12.43 | 13.04 |
| 2048³ | **27.03** | 31.36 | 31.23 |
| 4096³ | 18.84 | **43.87** | **45.01** |

<img src="images/iphone18pro_ane_fp16_fp8_int8_figures.png" width="320" alt="iPhone 18 Pro — ANE FP16/FP8/INT8 throughput vs matrix size">

*Same shape as H18, scaled up — FP8/INT8 keep climbing through 4096³ where H17/H18 start to roll off (see below).*

## Peak-TOPS summary and the INT8/FP16 ratio the app computes in-line

| Device | Peak FP16 | Peak INT8 | INT8/FP16 (app's own "Speedup" card) | Peak FP8 |
|---|---|---|---|---|
| iPhone 16 Pro (H17) | 13.03 TOPS | 19.68 TOPS | 1.51x | 1.73 TOPS (not real — GPU fallback) |
| iPhone 17 Pro (H18) | 14.42 TOPS | 18.46 TOPS | 1.28x | 18.33 TOPS (real) |
| iPhone 18 Pro (H19) | 27.03 TOPS | 43.87 TOPS | 1.62x | 45.01 TOPS (real) |

Two separate stories here, not one:

1. **INT8 vs FP16 is a smooth, every-generation win** (1.28x–1.62x), present
   on all three chips — unsurprising, since ANE INT8 has existed since H13.
2. **FP8 is a step function**, exactly where the generation table says it
   should be: worthless (literally a GPU number in ANE's clothing) on H17,
   then immediately competitive with INT8 — not a fraction of it — on both
   H18 and H19.

## A second, unrelated pattern worth flagging: FP16 rolls off at 4096³, low-precision doesn't (as much)

All three devices show **FP16 throughput peaking at 1024³–2048³ and then
dropping** at 4096³ (16 Pro: 13.03→6.96 TOPS, roughly halving; 17 Pro:
14.42→8.93 TOPS; 18 Pro: 27.03→18.84 TOPS). INT8 and FP8 show the same
roll-off on H17/H18 (e.g. 17 Pro INT8 18.46→15.30, FP8 18.33→16.75) but **not**
on H19, where both keep climbing all the way to 4096³ (31.36→43.87,
31.23→45.01). This reads like an ANE on-chip SRAM/working-set capacity
difference between generations — larger GEMMs' activations/weights spill out
of whatever's resident on H17/H18 earlier than on H19 — but this report
doesn't trace that further; it's a flagged-but-unexplained observation, not a
finding. (The companion GPU report's L2 SRAM framing in `matmulAsymmetric`'s
sweep design is the closest existing investigation of this general class of
effect, just not for ANE specifically.)

**Attempted and abandoned**: tried to confirm the SRAM-spill hypothesis via
the app's per-step PMU DMA counter (`ANECapacityEngine.swift`'s
`dmaRwBytes = pmuCounters["kANE_DMA_READWRITE_BYTES"] + pmuCounters["kANE_DMA_READ_BYTES"]`,
logged as `DMA=+... MB`) — if the hypothesis were right, DMA traffic should
jump sharply at the same size FP16 rolls off. Instead this surfaced a
separate, confirmed bug: **the DMA counter is nonsensical on iPhone 17 Pro**.
At 128³ (16KB-scale tensors), iPhone 16 Pro reports `DMA=+406.2 KB` and
iPhone 18 Pro `DMA=+1.5 MB` — physically plausible — but iPhone 17 Pro reports
`DMA=+4825.4 MB` for the *identical* 128³ shape, a ~10,000x outlier, and the
number keeps growing into the gigabyte-to-petabyte range as size increases
(`+18766197.8 MB` at 4096³). Whatever `kANE_DMA_READWRITE_BYTES`/
`kANE_DMA_READ_BYTES` mean on H18 specifically, it isn't bytes-of-DMA-traffic
at the same scale as H17/H19 — possibly a different PMU register
layout/unit for this counter on that generation that the app doesn't account
for. (`ALU Saturation` exceeding 100% at larger sizes on iPhone 18 Pro, e.g.
`163.6%` at 2048³, is a second, smaller sign that these derived PMU-based
metrics are approximate and shouldn't be over-trusted as literal hardware
utilization.) Net result: the SRAM-spill hypothesis for the FP16 roll-off
remains untested — the one counter that could have confirmed or killed it
isn't trustworthy on the one device (H18) where the comparison matters most.

## Verification: `matmulDepth` and `matmulAsymmetric` on ANE

Same follow-up as the companion GPU report: `runSingle`'s ANE-as-1×1-conv
mapping triggers for any `dims.opType == .matmul` with `target == .ane`
regardless of sweep type, so `matmulDepth` (fixed 1024³, `layers` 1–100) and
`matmulAsymmetric` (fixed K=N=1024, `M` 512–8192) go through the same code
path as `matmulDimensions`. Both were smoke-tested with `--all`
(FP16+FP8+INT8) on all three devices — including iPhone 16 Pro, the case
where FP8 silently falls back to GPU on every single step. All six runs
(2 sweeps × 3 devices) completed with no crash, hang, or watchdog reset, exit
code 0 throughout. Peak numbers from these runs (not independently
interesting beyond "it works," since they're the same GEMM-as-1×1-conv path
measured at fixed size/varying depth or fixed depth/varying M rather than a
new shape): iPhone 18 Pro reached 47.9 TOPS FP8 / 48.1 TOPS INT8 at
`matmulAsymmetric`'s `M=8192` step, both again far above its FP16 ceiling —
consistent with the main `matmulDimensions` findings above, not a
contradiction of them.

## Gaps

- Existing ANE conv (3×3, not 1×1-mapped-GEMM) peak-TOPS numbers already
  documented elsewhere in this repo (`ANE_H19_Theoretical_Peak_and_Winograd_FP8_Analysis.md`,
  `FP8_BENCHMARKING_AND_ANE_ARCHITECTURE.md`) were not re-run — they use a
  different workload shape (3×3 convolution, not GEMM) and already have
  iPhone 17 Pro and iPhone 18 Pro data; only iPhone 16 Pro was the genuinely
  missing data point there, and this report's GEMM numbers (not conv numbers)
  fill a different gap, not that one.
- The PMU DMA-counter bug on iPhone 17 Pro (above) isn't root-caused — only
  confirmed as a bug via cross-device comparison, not traced into
  `ANEClientBridge`/the PMU key definitions to find why H18 differs.
