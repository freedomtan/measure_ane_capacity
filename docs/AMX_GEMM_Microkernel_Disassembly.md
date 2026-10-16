# Annotated disassembly: the real FP16 GEMM microkernel behind Findings 6/12/13

## Correction to Findings 6, 12, and 13

Findings 6, 12, and 13 claimed the function hit by a live `lldb` breakpoint at
`libBNNS.dylib + 0xa73ca4` (static address `0x18f6e0ca4`) *contains* `smstart za` / `ldr zt0` /
`luti4` / `fmopa` — i.e. that the hot CPU matmul path is genuine SME2. That attribution was wrong.
It came from extracting a disassembly window `[0x18f6e0ca4, 0x18f72d000)` — far wider than the
function actually is — so the SME2 instructions found in that window belong to a *different,
later* function, not the one the breakpoint hits.

The real boundary of the function at `0x18f6e0ca4`, found from its own `pacibsp`/`ret` pair, is
only `0x18f6e0ca4`–`0x18f6e58a4` (19,456 bytes) — confirmed against the function-start table used
throughout this investigation. It contains **no** SME2 instructions at all. What it does contain
is three PAC-authenticated indirect calls (`blraa`/`blraaz`), which don't show a static target in
disassembly. Tracing one live with `lldb` (`step` through the `blraaz x8` at `+456`) resolves it to
`libBNNS.dylib + 0x1b0324` (`0x18ee1d324`), which calls (direct `bl`, confirmed statically) into
`libBNNS.dylib + 0x3f52e4` (`0x18f0622e4`) — a small wrapper that builds a signed callback pointer
to the very next function, `0x18f0623a4`, and hands it to what looks like an internal thread-pool
"run task" primitive (matching the `dispatch_apply`-based parallelism already seen in the
backtrace: `BNNSGraphContextExecute_v2 → ... → dispatch_apply_with_attr → ...`).

**`0x18f0623a4` is the real compute kernel**, and it is built entirely from the same undocumented
Apple AMX NOP-space instruction encoding found in Finding 11 — not SME2. Three independent checks
converge on this:
1. Live `lldb` step-through of the call chain above, instruction by instruction.
2. `sample`-profiling the real benchmark run: the hottest sampled addresses (`libBNNS.dylib +
   0x3f5590`/`0x3f55a4`) land exactly on two consecutive instructions inside this function
   (an AMX load, immediately followed by its loop-increment).
3. The function this kernel calls to write results out (`bl 0x18f04911c`, called twice per outer
   iteration) is *also* one of the hottest sampled addresses independently.

So: **the CPU matmul benchmark used throughout Findings 6–13 is AMX, not SME2.** The real SME2
code (`smstart za`/`zt0`/`luti4`/`fmopa`) genuinely exists in `libBNNS.dylib` — that part of
Finding 6 was accurate — but it belongs to some other kernel variant not reached by this
benchmark's specific shape/precision/compute-unit combination. Whether any configuration in this
repo reaches it is now an open question, not a settled one. Findings 6, 12, and 13's TOPS numbers
and cross-chip/cross-precision comparisons are all still valid as *timing* measurements; only the
"which hardware engine" conclusion needs retracting.

## Call chain (confirmed live)

```
BNNSGraphContextExecute_v2 (public API)
  -> E5RT::Ops::BnnsCpuInferenceOperation::ExecuteSync()           [Espresso]
  -> ... -> dispatch_apply_with_attr (parallelized across threads)  [libdispatch]
  -> 0x18f6e0ca4   outer per-thread driver (19,456 bytes, no SME/AMX itself)
       blraaz x8 @ +456
  -> 0x18ee1d324   recursive row-tiling wrapper (392 bytes)
       bl 0x18eeacf40   (small-K fast path, K <= 31)
       bl 0x18f0622e4   (K > 31 path, taken for this benchmark's K=1024)
  -> 0x18f0622e4   per-tile task-descriptor builder (192 bytes)
       builds a PAC-signed pointer to 0x18f0623a4, hands off to a thread-pool
       "run task" primitive (bl 0x18f51effc / bl 0x18f51f000)
  -> 0x18f0623a4   *** the actual AMX GEMM microkernel (1,224 bytes) ***
       bl 0x18f04911c (x2 per outer iteration: write low half, then high half of the output tile)
```

## Decoded AMX opcodes

Every `.long 0x0020xxxx` in this function is Apple's private AMX instruction set, hijacking ARM's
reserved NOP encoding space (`instr = 0x00201000 | (op << 5) | Xt`, `Xt` the GPR holding the
operand — the scheme documented by public reverse-engineering projects such as `corsix/amx`; not
an Apple-published spec, so treat exact op *names* below as "best available public mapping," not
Apple-confirmed). Decoding every distinct value in this function:

| Encoded | Offset | op | operand reg | Likely meaning |
|---|---|---|---|---|
| `0x201009`/`0x20100a`/`0x20100c`/`0x20100d` | 0x9/0xa/0xc/0xd | **0** | x9/x10/x12/x13 | `ldx` — load the AMX X staging register from the packed pointer in the given GPR |
| `0x201029` | 0x29 | **1** | x9 | `ldy` — load the AMX Y staging register (here: the per-tile FP16 scale vector built just before it) |
| `0x20108a` | 0x8a | **4** | x10 | `ldz` — load/select the AMX Z accumulator tile |
| `0x2011e8`/`0x2011e9`/`0x2011ea` | 0x1e8/0x1e9/0x1ea | **15** | x8/x9/x10 | an FMA/outer-product variant into Z (exact op name uncertain — appears immediately after the same "pack pointer with flag bit 62 set" idiom used for `ldz`, i.e. this is a *compute* op consuming the just-loaded X/Y tiles) |
| `0x201268`/`0x201269`/`0x20126a`/`0x20126c`/`0x20126d` | 0x268.. | **19** | x8/x9/x10/x12/x13 | the main tile outer-product accumulate (appears inside every hot per-row loop immediately after `ldx`, consuming one row of A against the resident Y/weight tile) |

(For comparison: Finding 11's legacy `BNNSFilterCreateLayerFullyConnected` kernel used op **12**
for its hot accumulate, decoded there as `fma64` from the same public table — a different op
number than this kernel's op 19/15. Either the earlier op-12 label was imprecise, or this is a
genuinely different, FP16-specific AMX op family from that legacy kernel's generic path; not
resolved further here.)

## Annotated disassembly

Register roles (inferred from access patterns, not from any symbol — this binary is fully
stripped): `x19` = pointer to a context/descriptor struct passed in `x0`; `x20` = clamped tile
height (`min(K, 32)`); `x22` = remaining tile width for the current block; `x23`/`x28` = row-tile
and byte-offset counters; `x25`/`x27` = base addresses derived from the context struct, each
packed with AMX flag bits in their high byte(s).

```asm
; --- Prologue: standard stack frame + 64-byte SP realignment for AMX's
; cache-line-aligned staging buffers, plus a stack-protector canary save. ---
pacibsp
stp  d9, d8,   [sp, #-0x70]!
stp  x28, x27, [sp, #0x10]
stp  x26, x25, [sp, #0x20]
stp  x24, x23, [sp, #0x30]
stp  x22, x21, [sp, #0x40]
stp  x20, x19, [sp, #0x50]
stp  x29, x30, [sp, #0x60]
add  x29, sp, #0x60
sub  x9, sp, #0x90
and  sp, x9, #0xffffffffffffc0      ; realign SP to 64 bytes
mov  x19, x0                        ; x19 = context struct ptr (arg0)
adrp x8, ...; ldr x8,[x8,#0x5c0]; ldr x8,[x8]
str  x8, [sp, #0x88]                ; stack-protector canary

; --- Load a per-tile FP16/BF16 scale value from the context struct and
; stage it into the AMX Y register via `ldy` (op 1). Three cases: scale==1.0
; uses a fixed flag-only encoding (no real load); otherwise the FP32 scale
; is converted to BF16 or FP16 depending on a mode word at [x19+0x64]. ---
ldr  s8, [x0, #0x60]                 ; s8 = scale (fp32)
ldr  w9, [x0, #0x64]                 ; w9 = mode flag
mov  w10, #0x18010                   ; sentinel mode value
cmp  w9, w10
csel x8, xzr, #0x80000000000, eq     ; x8 = AMX flag bits for this mode
fmov s0, #1.0
fcmp s8, s0
b.ne 0x18f062418
  mov  x10, #0x0005800000000000      ; scale==1.0 fast-path flag bits
  b    0x18f062448
0x18f062418:
cmp  w9, w10
b.ne 0x18f06242c
  bfcvt h0, s8                       ; mode==sentinel: fp32 -> bf16
  b    0x18f062434
0x18f06242c:
  fcvt h0, s8                        ; else: fp32 -> fp16
0x18f062434:
str  h0, [sp, #0x40]
add  x9, sp, #0x40
and  x9, x9, #0xffffffffffffc0
orr  x9, x9, #0x0700000000000000     ; pack AMX "load Y" flags into byte 56
.long 0x00201029                     ; ldy x9   (op=1: stage scale into Y)

; --- Clamp the per-call tile height to 32 rows, early-out if K==0. ---
ldr  x9, [x19, #0x20]                ; x9 = K (reduction dim)
cmp  x9, #0x20
csel x20, x9, #0x20, lo              ; x20 = min(K, 32)   <- the AMX tile height
cbz  x9, 0x18f062828                 ; K==0 -> straight to epilogue

; --- Build the two packed base-address registers (x25, x27) used by every
; inner loop below, folding in the per-mode AMX flag bits computed above. ---
mov  x13, #0x0
orr  x11, x10, x8
mov  x12, #0x40, lsl #32
orr  x11, x11, x12
add  x8, x8, x10
add  x25, x8, x12                    ; x25 = packed base pointer, "A" side
add  x8, x25, #0x100, lsl #12
stp  x8, x11, [sp, #0x18]
ldr  x8, [x19, #0x28]                ; x8 = M (outer row count)
add  x27, x25, #0x110, lsl #12       ; x27 = packed base pointer, "B"/weight side
fmov s9, #1.0
mov  x24, #0x0200000000000000        ; per-row AMX address increment (row stride flag)

; --- Outer loop over K in chunks of <=32 (x13 is the K offset, bounded by x20). ---
0x18f062490:  ; for (k0 = 0; k0 < K; k0 += tileHeight)
sub  x10, x9, x20
cmp  x10, x13
csel x11, x10, x13, lo
cbz  x8, 0x18f06281c                 ; M==0 -> next K chunk directly

; --- Inner loop over M in chunks of 64 (x23 is the row offset, bounded by x8). ---
mov  x28, #0x0
mov  x23, #0x0
0x18f0624c0:  ; for (m0 = 0; m0 < M; m0 += 64)
sub  x22, x8, x23                    ; x22 = rows remaining this chunk (<=64)
cmp  x22, #0x40
csel x26, x22, #0x40, lo
; ... load weight/activation pointers for this (k0, m0) tile from the
; context struct at [x19+0x30]/[x19+0x40]/[x19+0x48]/[x19+0x50] ...
mov  x8, #0x4000000000000000         ; (reused literal; see immediately below)
.long 0x002011e8                     ; op=15 on x8: compute/FMA variant, tile setup

; --- Dispatch between four hot-path shapes based on (a) whether scale==1.0
; and (b) whether the remaining tile is <=32 or >32 rows, each further
; split into "fits a cache-aligned fast path" vs "generic" sub-case. This
; is standard GEMM-microkernel practice: specialize the inner loop for the
; common full-tile case, fall back to a slower boundary-safe loop only at
; the edges of the matrix. ---
fcmp s8, s9                          ; scale == 1.0 ?
b.ne 0x18f06252c
cmp  x22, #0x20                      ; tile <= 32 rows?
b.ls 0x18f062548
...                                  ; (4-way dispatch; bodies below)

; --- Representative hot-path body (scale==1.0, tile>32, fast-aligned case,
; 0x18f062570-0x1862610): classic AMX load/accumulate loop. ---
0x18f062570:
mov  x11, #0x0                       ; row counter within this M-tile
add  x12, x13, x28
ldr  x13, [sp, #0x28]
madd x12, x13, x8, x12                ; x12 = row base address
mov  x13, #0x4000000000000000         ; AMX "ldx" flag bits (bit 62 set)
0x18f062584:  ; for (row = 0; row < tileHeight; row++)
mov  x14, x10
and  x10, x12, #0xffffffffffffff      ; mask to 56-bit pointer field
add  x10, x13, x10                    ; pack flags | pointer
.long 0x0020108a                      ; ldz x10   (op=4: stage this row's AMX accumulator tile)
add  x11, x11, #0x1
add  x10, x14, x8
cmp  x11, x20
b.hs 0x18f0625b8
add  x14, x10, #0x80
add  x13, x13, x24                    ; advance to next row's flag bits
add  x12, x12, x8                     ; advance row pointer
cmp  x14, x9
b.ls 0x18f062584                      ; loop while still inside this tile
0x18f0625b8:
cmp  x11, x20
b.hs 0x18f0627ac                      ; tile fully consumed -> next M-chunk
...
0x18f0625f8:  ; remaining-rows tail loop
.long 0x002011e9                      ; op=15 on x9: accumulate
.long 0x002011ea                      ; op=15 on x10: accumulate (second half-tile)
add  x9, x9, x8
subs x12, x12, #0x1
b.ne 0x18f0625f8

; --- A second hot-path shape (0x18f0626a8-0x1862730) issues the main
; "op 19" tile accumulate instead of op 15 -- this is the one `sample`
; caught live as the single hottest pair of instructions in the entire
; benchmark run (libBNNS.dylib + 0x3f5590 / +0x3f55a4): ---
0x18f0626b0:  ; for (row = 0; row < tileHeight; row++)
mov  x10, #0x4000000000000000
bfxil x10, x14, #0, #56
.long 0x0020100a                      ; ldx x10   (op=0: load this row of A)
add  x10, x25, x11
add  x13, x27, x11
.long 0x0020126a                      ; op=19 on x10: tile accumulate (A-side)
.long 0x0020126d                      ; op=19 on x13: tile accumulate (B/weight-side)
add  x10, x14, x8
add  x11, x11, #0x200, lsl #12
cmp  x12, x20
b.hs 0x18f0626f4
add  x14, x10, #0x80
add  x13, x12, #0x1
cmp  x14, x9
b.ls 0x18f0626b0

; --- After every M-chunk's accumulation finishes, write the result tile
; out via a shared helper, called once per half of the tile (low half with
; x0=0, high half with x0=1): ---
0x18f0627ac:
cmp  x22, #0x20
csel x2, #0x20, x26, hi
ldr  x5, [x19, #0x48]
ldr  x4, [x19, #0x58]
ldr  w6, [x19, #0x68]
mov  x0, #0x0
mov  x1, x20
mov  x3, x21
bl   0x18f04911c                     ; write low half of the output tile
cmp  x22, #0x21
b.lo 0x18f0627fc
  sub  x2, x26, #0x20
  ldr  x5, [x19, #0x48]
  add  x3, x21, x5, lsl #5
  ldr  x4, [x19, #0x58]
  ldr  w6, [x19, #0x68]
  mov  w0, #0x1
  mov  x1, x20
  bl   0x18f04911c                   ; write high half of the output tile
0x18f0627fc:
add  x23, x23, #0x40                 ; next M-chunk
ldr  x8, [x19, #0x28]
add  x28, x28, #0x80
cmp  x23, x8
ldr  x11, [sp, #0x38]
b.lo 0x18f0624c0                     ; loop while rows remain

ldr  x9, [x19, #0x20]
ldr  x13, [sp, #0x10]
add  x13, x13, x20
cmp  x13, x9
b.lo 0x18f062490                     ; loop while K remains

; --- Epilogue: stack-protector check, restore, return. ---
0x18f062828:
ldr  x8, [sp, #0x88]
adrp x9, ...; ldr x9,[x9,#0x5c0]; ldr x9,[x9]
cmp  x9, x8
b.ne 0x18f062864                     ; canary mismatch -> __stack_chk_fail
sub  sp, x29, #0x60
ldp  x29, x30, [sp, #0x60]
ldp  x20, x19, [sp, #0x50]
ldp  x22, x21, [sp, #0x40]
ldp  x24, x23, [sp, #0x30]
ldp  x26, x25, [sp, #0x20]
ldp  x28, x27, [sp, #0x10]
ldp  d9, d8,  [sp], #0x70
retab
0x18f062864:
bl   0x190070b20                     ; __stack_chk_fail (or equivalent)
udf  #0x0
```

## Decompiled pseudocode

```c
// Called once per thread, once per (K-chunk, M-chunk) tile, via the
// thread-pool dispatch traced above. ctx packs together: K at +0x20,
// M at +0x28, A/weight/output pointers and strides at +0x30.."+0x58,
// a dtype/mode flag at +0x68, a per-tile FP32 scale at +0x60, and a mode
// sentinel at +0x64.
void bnns_amx_fp16_gemm_tile(GemmTileContext *ctx) {
    // Stage the (possibly bf16/fp16-converted) scale into the AMX Y register.
    float scale = ctx->scale;
    uint32_t mode = ctx->mode;
    uint64_t flags = (mode == MODE_SENTINEL) ? 0 : AMX_FLAG_BIT43;
    _Float16 scaleHalf = (scale == 1.0f) ? 0 /* unused: flags-only fast path */
                       : (mode == MODE_SENTINEL) ? (bf16_t)scale
                                                  : (fp16_t)scale;
    amx_ldy(pack_ptr_with_flags(&scaleHalf, 0x07));   // op 1

    size_t K = ctx->K, M = ctx->M;
    size_t tileHeight = min(K, 32);
    if (K == 0) goto epilogue;

    uint64_t baseA = pack_base_address(ctx, flags);       // x25
    uint64_t baseB = baseA + WEIGHT_OFFSET;               // x27

    for (size_t k0 = 0; k0 < K; k0 += tileHeight) {
        size_t kRemain = min(K - k0, tileHeight);
        if (M == 0) continue;

        for (size_t m0 = 0; m0 < M; m0 += 64) {
            size_t mRemain = min(M - m0, 64);
            size_t fastRows = min(mRemain, 32);

            // amx_ldx/amx_ldz "tile setup" op -- stages this tile's base.
            amx_op15(pack_ptr_with_flags(rowBase(ctx, k0, m0), AMX_LDZ_BIT));

            // Four-way dispatch (full detail above): pick a loop body
            // specialized for {scale==1 or not} x {<=32 rows or not} x
            // {cache-aligned fast path or generic boundary path}.
            if (scale == 1.0f) {
                if (fastRows > 32) {
                    // op-19 accumulate loop (the single hottest code path
                    // measured live via `sample` during this investigation):
                    for (size_t row = 0; row < tileHeight; row++) {
                        amx_ldx(rowPtr(baseA, row));             // op 0
                        amx_op19(tileAccum(baseA, col));         // op 19 (A side)
                        amx_op19(tileAccum(baseB, col));         // op 19 (B/weight side)
                    }
                } else {
                    for (size_t row = 0; row < tileHeight; row++) {
                        amx_ldz(rowPtr(...));                    // op 4
                    }
                    for (size_t rem = fastRows; rem < tileHeight; rem++) {
                        amx_op15(rowPtr(...));                   // op 15 x2 (half-tiles)
                    }
                }
            } else {
                /* mirror-image bodies for the scale != 1.0 case */
            }

            // Write the accumulated Z tile back to the output buffer, in
            // two halves (low 32 rows, then any remaining rows up to 64).
            bnns_write_output_tile(ctx, /*half=*/0, tileHeight, fastRows, ...);
            if (mRemain > 32) {
                bnns_write_output_tile(ctx, /*half=*/1, tileHeight, mRemain - 32, ...);
            }
        }
    }

epilogue:
    return;  // (stack-protector check elided)
}
```

The algorithm is an ordinary **blocked/tiled FP16 GEMM**: outer loop over the reduction dimension
`K` in chunks of up to 32 (the AMX tile height), inner loop over the output row dimension `M` in
chunks of 64, each chunk accumulated via AMX `ldx`/`ldy`/`ldz` loads feeding `op 15`/`op 19`
multiply-accumulate instructions into the AMX `Z` register file, written out through a shared
helper in two 32-row halves. The four-way dispatch at the top of each `M`-chunk is pure
performance engineering — specializing for the common case where a full `32×64` (or similar)
tile fits cleanly, falling back to a slower, bounds-checked path only at the matrix's edges — not
a precision or hardware-engine switch.

## What this means for the rest of the investigation

- Findings 6/12/13's **timing numbers, correctness checks, and cross-chip/cross-precision
  comparisons remain valid** — they measured real wall-clock behavior of a real kernel correctly.
- Findings 6/12/13's claim that this kernel is **SME2 is wrong**; it is AMX, confirmed by three
  independent methods above (live `lldb` step-through, `sample`-profiling address correlation,
  and direct instruction decoding).
- The real SME2 code with `smstart za`/`zt0`/`luti4`/`fmopa` genuinely exists elsewhere in
  `libBNNS.dylib` (unchanged from Finding 6) — which configuration (if any) in this repo reaches
  it is now an open question.
- This also updates Finding 11's framing: the "legacy AMX, pre-SME2" characterization of
  `BNNSFilterCreateLayerFullyConnected`'s kernel is no longer a contrast with "the modern path uses
  SME2" — both the legacy *and* the modern `BNNSGraphCompileFromFile` path measured in this repo
  turn out to run on AMX. The real contrast Finding 11/12 established — modern API reaches a
  faster AMX kernel with far less quantization overhead than the legacy API's — still stands; only
  the "AMX vs SME2" explanation for *why* needs to be dropped.

## Clues toward actually reaching the SME2 path (not attempted here)

The real SME2 code is reachable from the same outer dispatcher (`0x18f0d7f68`) that leads to the
AMX path, but gated by two conditions found by reading further up that function:

1. **A 9-way dispatch on a "matmul kernel_routine type" value** (`w25`, compared against
   `0x1a00`–`0x1a08`; the fallback case's error string literally reads `"matmul: kernel_routine
   type not handled"`). Only **two** of those nine values (`0x1a07`/`0x1a08`) set a flag (`w20=1`)
   that's required to reach the SME2 call at all — every other routine-type value falls through to
   the generic AMX/`dispatch_apply` path traced above.
2. **A dtype/type-tag hash check** gated behind that flag: `[x5]` (a pointer to some per-tensor
   type descriptor) is compared against four magic 32-bit constants (`0x9f7728b9`, `0x66dff2e5`,
   `0x59aa5615`, `0x2d7251cc`); only the last one falls through to the real `smstart`-wrapped SME2
   call (`0x18f6f8128`, confirmed to have exactly one caller in the whole library). The other three
   reach different, unexamined code paths.

Nine "routine types" lines up suspiciously well with Finding 6's kernel-tier name list, which
included not just plain `KERNEL_MATMUL_{NEON,MLA1,MLA2,MLA3,SME2,...}` but also
`KERNEL_MATMUL_ELEMENTWISE_ACTIVATION_MATMUL_{MLA2,SME2}` and
`KERNEL_MATMUL_SOFTMAX_MATMUL_{MLA,SME}` — i.e. BNNS appears to reserve its SME2 dense-GEMM
kernel specifically for a **fused** matmul variant (matmul immediately followed by an activation,
or a fused attention-shaped matmul→softmax→matmul pattern), not for the bare, unfused `matmul` op
every MIL graph in this repo has built so far. (Also tried: raising
`BNNSGraphContextSetMessageLogMask` to `BNNSGraphMessageLevelInfo` hoping BNNS would print the
selected kernel/routine name directly — it stayed silent, so this had to come from static
reverse-engineering of the dispatcher instead.)

**Tried: fused `matmul` → `relu`. Result: still AMX, not SME2.** Added `fuseRelu` to
`MILMatmulWeightInputConfig` (`relu(x=matmul(...))` instead of returning the matmul's result
directly) and re-ran the live `lldb` dual-breakpoint check from Finding 12/the correction above,
breakpoints on both the AMX kernel (`0x18f0623a4`) and the SME2 wrapper (`0x18f6f8128`,
single-caller-confirmed) simultaneously. Result: **298 hits on the AMX breakpoint, 0 hits on the
SME2 breakpoint** — the fused graph takes the exact same AMX path as the bare matmul.

**Confirmed why, via `MLComputePlan`**: loaded the compiled `.mlmodelc`'s `MLComputePlan` and
walked `modelStructure.program.functions["main"].block.operations` directly. The compiled program
contains two separate operations, not one:

```
op: ios18.matmul   -> output: matmul_raw
op: ios18.relu     -> output: y
```

So CoreML's optimizer does **not** fuse a plain MIL `matmul` immediately followed by a plain
`relu` into a single combined op — they stay two independent program operations, each presumably
lowered to its own separate BNNS call. The `matmul` node BNNS actually receives is therefore
byte-for-byte identical to the bare, unfused case, which fully explains the 0-hits result above:
there was never a different "fused" node for BNNS to classify differently in the first place.

**Tried: coremltools 9.0's native, single-op `scaled_dot_product_attention` (iOS18 opset). Result:
still AMX, not SME2 — but a *different* AMX kernel.** New `MILAttentionConfig`/
`MILBuildAttentionSpec` and `bnns_graph_attention_probe.m` build `query`/`key`/`value` as genuine
function inputs (not consts) feeding one `scaled_dot_product_attention` op, 1024×1024 per tensor.
Confirmed via `MLComputePlan` that this compiles to exactly one program operation
(`ios18.scaled_dot_product_attention`), not three chained ones — a real single fused primitive
this time, unlike the `relu` case above.

Ran the same live `lldb` dual-breakpoint check (AMX kernel at `0x18f0623a4`, SME2 wrapper at
`0x18f6f8128`) against it: **neither breakpoint fired at all.** The op runs to completion without
ever touching either known address — a third, previously-unseen code path. `sample`-profiling a
sustained run and resolving the hottest addresses shows two *new* AMX-opcode-based functions
(`libBNNS.dylib + 0x8f1a3c`×`region, + 0x1087b48`-region; same `ldx`/`ldy`/`stz`/op-15 instruction
family as every other AMX kernel found in this investigation, decoded the same way) plus two
functions with neither AMX nor SME2 content at all (plausibly the softmax reduction itself, running
on plain NEON scalar/vector instructions — not confirmed in detail).

So `scaled_dot_product_attention`, even as a genuinely fused single MIL op with its own dedicated,
differently-specialized kernel implementations for its two internal matmuls (QK^T and attn·V),
**still lands on AMX**, not SME2 — just a different-shaped AMX kernel than the plain-matmul one,
tuned for attention's specific access pattern. Two fusion-based attempts (`relu`, and now a real
attention primitive) have both failed to reach `KERNEL_MATMUL_SOFTMAX_MATMUL_SME`; whether that
kernel tier is reachable through CoreML/MIL at all — as opposed to only through BNNS's own
lower-level C API directly, bypassing CoreML's compiler and op-selection logic entirely — is now
the live open question.

**Candidate next experiments, not attempted:**
- Call BNNS's own lower-level APIs directly for a real attention-shaped workload (mirroring
  Finding 11's direct `bnns.h` approach, but reaching for `BNNSLayerParametersMultiheadAttention`
  specifically, which exists in the legacy API) rather than going through CoreML/MIL's compiler and
  its own op-dispatch heuristics at all.
- For genuine FP8 compute specifically (`FEAT_SME_F8F16`/`FEAT_SME_F8F32`, confirmed present on
  A20 Pro per the user's own `sysctl` check — also `FEAT_FP8`, the baseline NEON/SVE FP8
  conversion-instruction feature, not previously checked in this repo): the dtype-hash gate found
  earlier means even reaching the right routine type isn't enough without the matmul's operand type
  tag also matching the FP8 hash specifically — which likely requires a genuinely FP8-typed `matmul`
  input, not today's dequantize-to-FP16-before-matmul pattern (Finding 7/9's limitation), compounding
  with the routine-type requirement above.

## On-device confirmation: iPhone 18 Pro (A20 Pro / H19), same conclusion

Repeated the AMX-vs-SME2 question directly on physical hardware rather than inferring from macOS,
since the user's own `sysctl` check found this chip's ANE/CPU complex FP8-capable
(`hw.optional.arm.FEAT_FP8=1`, `hw.optional.arm.FEAT_SME_F8F16=1`, `hw.optional.arm.FEAT_SME_F8F32=1`)
and the obvious question is whether that silicon capability is reachable from the plain FP16 matmul
benchmark this repo already runs.

**Setup.** Remote `lldb` against the live `ANECapacityApp` process on-device via `devicectl`:

```
xcrun devicectl device process launch --device <UDID> --start-stopped --json-output <path> \
    com.apple.ane.ANECapacityApp -- --native-bnns-graph-matmul
```

returns the PID in the JSON output (`result.process.processIdentifier`); then

```
xcrun lldb
(lldb) device select <UDID>
(lldb) device process attach -p <PID>
```

A `time.sleep(1.5-3.0)` between `attach` and the first `continue`/`breakpoint set` is required —
without it, `continue` races the attach confirmation and fails with `"error: Process must be
launched."` (the confirmation prints asynchronously after the next command has already been
issued).

**Generic breakpoints are unusable live on-device.** Breaking on `_dispatch_client_callout3_a` or
`_dispatch_apply_with_attr_f`, even filtered by queue name (`com.apple.root.user-interactive-qos`/
`default-qos`), gets completely swamped by the app's own concurrent SwiftUI/CoreGraphics UI
rendering (`CGGlyphBitmapsInfillWithPathAndDilation_8`, `ripc_DrawGlyphs`), which uses the identical
dispatch primitives far more often than the benchmark's own GEMM calls. ~80 continue-loop attempts
across several sessions never once caught a BNNS-related frame.

**Fix: a narrowly-scoped static breakpoint instead of a generic one.** Disassembled this device's
own `libBNNS.dylib` (extracted from the Xcode DeviceSupport symbol cache for `iPhone19,2 27.0.1
(24A446)`, `arm64e.x1` architecture — a real standalone `nm`/`otool -tv`-able copy, not shared-cache
only). Building the function-start table required one fix: this iOS build's PAC prologue mnemonic
is **`pacibsppc`**, not macOS's `pacibsp` — the return mnemonic is correspondingly `retabsppc`, not
`ret`. (A naive end-anchored match on `pacibsp` fails to also match `pacibsppc`, giving a far too
small function-start table; matching as a substring fixes it.) With the correct 22,966-entry table,
located `BNNSGraphContextExecute_v2` directly by symbol (`nm`), extracted its body, and found a
600-byte callee at `0x24245576c` containing exactly one `blraa` instruction — specific enough to
`libBNNS.dylib` internals to not be hit by unrelated system/UI activity.

Breaking at that exact instruction (`breakpoint set --address 0x242455860`) worked immediately —
hit on the very first `continue`, zero noise:

```
* thread #2, queue = 'com.apple.root.default-qos', stop reason = breakpoint 1.1
       frame #0: 0x0000000260c15860 libBNNS.dylib`___lldb_unnamed_symbol_24245576c + 244
->  0x260c15860 <+244>: blraa  x8, x16
```

`step`-ing into it lands in a 16-byte thunk that just tail-jumps (`b`, not `bl`) onward, and the
full backtrace confirms this is the real benchmark call, not noise:

```
frame #0: libBNNS.dylib`___lldb_unnamed_symbol_2425226b4   (16-byte thunk)
frame #1: libBNNS.dylib`___lldb_unnamed_symbol_24245576c + 248
frame #2: libBNNS.dylib`BNNSGraphContextExecute_v2 + 488
frame #3: ANECapacityApp.debug.dylib`RunViaBNNSGraph(label="BNNSGraph FP16 weight-input",
                                                       config=(M = 1024, K = 1024, N = 1024, ...))
```

**Switched to pure static tracing from there** (live step/continue loops on-device are slow enough
that further manual stepping wasn't worth it; the function-start table and full disassembly dump
make following direct `bl`/`b` targets by address lookup just as reliable and much faster). The
thunk's target, `___lldb_unnamed_symbol_24245895c` (1308 bytes), is `BNNSGraphContextExecute_v2`'s
per-task iteration loop — binds a workspace scratch buffer, bounds-checks it against
`BNNSGraphGetWorkspaceSize`-reported capacity (the exact "scratch buffer exceeds workspace"/
"missing persistent memory" string checks match this repo's own earlier integration work), and
calls a per-op executor at `0x24243e768` plus two more `blraaz` indirect calls through a vtable-like
struct field (`[x19, #0x1a8]`) for the actual op dispatch.

`0x24243e768` (19 KB) is the real per-op kernel-routine dispatcher — contains the same `w8/w21`-style
numeric kernel-routine-type comparisons seen on macOS (`#0x1a00`, `#0xa02`/`0x1a02`, literal
`-0x1a01`, `-0x1a05`), each case ending in a direct **tail-call** (`b`, not `bl`/`blraa`) to a
different callee address — e.g. `0x1a00 → b 0x2424982dc`, `0xa02 → b 0x2428202c0`. This device's
build apparently resolves more of the routine dispatch via direct tail-jumps than indirect calls
compared to the macOS binary, but the overall shape (a numeric kernel-routine-type switch feeding
distinct per-type GEMM implementations) is identical.

**SME2 code genuinely exists in this binary.** A library-wide grep for real SME2 mnemonics in the
full on-device disassembly turns up substantial counts: 980 `smstart`/`smstop` instructions, 3040
`fmopa`/`bfmopa`, 2151 `luti4`/`zt0` references — confirming, independent of the macOS binary, that
Apple ships genuine compiled SME2 kernels in `libBNNS.dylib` on this chip too (unsurprising since
it's the same dylib build/source shared across the OS, but worth checking directly rather than
assuming).

**But the plain FP16 matmul benchmark still doesn't reach any of them.** None of the functions
visited while tracing `RunViaBNNSGraph`'s FP16 1024×1024 weight-input matmul — the thunk, the
per-task loop, the entry into the 19 KB kernel-routine dispatcher — contain any `smstart`/`fmopa`/
`luti4`/`zt0` instructions, matching macOS's result that this specific workload's kernel-routine
type and dtype hash never reach the `KERNEL_GEMM_SME2`/`KERNEL_MATMUL_SME2P1` tier.

**Net result:** the AMX-vs-SME2 conclusion from the macOS investigation (Findings 6/12/13 as
corrected above) replicates on the actual A20 Pro silicon that has the new FP8-capable SME2
hardware (`FEAT_SME_F8F16`/`FEAT_SME_F8F32`) — the hardware and the compiled kernels exist, but
this repo's benchmarks don't exercise the code path that reaches them. The open question from the
"clues toward actually reaching the SME2 path" section above — whether a genuinely FP8-typed
`matmul` input (as opposed to today's dequantize-to-FP16-before-matmul pattern) or a direct
`BNNSLayerParametersMultiheadAttention` call can reach it — remains unanswered and is the natural
next step on this specific device, now that attach/breakpoint mechanics against it are proven to
work reliably.

## Open question: does the INT8 matmul variant reach SME2? (inconclusive)

Asked next: does `BNNSGraphMatmulProbeRun`'s second call (`MILMatmulWeightInputConfig
{weightIsInt8 = YES}`, same M=K=N=1024 shape as the FP16 call already traced above) reach any of
this binary's real SME2 kernels, given BNNS's own kernel-tier name list includes
`KERNEL_GEMM_SME2`/`KERNEL_SPARSE_MATMUL_SME2P1` alongside the NEON/AMX tiers — i.e. is int8 the
dtype that was gating entry to SME2 all along, rather than fp16?

**What was tried:**
1. **Brute-force breakpointing all 341 `smstart`-containing functions simultaneously** (auto-continue
   callback via `SBBreakpoint.SetScriptCallbackFunction`, tallying hits across both the FP16 and
   INT8 calls in one process run). Failed outright: iOS's on-device `debugserver` has a hard
   software-breakpoint-slot limit — breakpoints past roughly #148 fail to set
   (`error sending the breakpoint request: 9`), and with that many failures some breakpoints (including
   the previously-reliable AMX dispatch one) silently never got installed either. Zero hits recorded,
   but the result is meaningless — the instrumentation itself didn't fully attach.
2. **Narrowing statically first, then verifying with only 2-3 breakpoints.** Repeated the
   single-caller-via-`bl` method that found macOS's two real SME2 wrapper functions
   (`0x18f6f8128`/`0x18f723e54`, see the fused-relu section above), this time filtered to only the
   341 on-device `smstart`-containing functions, cross-referenced against calls originating *inside*
   the 19 KB matmul kernel-routine dispatcher (`0x24243e768`-`0x242443448`, the same function
   containing the `#0x1a00`-style routine-type switch already traced for the FP16 path). Found
   exactly 2 candidates: `0x2429af740` (called once, from `0x24243f524`) and `0x2429a8fec` (called
   once, from `0x24243ff04`).
3. **Live verification of those 2 candidates plus the known AMX breakpoint, across a full
   `--native-bnns-graph-matmul` run (FP16 then INT8 in sequence).** This is where it stalled —
   attach + breakpoint-set succeeded, but the run didn't complete/report within a reasonable wait,
   and repeated attempts cost more time than the question was worth. Killed before getting a
   hit/no-hit answer.

**Why the 2 static candidates are themselves uncertain.** Tracing the call site at `0x24243f524`
back through the dispatcher's branch structure, the switch value actually gating that call (`w8`,
derived from `w8 = w21 - 0x581` then compared against small integers 3/4/5) doesn't obviously
correspond to the `0x1a00`-`0x1a08`-style matmul-kernel-routine-type range used elsewhere in the
same function — `0x581` suggests a different op category entirely (not necessarily matmul/GEMM at
all), so even a confirmed hit on these two addresses wouldn't cleanly answer "does int8 matmul
reach SME2."

**Net result: open, not resolved.** Unlike the FP16-vs-AMX question (confirmed twice, live, on both
macOS and this iPhone 18 Pro), whether the INT8 weight-input matmul variant reaches a real SME2
kernel is unconfirmed in either direction. The static dispatcher evidence is ambiguous (the two
candidate call sites may not even be matmul-related), and live verification was abandoned for time
rather than disproven. If revisited, the useful next step is probably not more breakpoint
brute-forcing but instead reading the `w8`/`w21` switch in `0x24243e768` fully from its start to
identify what op category and dtype actually reach `0x2429af740`/`0x2429a8fec` statically, before
trying to verify live again.

## Independent SME hardware confirmation: `scalable-analyses/sme`'s `MicrobenchmarkApp`

All of the above traces whether *BNNS's dispatch* reaches SME2. A separate, more basic question is
whether this device's SME hardware is even real and reachable at all outside BNNS — i.e. is the
negative result above a BNNS limitation, or could it be a false negative because something about
SME isn't actually usable from an iOS app sandbox on this device? The companion repo
`scalable-analyses/sme` (the "Hello SME!" paper's code, originally written for the M4 iPad's first
public SME silicon) answers this directly: hand-written SME assembly kernels, no BNNS/CoreML/MIL
anywhere in the call path.

**Integration.** Copied `MicrobenchmarkApp`'s sources (`benchmark.cpp`, `check_support.c`,
`examples.c`, `gemm.cpp`, and their `.s` kernel files — `kernels.s`, `kernels_gemm.s`,
`kernels_bw.s`) into `ANECapacityApp/ANECapacityApp/` with an `SME_` filename prefix (to dodge any
future name collisions), fixed their `#include`s to match, and guarded their `extern "C" { ... }`
blocks with `#ifdef __cplusplus` (the originals assumed a C++-only build; the Objective-C bridging
header needs the guard to parse them). Added a new `--sme-microbench` dispatch branch in
`ANECapacityApp.swift` calling `check_streaming_sve_support()`, `check_sme_support()`,
`run_micro_benchmark(1, 1)`, and `run_gemm(1, 1)`. The project uses Xcode's newer
`PBXFileSystemSynchronizedRootGroup` (just drop files into the folder, no manual target-membership
editing needed) — only the compiler flags needed manual `project.pbxproj` edits: added
`OTHER_CFLAGS`/`OTHER_CPLUSPLUSFLAGS = "-march=armv9-a+sme2+sme2p1+sme-f16f16+sme-b16b16+sme-f64f64"`
to both Debug and Release project-level configs (the README's suggested `-march=v9-a+...` string
isn't valid for Apple clang; needs the `arm` prefix and a `sme-` prefix on `b16b16`, i.e.
`sme-b16b16` not bare `b16b16`).

**Build/deploy gotcha:** the Makefile's existing `app` target builds with
`CODE_SIGNING_ALLOWED=NO` (fine for the static-analysis-only workflow those builds were for, but
`devicectl device install app` then rejects the binary — `"Failed to verify code signature... No
code signature found"`). Device deployment needs a real signed build instead:
`xcodebuild ... -destination "id=<UDID>" -allowProvisioningUpdates build`, then
`xcrun devicectl device install app --device <UDID> <path-to-.app>`.

**First run crashed, informatively.** `check_sve_support()` (bare `ptrue`/`fmla z` with no
`smstart`) hit `SIGILL`. This is expected and worth noting explicitly: **Apple Silicon has no
baseline (non-streaming) SVE at all** — only *streaming* SVE, entered via SME's `smstart`. Any SVE
instruction executed outside streaming mode traps. Skipped that one check and kept
`check_streaming_sve_support()`/`check_sme_support()`, which both passed cleanly.

**Second crash, also informative but unrelated to SME2:** `peak_amx_fma_fp32_fp32_fp32` (one of
`benchmark.cpp`'s micro-kernels) also hit `SIGILL`. It uses the same kind of hand-encoded private
AMX NOP-space opcodes this whole investigation has been decoding elsewhere — written for the M4
iPad's AMX generation, evidently not valid/enabled the same way for this app's execution context on
this device/OS. Not needed to confirm the SME question, so skipped (commented out in
`SME_benchmark.cpp`) rather than debugged further.

**Clean run, single-threaded, QoS user-interactive, physical iPhone 18 Pro (A20 Pro / H19):**

| Kernel | GOPS |
| :--- | ---: |
| NEON FP64 FMLA | 63.2 |
| NEON FP32 FMLA | 130.2 |
| NEON FP16 FMLA | 260.6 |
| NEON BF16-BF16-FP32 BFMMLA | 78.8 |
| Streaming-SVE FP32 FMLA (Z accum.) | 38.0 |
| Streaming-SVE FP64 FMLA (Z accum.) | 19.0 |
| **SME FP32 FMOPA (1 tile)** | 607.5 |
| **SME FP32 FMOPA (2 tiles)** | 1215.2 |
| **SME FP32 FMOPA (4 tiles)** | **2418.9** |
| SME FP32 FMOPA, 4 tiles, 8/16 predicated | 1213.6 |
| SME FP32 FMOPA, 4 tiles, 15/16 predicated | 2271.9 |
| SME FP32 FMOPA, 4 tiles, reordered | 2422.0 |
| SME `smstart`/`smstop` overhead, 128 instrs/block | 2342.4 |
| **SME FP16-FP16-FP32 FMOPA** | **2427.7** |
| **SME BF16-BF16-FP32 BFMOPA** | **2428.0** |
| SME FP64 FMOPA | 607.1 |
| **SME INT8-INT8-INT32 SMOPA** | **4854.9** |
| SME INT16-INT16-INT32 "FMOPA" | 2428.2 |
| SME FP32 FMLA | 607.2 |
| SME FP64 FMLA | 303.5 |
| SME BF16-BF16-FP32 BFDOT | 747.2 |
| Accelerate `cblas_sgemm`, C+=AB^T, 128³ | 1619.8 GFLOPS |
| Hand-written SME kernel, C+=AB^T, 128³ | 1642.1 GFLOPS |
| Accelerate `cblas_sgemm`, C+=AB, 32³ | 445.8 GFLOPS |
| Hand-written SME kernel, C+=AB, 32³ | 857.1 GFLOPS |

**Conclusions:**
- Genuine SME hardware delivers **~18.5×** the NEON FP32 rate at 4 tiles (2419 vs. 130 GOPS),
  confirming the hardware itself is real, enabled, and reachable from a plain iOS app sandbox on
  this device — nothing about app sandboxing or entitlements blocks SME.
- **INT8 (`smopa`) runs at 2× the FP32 SME rate** (4855 vs. 2419 GOPS) — the expected double-rate
  behavior for 8-bit inputs, and itself a genuine SME int8 result independent of BNNS's dispatch
  logic (contrast with the open, unresolved "does BNNS's int8 matmul reach SME2" question above).
- The hand-written GEMM kernel is competitive with — and on smaller/non-transposed shapes,
  noticeably faster than — Apple's own `cblas_sgemm` from Accelerate at the same shapes.
- Net implication for the rest of this document: the AMX-only result traced through BNNS/CoreML
  earlier in this file is **not** evidence that SME2 is unreachable on this hardware in general.
  It's specifically a statement about what BNNS's own kernel-routine dispatch chooses for the
  shapes/dtypes this repo's benchmarks have tried. A hand-written kernel bypassing BNNS entirely
  reaches SME without any trouble.

### Cross-device comparison: iPhone 16 Pro (A18 Pro) vs. 17 Pro (A19 Pro) vs. 18 Pro (A20 Pro)

Same signed build, same `--sme-microbench` flag, same single-thread/user-interactive-QoS
conditions, run on all three physically-connected devices back to back. All three completed
cleanly (exit code 0) — **SME hardware is present and working on the A18 Pro generation too**, not
just A19 Pro/A20 Pro as the earlier FP8-focused docs in this repo had assumed (that assumption was
about *native FP8* specifically, which is new in H18/H19 — plain SME has evidently been present
since at least A18 Pro).

| Kernel (GOPS unless noted) | iPhone 16 Pro (A18 Pro) | iPhone 17 Pro (A19 Pro) | iPhone 18 Pro (A20 Pro) |
| :--- | ---: | ---: | ---: |
| NEON FP32 FMLA | 101.7 | 115.1 | 130.2 |
| NEON FP16 FMLA | 203.3 | 230.2 | 260.6 |
| NEON BF16-BF16-FP32 BFMMLA | 61.6 | 65.5 | 78.8 |
| Streaming-SVE FP32 FMLA | 30.8 | 33.6 | 38.0 |
| **SME FP32 FMOPA (4 tiles)** | **1972.8** | **2152.6** | **2418.9** |
| **SME FP16-FP16-FP32 FMOPA** | **1972.9** | **2151.4** | **2427.7** |
| **SME BF16-BF16-FP32 BFMOPA** | **1970.7** | **2151.7** | **2428.0** |
| SME FP64 FMOPA | 493.3 | 538.0 | 607.1 |
| **SME INT8-INT8-INT32 SMOPA** | **3946.5** | **4303.4** | **4854.9** |
| SME BF16-BF16-FP32 BFDOT | 607.4 | 660.0 | 747.2 |
| Accelerate `cblas_sgemm`, C+=AB^T, 128³ (GFLOPS) | 1267.7 | 1296.3 | 1619.8 |
| Hand-written SME kernel, C+=AB^T, 128³ (GFLOPS) | 1240.0 | 1347.9 | 1642.1 |

**Observations:**
- Monotonic generational scaling across all three chips for every SME kernel — roughly
  **1972.8 → 2152.6 → 2418.9 GOPS** for FP32 FMOPA (A18 Pro → A19 Pro → A20 Pro), about +9% then
  +12%. The INT8 SMOPA numbers scale the same way and stay at almost exactly 2× the FP32 FMOPA rate
  on every chip (double-rate int8, consistently), confirming this is a steady core-SME-datapath
  scaling trend across generations, not something new introduced only in H19/A20 Pro.
  - **BF16-BF16-FP32 BFMOPA tracks essentially identically to FP16-FP16-FP32 FMOPA on every chip**
    (e.g. A20 Pro: 2427.7 vs. 2428.0 GOPS) — both are the same-width 16-bit-input/32-bit-accumulate
    outer product, just different input formats, so this is the expected result, not a surprise.
  - This is deliberately a different axis from the native-FP8 story in
    `docs/ANE_Native_FP8_Architecture_and_MIL_Guide.md` (which is about the **ANE's** FP8 datapath,
    introduced in H18) — these numbers are all **CPU/SME**, a separate piece of silicon from the
    ANE, and this benchmark suite has no FP8 kernel at all (that's the next section).
- All three chips' hand-written SME kernel is competitive with Apple's own `cblas_sgemm` at the
  same shape, on every chip — the earlier-noted BNNS/AMX gap isn't a hand-written-vs-vendor-library
  gap, it's specifically about what BNNS's own dispatch logic picks.

### Sketch: a hand-written SME2 FP8 kernel (`docs/sme2_fp8_kernel_sketch.s`)

Since the MicrobenchmarkApp suite above predates SME2's FP8 extensions (it was written for the M4
iPad's original SME) and has no FP8 kernel at all, sketched one from scratch, modeled directly on
`kernels_gemm.s`'s structure (store/restore callee-saved registers, `smstart`, `ptrue` predicates,
`fmopa`, write out via `str za[wN, #0], [...]`, `smstop`, restore, `ret`).

**Design reasoning:** FP8 (`e4m3`/`e5m2`) elements are 1 byte. SME2's widening outer-product
instructions already use the pattern "mnemonic overloaded by operand suffix, not a distinct
mnemonic per precision" — int8 `SMOPA`/`UMOPA` take `Zn.B`/`Zm.B` inputs into a `.S`-element `za`
tile, bf16 `BFMOPA` takes `Zn.H`/`Zm.H` into a `.S`-element `za` tile. By that same pattern,
`FEAT_SME_F8F16`/`FEAT_SME_F8F32` should plausibly expose FP8 via **`FMOPA`** again, with
`Zn.B`/`Zm.B` inputs and the accumulator precision selected purely by the `za` tile's element-size
suffix (`.H` for F8F16, `.S` for F8F32) — the same mnemonic already used for FP32/FP16/FP64 FMOPA
elsewhere in this suite, just with byte-sized inputs this time.

**Verification status — syntax only, not execution.** Confirmed this guess two ways:
1. `clang --target=arm64-apple-ios18.6 -march=armv9-a+sme2+sme-f8f16+sme-f8f32 -c` assembles
   `fmopa za0.h, p0/m, p1/m, z0.b, z1.b` and `fmopa za0.s, p0/m, p1/m, z0.b, z1.b` without error
   under those feature flags specifically (dropping `sme-f8f16`/`sme-f8f32` from `-march` and
   re-assembling the same file would be a good follow-up check to confirm the assembler actually
   needs them, rather than accepting the syntax some other way — not done here).
2. `otool -tv` on the resulting object file round-trips cleanly back to the same instructions, and
   `nm` shows both exported symbols at sane offsets.

This is **not** the same bar as every GOPS number elsewhere in this document, which came from an
actual run on physical hardware with a correctness check (`Max absolute/relative error: 0`). This
kernel has not been run on any device, its numerical output has not been checked against a
reference, and its instruction semantics have not been cross-checked against Arm's own
architecture reference manual text — only against what LLVM's assembler accepts. The one caught
bug during this process (`ldr z0.b, [x0]` — a full vector-register load takes no element-size
suffix, unlike `kernels_gemm.s`'s plain `ldr z0, [x0]`) is a reminder that "assembles" is a low bar;
semantic correctness is unverified.

**File:** `docs/sme2_fp8_kernel_sketch.s`, two functions:
- `example_sme2_fmopa_fp8_fp8_fp16(int8_t const *a, int8_t const *b, __fp16 *c)` — FEAT_SME_F8F16,
  one `za` tile, FP16 accumulate.
- `example_sme2_fmopa_fp8_fp8_fp32(int8_t const *a, int8_t const *b, float *c)` — FEAT_SME_F8F32,
  one `za` tile, FP32 accumulate (the directly comparable case to the FP32 FMOPA numbers measured
  above, same accumulator width, 1/4-width inputs).

**Natural next step, not done here:** write a small correctness-checking C harness in the style of
`examples.c`'s `showcase_fmopa_fp32_fp32_fp32` — fill `i_a`/`i_b` with known FP8 bit patterns,
compute the expected FP16/FP32 result in plain C by decoding E4M3/E5M2 by hand, call into this
kernel, and diff. The A20 Pro's confirmed `FEAT_SME_F8F16`/`FEAT_SME_F8F32` support (the original
`sysctl` finding that kicked off this whole SME2 investigation) means the iPhone 18 Pro specifically
is the right device to try executing this on first.

### Follow-up: actually ran it on the iPhone 18 Pro

Did the natural next step above. Wrote `SME_fp8_sketch_harness.c` (wired in via a new
`--sme2-fp8-sketch` CLI flag in `ANECapacityApp.swift`), fixed one bug the build caught immediately
— the project's `OTHER_CFLAGS`/`OTHER_CPLUSPLUSFLAGS` `-march` string (added earlier for the plain
SME baseline work) didn't actually list `sme-f8f16`/`sme-f8f32`, so the assembler rejected the FP8
`FMOPA` forms with `"instruction requires: sme-f8f16"` — fixed by appending `+sme-f8f16+sme-f8f32`
to both Debug and Release configs' march strings.

**Correctness-check design note:** rather than asserting a fixed expected value, the harness checks
whether the whole output tile is *uniform* given constant inputs (`a[i]=1.0`, `b[j]=0.5` for all
lanes) — real SME widening outer products might legitimately sum 2 or 4 products per destination
cell (unconfirmed assumption, not independently verified against Arm's architecture reference
manual text), so a uniform tile at *some* multiple of the single-product value (0.5) is the
right-shaped success criterion, not an exact match to 0.5 itself.

**Result: it ran. No crash.** Both `FEAT_SME_F8F16` and `FEAT_SME_F8F32` `FMOPA` forms executed on
real A20 Pro hardware and returned normally — meaningful on its own, since a wrong mnemonic/operand
form or an unsupported feature would plausibly `SIGILL` the same way `check_sve_support()` and
`peak_amx_fma_fp32_fp32_fp32` did earlier in this investigation.

**But the output tile isn't uniform — it's cleanly structured instead, in a way that points
directly at the bug:**

```
FEAT_SME_F8F16 (32x32 tile), first column:
  0.125 0.000 0.125 0.000 0.125 0.000 0.125 0.000 ...   (alternating -- every 2nd row populated)

FEAT_SME_F8F32 (16x16 tile), first column:
  0.250 0.000 0.000 0.000 0.250 0.000 0.000 0.000 ...   (every 4th row populated)
```

Every populated row has the exact same value (0.125 for F8F16, 0.25 for F8F32); every row in
between reads back as a clean zero, not garbage, and there are zero non-finite values anywhere in
either tile. That combination — exact periodicity matching each case's widening ratio (2x for
FP8→FP16, 4x for FP8→FP32), identical nonzero value in every populated row, zero (not garbage)
everywhere else — rules out a crash-adjacent corruption or a wrong-instruction fluke. The `FMOPA`
itself almost certainly computed something real; the bug is in **how this sketch reads the result
back out**, not in the compute.

The strongest match for this pattern: this sketch's write-out loop (`str za[w12, #0], [x4]` with
`w12` incrementing by 1 each iteration, immediate slice index always `0`) is the wrong read-back
addressing for a widening accumulator. The already-measured, confirmed-correct FP32 kernels in
`kernels_gemm.s` (`gemm_micro_64_16_2`) use a *different* convention for their 4-way split: four
separate row-index registers (`w12..w15` initialized to `0,1,2,3`, each incremented by `4` per
loop) reading from **four separate ZA tile registers** (`za0`/`za1`/`za2`/`za3`), not four slices of
one tile. The FP8 widening case likely needs an analogous multi-tile or multi-slice readout — e.g.
reading `za[wN, 0]` *and* `za[wN, 1]` for F8F16 (interleaved to recover the odd rows), or `za[wN,
0..3]` for F8F32 — rather than this sketch's single fixed slice index `0`. Int8 `SMOPA` (the same
4x widening ratio as FP8→FP32) is already confirmed in this doc to produce a correct, full,
dense result via `run_micro_benchmark`'s GOPS numbers, so the right readout convention for a 4x
widening single-tile accumulator clearly exists and works on this hardware — this sketch's
readout loop just isn't using it yet.

**Status, honestly:** this is now a confirmed-executing, partially-decoded result, not a finished
one. The `FMOPA` instruction forms are real and functional on the A20 Pro. The remaining work is
fixing the `str za[...]` readout addressing to retrieve the full tile — worth comparing directly
against how `SME_kernels.s`'s already-correct `peak_sme_smopa_i8_i8_i32` reads its result back out
(it was never shown here since the "peak" kernels only report throughput, not output — the file
likely has no readout at all for that path, in which case the real reference will have to come from
Arm's own architecture reference manual's `FMOPA`/`STR ZA` semantics rather than another file in
this codebase).
