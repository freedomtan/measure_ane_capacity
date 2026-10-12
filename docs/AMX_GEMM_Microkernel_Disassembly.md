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
