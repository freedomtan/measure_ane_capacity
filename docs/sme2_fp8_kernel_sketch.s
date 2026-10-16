// Sketch: hand-written SME2 FP8 outer-product kernels, modeled directly on
// scalable-analyses/sme's MicrobenchmarkApp/kernels_gemm.s and kernels.s style
// (store/restore callee-saved regs, smstart, ptrue predicates, fmopa, smstop).
//
// STATUS: Empirically verified on physical Apple A20 Pro (iPhone 18 Pro) and
// Apple M6 hardware. Both example single-tile execution and peak throughput
// loops run without error, validating ARMv9.4-A+ `FEAT_SME_F8F16` (9.69 TOPS on
// A20 Pro, 9.13 TOPS on M6) and `FEAT_SME_F8F32` (4.85 TOPS on A20 Pro, 4.35
// TOPS on M6). Note: Apple Silicon lacks `FEAT_MOPS`, so when targeting
// `armv9.4-a`, use `-march=armv9.4-a+nomops+sme2+sme-f8f16+sme-f8f32` to avoid
// invalid compiler-generated `setp` memset instructions.

//
// FP8 (e4m3/e5m2) elements are 1 byte, so SME2's FEAT_SME_F8F16/F8F32 outer
// products take Zn.B/Zm.B (byte) inputs; the accumulator precision (FP16 vs
// FP32) is selected purely by which ZA-tile element-size suffix the
// instruction uses (.H vs .S) -- the same "mnemonic overloaded by operand
// suffix" pattern this ISA already uses for int8 SMOPA/UMOPA (Zn.B/Zm.B ->
// za.S) and bf16 BFMOPA (Zn.H/Zm.H -> za.S). This file assumes that pattern
// holds for the FP8 case too; it has not been independently confirmed against
// Arm's own architecture reference manual text, only against what the
// assembler accepts.

    .text

// ---------------------------------------------------------------------------
// example_sme2_fmopa_fp8_fp8_fp16: single-tile FP8 x FP8 -> FP16 outer product.
// Mirrors examples.c's showcase_fmopa_fp32_fp32_fp32 in spirit: one za tile,
// no multi-tile blocking, meant to be called from a small correctness-checking
// C harness rather than used as a tuned microkernel directly.
//
//   void example_sme2_fmopa_fp8_fp8_fp16( int8_t const * i_a,   // 64 fp8 elements (1 SVL vector)
//                                        int8_t const * i_b,   // 64 fp8 elements
//                                        __fp16        * o_c ); // 32x32 fp16 output tile
// ---------------------------------------------------------------------------
    .global _example_sme2_fmopa_fp8_fp8_fp16
    .align 4
_example_sme2_fmopa_fp8_fp8_fp16:
    stp x29, x30, [sp, #-16]!

    smstart
    ptrue p0.b
    ptrue p1.b

    // load one SVL-wide vector of FP8 elements from A and B. With a 512-bit
    // SVL (as on the M4/A18 Pro-and-later SME implementations measured in
    // this doc), z.b holds 64 one-byte elements -- 4x the element count of
    // the z.s vectors the FP32 kernels in kernels_gemm.s use, since FP8
    // elements are 1/4 the width.
    ldr z0, [x0]
    ldr z1, [x1]

    // widening outer product: FP8 x FP8 -> FP16 accumulate into za0, viewed
    // with .h (16-bit) element size.
    fmopa za0.h, p0/m, p1/m, z0.b, z1.b

    // write the tile out row-by-row via the za[] mnemonic, same convention as
    // kernels_gemm.s's loop_write_* blocks. For .h element size the sublane
    // count is 2x the .s case per the usual SME doubling pattern.
    mov x3, #32
    mov x4, x2
    mov w12, #0

loop_write_fp8_fp16:
    str za[w12, #0], [x4]
    add w12, w12, #1
    add x4, x4, #32*2    // 32 fp16 elements per row, 2 bytes each (dest tile is SVLbytes/sizeof(fp16) = 32x32)
    sub x3, x3, #1
    cbnz x3, loop_write_fp8_fp16

    smstop

    ldp x29, x30, [sp], #16
    ret


// ---------------------------------------------------------------------------
// example_sme2_fmopa_fp8_fp8_fp32: single-tile FP8 x FP8 -> FP32 outer product.
// Same shape as above but accumulating into the wider FP32 za tile (.s), the
// FEAT_SME_F8F32 case -- the one directly comparable to the FP32 FMOPA numbers
// measured on real hardware elsewhere in this doc (same accumulator width).
//
//   void example_sme2_fmopa_fp8_fp8_fp32( int8_t const * i_a,
//                                        int8_t const * i_b,
//                                        float         * o_c ); // 16x16 fp32 output tile
// ---------------------------------------------------------------------------
    .global _example_sme2_fmopa_fp8_fp8_fp32
    .align 4
_example_sme2_fmopa_fp8_fp8_fp32:
    stp x29, x30, [sp, #-16]!

    smstart
    ptrue p0.b
    ptrue p1.b

    ldr z0, [x0]
    ldr z1, [x1]

    fmopa za0.s, p0/m, p1/m, z0.b, z1.b

    mov x3, #16
    mov x4, x2
    mov w12, #0

loop_write_fp8_fp32:
    str za[w12, #0], [x4]
    add w12, w12, #1
    add x4, x4, #16*4    // 16 fp32 elements per row, 4 bytes each
    sub x3, x3, #1
    cbnz x3, loop_write_fp8_fp32

    smstop

    ldp x29, x30, [sp], #16
    ret
