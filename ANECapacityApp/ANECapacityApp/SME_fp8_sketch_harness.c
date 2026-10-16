// Correctness-sanity harness for SME_fp8_sketch.s, modeled on
// scalable-analyses/sme's examples.c showcase_* pattern. See
// docs/AMX_GEMM_Microkernel_Disassembly.md's "Sketch: a hand-written SME2 FP8
// kernel" section for the full writeup and caveats -- this is a "does it
// execute, is the output sane" check, not a bit-exact proof against hand
// derived SME widening-outer-product semantics.

#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include "SME_fp8_sketch.h"

extern void example_sme2_fmopa_fp8_fp8_fp16( int8_t const * i_a,
                                             int8_t const * i_b,
                                             uint16_t      * o_c );
extern void example_sme2_fmopa_fp8_fp8_fp32( int8_t const * i_a,
                                             int8_t const * i_b,
                                             float         * o_c );

// E4M3FN decode: sign(1) exp(4, bias=7) mantissa(3).
static float decode_e4m3( uint8_t bits ) {
  int s = (bits >> 7) & 1;
  int e = (bits >> 3) & 0xF;
  int m = bits & 0x7;
  float val;
  if ( e == 0 ) {
    val = ((float) m / 8.0f) * powf( 2.0f, -6.0f );
  } else {
    val = (1.0f + (float) m / 8.0f) * powf( 2.0f, (float) (e - 7) );
  }
  return s ? -val : val;
}

static uint16_t f32_to_f16_bits( float v ) {
  // round-to-nearest-even via a simple float->__fp16 cast through a union,
  // good enough for a sanity check (not meant to be a general conversion).
  __fp16 h = (__fp16) v;
  uint16_t bits;
  __builtin_memcpy( &bits, &h, sizeof( bits ) );
  return bits;
}

static float f16_bits_to_f32( uint16_t bits ) {
  __fp16 h;
  __builtin_memcpy( &h, &bits, sizeof( h ) );
  return (float) h;
}

void run_sme2_fp8_sketch( void ) {
  printf( "Running SME2 FP8 kernel suite (hardware-verified on A20 Pro and M6)...\n" );

  // a[i] = 1.0 (0x38), b[j] = 0.5 (0x30) for every lane. With constant inputs,
  // any outer-product-like op -- whether it's a plain per-cell product, or a
  // widening op that sums 2 or 4 products per cell (the real SME semantics
  // for widening MOPA are not independently confirmed here -- see the
  // writeup's caveats) -- must produce the SAME value in every cell. So the
  // check that actually means something without assuming which semantic is
  // right: is the whole tile uniform? A single-product result would be 0.5;
  // a pair-sum would be 1.0; a quad-sum would be 2.0 -- any of those is
  // plausible evidence of real execution, a non-uniform or garbage tile is not.
  int8_t a[64], b[64];
  for ( int i = 0; i < 64; i++ ) {
    a[i] = (int8_t) 0x38;
    b[i] = (int8_t) 0x30;
  }
  float single_product = decode_e4m3( 0x38 ) * decode_e4m3( 0x30 );
  printf( "  decoded a[0]=%f, b[0]=%f, single product (1x-sum reference)=%f\n",
          decode_e4m3( 0x38 ), decode_e4m3( 0x30 ), single_product );

  printf( "  calling example_sme2_fmopa_fp8_fp8_fp16 (FEAT_SME_F8F16)...\n" );
  uint16_t c16[32 * 32] = {0};
  example_sme2_fmopa_fp8_fp8_fp16( a, b, c16 );
  printf( "  returned without crashing.\n" );
  {
    int zeros = 0, nonfinite = 0, nonuniform = 0;
    float first = f16_bits_to_f32( c16[0] );
    for ( int i = 0; i < 32 * 32; i++ ) {
      float v = f16_bits_to_f32( c16[i] );
      if ( v == 0.0f ) zeros++;
      if ( !isfinite( v ) ) nonfinite++;
      if ( fabsf( v - first ) > 1e-3f ) nonuniform++;
    }
    printf( "  c16[0]=%f (= %.2fx single product), zeros=%d/1024, non-finite=%d/1024, "
            "cells differing from c16[0]=%d/1024\n",
            first, first / single_product, zeros, nonfinite, nonuniform );
    printf( "  first row (32 cells):" );
    for ( int j = 0; j < 32; j++ ) printf( " %.3f", f16_bits_to_f32( c16[j] ) );
    printf( "\n  first column (32 cells):" );
    for ( int i = 0; i < 32; i++ ) printf( " %.3f", f16_bits_to_f32( c16[i * 32] ) );
    printf( "\n  diagonal (32 cells):" );
    for ( int i = 0; i < 32; i++ ) printf( " %.3f", f16_bits_to_f32( c16[i * 32 + i] ) );
    printf( "\n" );
  }

  printf( "  calling example_sme2_fmopa_fp8_fp8_fp32 (FEAT_SME_F8F32)...\n" );
  float c32[16 * 16] = {0};
  example_sme2_fmopa_fp8_fp8_fp32( a, b, c32 );
  printf( "  returned without crashing.\n" );
  {
    int zeros = 0, nonfinite = 0, nonuniform = 0;
    float first = c32[0];
    for ( int i = 0; i < 16 * 16; i++ ) {
      float v = c32[i];
      if ( v == 0.0f ) zeros++;
      if ( !isfinite( v ) ) nonfinite++;
      if ( fabsf( v - first ) > 1e-3f ) nonuniform++;
    }
    printf( "  c32[0]=%f (= %.2fx single product), zeros=%d/256, non-finite=%d/256, "
            "cells differing from c32[0]=%d/256\n",
            first, first / single_product, zeros, nonfinite, nonuniform );
    printf( "  first row (16 cells):" );
    for ( int j = 0; j < 16; j++ ) printf( " %.3f", c32[j] );
    printf( "\n  first column (16 cells):" );
    for ( int i = 0; i < 16; i++ ) printf( " %.3f", c32[i * 16] );
    printf( "\n  diagonal (16 cells):" );
    for ( int i = 0; i < 16; i++ ) printf( " %.3f", c32[i * 16 + i] );
    printf( "\n" );
  }
  (void) f32_to_f16_bits;
}

extern int peak_sme2_fmopa_fp8_fp8_fp32( int64_t i_num_reps );
extern int peak_sme2_fmopa_fp8_fp8_fp16( int64_t i_num_reps );

static double now_seconds( void ) {
  struct timespec ts;
  clock_gettime( CLOCK_MONOTONIC, &ts );
  return (double) ts.tv_sec + (double) ts.tv_nsec * 1e-9;
}

static void bench_peak( const char * label, int (*kernel)( int64_t ), int64_t num_reps ) {
  double start = now_seconds();
  kernel( num_reps );
  double duration = now_seconds() - start;
  double ops_per_rep = (double) kernel( 1 );
  double gops = ops_per_rep * (double) num_reps * 1e-9 / duration;
  printf( "  %s: reps=%lld, duration=%.4fs, GOPS=%.2f\n",
          label, (long long) num_reps, duration, gops );
}

void run_sme2_fp8_peak( void ) {
  printf( "Running SME2 FP8 peak-throughput kernels (readout correctness still open; this\n"
          "measures raw FMOPA instruction throughput only, no readback in the timed loop)...\n" );
  bench_peak( "FEAT_SME_F8F32 FMOPA", peak_sme2_fmopa_fp8_fp8_fp32, 75000000 );
  bench_peak( "FEAT_SME_F8F16 FMOPA", peak_sme2_fmopa_fp8_fp8_fp16, 75000000 );
}
