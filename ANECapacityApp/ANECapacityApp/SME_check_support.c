// Derived from https://github.com/scalable-analyses/sme

#include <stdio.h>
#include <unistd.h>

#include "SME_check_support.h"

extern void neon_bf16_support();
extern void sve_support();
extern void sve_streaming_support();
extern void sme_support();
extern void sme2_support();
extern void sme_f16f16_support();
extern void sme_b16b16_support();
extern void sme_f8f16_support();
extern void sme_f8f32_support();
extern void sve_streaming_vlength( float * i_a,
                                   float * i_b );

#include <sys/sysctl.h>

static int query_sysctl_int(const char *name) {
  int val = 0;
  size_t size = sizeof(val);
  if (sysctlbyname(name, &val, &size, NULL, 0) == 0) {
    return val;
  }
  return 0;
}

int check_neon_bf16_support(void){
  printf( "Checking for NEON BF16 support...\n" );
  neon_bf16_support();
  printf( "  NEON BF16 is supported\n" );
  return 1;
}

int check_sve_support(void){
  printf( "Checking for SVE support...\n" );
  sve_support();
  printf( "SVE is supported\n" );
  return 1;
}

int check_streaming_sve_support(void){
  printf( "Checking for SVE streaming support...\n" );
  sve_streaming_support();
  printf( "  Streaming SVE is supported\n"  );
  return 1;
}

int check_sme_support(void){
  printf( "Checking for SME support...\n" );
  sme_support();
  printf( "  SME is supported\n" );  
  return 1;
}

int check_sme2_support(void){
  printf( "Checking for SME2 support...\n" );
  int sup = query_sysctl_int("hw.optional.arm.FEAT_SME2");
  if (sup) {
    sme2_support();
    printf( "  SME2 is supported\n" );
  } else {
    printf( "  SME2 is NOT supported\n" );
  }
  return sup;
}

int check_sme_f16f16_support(void){
  printf( "Checking for SME F16F16 support...\n" );
  int sup = query_sysctl_int("hw.optional.arm.FEAT_SME_F16F16");
  if (sup) {
    sme_f16f16_support();
    printf( "  SME F16F16 is supported\n" );
  } else {
    printf( "  SME F16F16 is NOT supported\n" );
  }
  return sup;
}

int check_sme_b16b16_support(void){
  printf( "Checking for SME B16B16 support...\n" );
  int sup = query_sysctl_int("hw.optional.arm.FEAT_SME_B16B16");
  if (sup) {
    sme_b16b16_support();
    printf( "  SME B16B16 is supported\n" );
  } else {
    printf( "  SME B16B16 is NOT supported\n" );
  }
  return sup;
}

int check_sme_f8f16_support(void){
  printf( "Checking for SME F8F16 support...\n" );
  int sup = query_sysctl_int("hw.optional.arm.FEAT_SME_F8F16");
  if (sup) {
    sme_f8f16_support();
    printf( "  SME F8F16 is supported\n" );
  } else {
    printf( "  SME F8F16 is NOT supported\n" );
  }
  return sup;
}

int check_sme_f8f32_support(void){
  printf( "Checking for SME F8F32 support...\n" );
  int sup = query_sysctl_int("hw.optional.arm.FEAT_SME_F8F32");
  if (sup) {
    sme_f8f32_support();
    printf( "  SME F8F32 is supported\n" );
  } else {
    printf( "  SME F8F32 is NOT supported\n" );
  }
  return sup;
}


int check_sve_streaming_length(){
  printf( "Checking vector length of SVE in streaming mode \n" );
  float l_a[32];
  float l_b[32] = {0};
  for( int l_i = 0; l_i < 32; l_i++ ){
    l_a[l_i] = (float) l_i + 1;
  }
  sve_streaming_vlength( l_a, l_b );

  int l_num_bits = 0;
  for( int l_i = 0; l_i < 32; l_i++ ){
    if( l_b[l_i] > 0 ){
      l_num_bits += 32;
    }
  }
  printf( "  %d bits\n", l_num_bits );

  return l_num_bits;
}
