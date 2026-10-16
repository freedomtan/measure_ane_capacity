// Derived from https://github.com/scalable-analyses/sme

#ifndef CHECK_SUPPORT_H
#define CHECK_SUPPORT_H


#ifdef __cplusplus
extern "C" {
#endif

int check_neon_bf16_support(void);
int check_sve_support(void);
int check_streaming_sve_support(void);
int check_sme_support(void);
int check_sme2_support(void);
int check_sme_f16f16_support(void);
int check_sme_b16b16_support(void);
int check_sme_f8f16_support(void);
int check_sme_f8f32_support(void);
int check_sve_streaming_length(void);

#ifdef __cplusplus
}
#endif

#endif

