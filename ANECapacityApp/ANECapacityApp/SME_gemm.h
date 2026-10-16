// Derived from https://github.com/scalable-analyses/sme

#ifndef GEMM_H
#define GEMM_H


#ifdef __cplusplus
extern "C" {
#endif
  void run_gemm( int i_num_threads,
                 int i_qos_type );
#ifdef __cplusplus
}
#endif

#endif
