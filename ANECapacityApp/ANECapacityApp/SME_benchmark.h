// Derived from https://github.com/scalable-analyses/sme

#ifndef BENCHMARK_H
#define BENCHMARK_H


#ifdef __cplusplus
extern "C" {
#endif
  void run_micro_benchmark( int i_num_threads,
                            int i_qos_class );
  void run_cblas_benchmark(void);
  void run_bandwidth_benchmark( int i_kernel_type,
                                int i_align_bytes,
                                int i_qos_class );
#ifdef __cplusplus
}
#endif

#endif