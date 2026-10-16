// Apple Silicon SME / SME2 Microbenchmark Suite
// Derived from and based on https://github.com/scalable-analyses/sme

#include <stdio.h>
#include <stdlib.h>

#include <vector>
#include <thread>
#include <chrono>
#include <stdint.h>
#include <string.h>
#include "ANECapacityApp/ANECapacityApp/SME_check_support.h"

extern "C" {
  int peak_neon_fmla_fp64_fp64_fp64( int64_t i_num_reps );
  int peak_neon_fmla_fp32_fp32_fp32( int64_t i_num_reps );
  int peak_neon_fmla_fp16_fp16_fp16( int64_t i_num_reps );
  int peak_neon_bfmmla_bf16_bf16_fp32( int64_t i_num_reps );
  int peak_sve_fmla_streaming_fp32_fp32_fp32( int64_t i_num_reps );
  int peak_sve_fmla_streaming_fp64_fp64_fp64( int64_t i_num_reps );
  int peak_sme_fmopa_1_fp32_fp32_fp32( int64_t i_num_reps );
  int peak_sme_fmopa_2_fp32_fp32_fp32( int64_t i_num_reps );
  int peak_sme_fmopa_4_fp32_fp32_fp32( int64_t i_num_reps );
  int peak_sme_fmopa_fp64_fp64_fp64( int64_t i_num_reps );
  int peak_sme_fmopa_fp16_fp16_fp32( int64_t i_num_reps );
  int peak_sme_fmopa_fp16_fp16_fp16( int64_t i_num_reps );
  int peak_sme_bfmopa_bf16_bf16_fp32( int64_t i_num_reps );
  int peak_sme_bfmopa_bf16_bf16_bf16( int64_t i_num_reps );
  int peak_sme_smopa_i8_i8_i32( int64_t i_num_reps );
  int peak_sme_smopa_i16_i16_i32( int64_t i_num_reps );
  int peak_sme2_fmopa_fp8_fp8_fp32( int64_t i_num_reps );
  int peak_sme2_fmopa_fp8_fp8_fp16( int64_t i_num_reps );
}

static void bench_kernel(const char* name, int (*kernel)(int64_t), int64_t reps, int num_threads) {
    printf("  %-38s | ", name);
    fflush(stdout);

    int ops_per_rep = kernel(1);
    if (ops_per_rep <= 0) {
        printf("SKIPPED (invalid ops_per_rep)\n");
        return;
    }

    auto start = std::chrono::steady_clock::now();
    if (num_threads <= 1) {
        kernel(reps);
    } else {
        std::vector<std::thread> workers;
        workers.reserve(num_threads);
        for (int t = 0; t < num_threads; t++) {
            workers.emplace_back([=]() {
                kernel(reps);
            });
        }
        for (auto& w : workers) {
            w.join();
        }
    }
    auto end = std::chrono::steady_clock::now();
    double duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();

    double total_ops = (double)ops_per_rep * (double)reps * (double)num_threads;
    double gops = (total_ops * 1e-9) / duration;
    double tops = gops / 1000.0;

    printf("%8.4f s | %10.2f GOPS (%7.3f TOPS)\n", duration, gops, tops);
    fflush(stdout);
}

int main(int argc, char* argv[]) {
    int num_threads = 1;
    if (argc > 1) {
        num_threads = atoi(argv[1]);
    }

    printf("=================================================================\n");
    printf("  Apple Silicon M6 Hardware Peak Throughput Benchmark (SME/NEON) \n");
    printf("  Threads: %d | SVL: %d bits\n", num_threads, check_sve_streaming_length());
    printf("=================================================================\n");

    printf("\n--- Neon Baseline Benchmarks ---\n");
    bench_kernel("Neon FP64 FMLA", peak_neon_fmla_fp64_fp64_fp64, 100000000, num_threads);
    bench_kernel("Neon FP32 FMLA", peak_neon_fmla_fp32_fp32_fp32, 100000000, num_threads);
    bench_kernel("Neon FP16 FMLA", peak_neon_fmla_fp16_fp16_fp16, 100000000, num_threads);
    bench_kernel("Neon BF16 BFMMLA", peak_neon_bfmmla_bf16_bf16_fp32, 50000000, num_threads);

    printf("\n--- Streaming SVE (SSVE) Benchmarks ---\n");
    bench_kernel("SSVE FP64 FMLA", peak_sve_fmla_streaming_fp64_fp64_fp64, 25000000, num_threads);
    bench_kernel("SSVE FP32 FMLA", peak_sve_fmla_streaming_fp32_fp32_fp32, 25000000, num_threads);

    printf("\n--- SME1 Outer-Product (FMOPA / SMOPA) Benchmarks ---\n");
    bench_kernel("SME FP32 FMOPA (1 tile)", peak_sme_fmopa_1_fp32_fp32_fp32, 25000000, num_threads);
    bench_kernel("SME FP32 FMOPA (2 tiles)", peak_sme_fmopa_2_fp32_fp32_fp32, 25000000, num_threads);
    bench_kernel("SME FP32 FMOPA (4 tiles)", peak_sme_fmopa_4_fp32_fp32_fp32, 50000000, num_threads);
    bench_kernel("SME FP64 FMOPA (8 tiles)", peak_sme_fmopa_fp64_fp64_fp64, 25000000, num_threads);
    bench_kernel("SME FP16->FP32 FMOPA", peak_sme_fmopa_fp16_fp16_fp32, 50000000, num_threads);
    bench_kernel("SME BF16->FP32 BFMOPA", peak_sme_bfmopa_bf16_bf16_fp32, 50000000, num_threads);
    bench_kernel("SME INT8->INT32 SMOPA", peak_sme_smopa_i8_i8_i32, 50000000, num_threads);
    bench_kernel("SME INT16->INT32 SMOPA", peak_sme_smopa_i16_i16_i32, 50000000, num_threads);

    printf("\n--- SME2 / ARMv9.4-A Architectural Extensions ---\n");
    if (check_sme_f16f16_support()) {
        bench_kernel("SME2 FP16->FP16 FMOPA (F16F16)", peak_sme_fmopa_fp16_fp16_fp16, 50000000, num_threads);
    }
    if (check_sme_b16b16_support()) {
        bench_kernel("SME2 BF16->BF16 BFMOPA (B16B16)", peak_sme_bfmopa_bf16_bf16_bf16, 50000000, num_threads);
    }
    if (check_sme_f8f32_support()) {
        bench_kernel("SME2 FP8->FP32 FMOPA (F8F32)", peak_sme2_fmopa_fp8_fp8_fp32, 75000000, num_threads);
    }
    if (check_sme_f8f16_support()) {
        bench_kernel("SME2 FP8->FP16 FMOPA (F8F16)", peak_sme2_fmopa_fp8_fp8_fp16, 75000000, num_threads);
    }

    printf("\nBenchmark complete.\n");
    return 0;
}
