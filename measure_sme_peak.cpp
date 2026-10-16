// Apple Silicon SME / SME2 Peak Microbenchmark
// Derived from and based on https://github.com/scalable-analyses/sme

#include <iostream>
#include <cstdlib>

#include "ANECapacityApp/ANECapacityApp/SME_benchmark.h"
#include "ANECapacityApp/ANECapacityApp/SME_fp8_sketch.h"
#include "ANECapacityApp/ANECapacityApp/SME_check_support.h"

extern "C" int peak_neon_fmla_fp64_fp64_fp64(int64_t);

int main(int argc, char *argv[]) {
    setbuf(stdout, NULL);
    int threads = 1;
    int qos = 1; // User Interactive
    if (argc > 1) {
        threads = atoi(argv[1]);
    }
    if (argc > 2) {
        qos = atoi(argv[2]);
    }
    std::cout << "=== Apple Silicon SME / SME2 Peak Micro-Benchmark ===" << std::endl;
    std::cout << "Threads: " << threads << ", QoS: " << qos << std::endl;
    std::cout << "Hardware Feature Probes:" << std::endl;
    std::cout << "  FEAT_SME:        " << check_sme_support() << std::endl;
    std::cout << "  FEAT_SME2:       " << check_sme2_support() << std::endl;
    std::cout << "  FEAT_SME_F16F16: " << check_sme_f16f16_support() << std::endl;
    std::cout << "  FEAT_SME_B16B16: " << check_sme_b16b16_support() << std::endl;
    std::cout << "  FEAT_SME_F8F16:  " << check_sme_f8f16_support() << std::endl;
    std::cout << "  FEAT_SME_F8F32:  " << check_sme_f8f32_support() << std::endl;
    std::cout << "  SVL (bytes):     " << check_sve_streaming_length() << std::endl;
    std::cout << std::endl;

    std::cout << "--- 1. SME2 FP8 Kernel Peak ---" << std::endl;
    if (check_sme2_support() && check_sme_f8f32_support()) {
        std::cout << "-> Starting run_sme2_fp8_peak()..." << std::endl;
        run_sme2_fp8_peak();
    } else {
        std::cout << "Skipping SME2 FP8 tests (not supported)" << std::endl;
    }
    std::cout << std::endl;

    std::cout << "Testing peak_neon_fmla_fp64_fp64_fp64 directly..." << std::endl;
    int ops = peak_neon_fmla_fp64_fp64_fp64(10);
    std::cout << "peak_neon_fmla_fp64_fp64_fp64 returned: " << ops << std::endl;

    std::cout << "--- 2. Full SME / Neon Microbenchmarks ---" << std::endl;
    run_micro_benchmark(threads, qos);

    return 0;
}
