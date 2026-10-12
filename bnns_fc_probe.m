// Standalone BNNSFilterCreateLayerFullyConnected harness -- the legacy
// public bnns.h API, independent of BNNSMatMul (which rejects any integer
// BNNSDataType outright, see bnns_matmul_probe.m) and independent of
// CoreML/MIL entirely (so no constant folding, see Finding 10).
//
// BNNSLayerParametersFullyConnected gives i_desc/w_desc/o_desc each their
// own independent BNNSNDArrayDescriptor, so the weight can be genuinely
// BNNSDataTypeInt8 (with data_scale) while the input/output stay
// BNNSDataTypeFloat16 -- a real weight-only-quantized matmul (FC is
// Output(o) = sum_i Weight(o,i)*Input(i), i.e. exactly a matmul against a
// fixed weight matrix), unlike the MIL path where the weight gets
// constant-folded back to FP16 before BNNS ever runs.
//
// A FullyConnected layer processes one in_size-vector at a time;
// BNNSFilterApplyBatch applies it to `batch_size` vectors, which is exactly
// our A (MxK) @ W^T (KxN) = C (MxN) GEMM with batch_size=M, in_size=K,
// out_size=N, row-major A as the batch of input vectors.

#import <Accelerate/Accelerate.h>
#include <mach/mach_time.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static double NowSeconds(void) {
  static mach_timebase_info_data_t timebase;
  if (timebase.denom == 0) mach_timebase_info(&timebase);
  uint64_t t = mach_absolute_time();
  return (double)t * timebase.numer / timebase.denom / 1e9;
}

static void FillInt8(int8_t *buf, size_t count, uint64_t seed) {
  uint64_t state = seed;
  for (size_t i = 0; i < count; i++) {
    state ^= state << 13; state ^= state >> 7; state ^= state << 17;
    buf[i] = (int8_t)(1 + (state % 15));
  }
}

static void FillFloat16(uint16_t *buf, size_t count, uint64_t seed) {
  uint64_t state = seed;
  for (size_t i = 0; i < count; i++) {
    state ^= state << 13; state ^= state >> 7; state ^= state << 17;
    buf[i] = (state & 1) ? 0x2C00 : 0xAC00;
  }
}

static BNNSNDArrayDescriptor MakeVecDesc(BNNSDataType type, size_t n, void *data, float scale, float bias) {
  BNNSNDArrayDescriptor d = {0};
  d.layout = BNNSDataLayoutVector;
  d.size[0] = n;
  d.data = data;
  d.data_type = type;
  d.data_scale = scale;
  d.data_bias = bias;
  return d;
}

// Weight(o,i) stored at weights[i + o*in_size] per the header's own layout
// doc -- that's BNNSDataLayoutRowMajorMatrix with size[0]=in_size (i),
// size[1]=out_size (o).
static BNNSNDArrayDescriptor MakeWeightDesc(BNNSDataType type, size_t inSize, size_t outSize,
                                             void *data, float scale, float bias) {
  BNNSNDArrayDescriptor d = {0};
  d.layout = BNNSDataLayoutRowMajorMatrix;
  d.size[0] = inSize;
  d.size[1] = outSize;
  d.data = data;
  d.data_type = type;
  d.data_scale = scale;
  d.data_bias = bias;
  return d;
}

static void RunFC(const char *label, size_t M, size_t K, size_t N, int iterations,
                   BNNSDataType weightType, float weightScale) {
  uint16_t *a = malloc(M * K * sizeof(uint16_t));
  uint16_t *c = calloc(M * N, sizeof(uint16_t));
  FillFloat16(a, M * K, 0x1234);

  void *w = NULL;
  if (weightType == BNNSDataTypeInt8) {
    int8_t *w8 = malloc(K * N);
    FillInt8(w8, K * N, 0x5678);
    w = w8;
  } else {
    uint16_t *w16 = malloc(K * N * sizeof(uint16_t));
    FillFloat16(w16, K * N, 0x5678);
    w = w16;
  }

  BNNSLayerParametersFullyConnected params = {0};
  params.i_desc = MakeVecDesc(BNNSDataTypeFloat16, K, NULL, 0, 0);
  params.o_desc = MakeVecDesc(BNNSDataTypeFloat16, N, NULL, 0, 0);
  params.w_desc = MakeWeightDesc(weightType, K, N, w, weightScale, 0.0f);
  params.bias = (BNNSNDArrayDescriptor){0}; // no bias: data == NULL
  params.activation.function = BNNSActivationFunctionIdentity;

  BNNSFilter filter = BNNSFilterCreateLayerFullyConnected(&params, NULL);
  if (!filter) {
    printf("[%s] BNNSFilterCreateLayerFullyConnected failed (NULL)\n", label);
    free(a); free(c); free(w);
    return;
  }

  int rc = BNNSFilterApplyBatch(filter, M, a, K, c, N);
  if (rc != 0) {
    printf("[%s] BNNSFilterApplyBatch failed, rc=%d\n", label, rc);
    BNNSFilterDestroy(filter);
    free(a); free(c); free(w);
    return;
  }

  for (int i = 0; i < 3; i++) BNNSFilterApplyBatch(filter, M, a, K, c, N);

  double start = NowSeconds();
  for (int i = 0; i < iterations; i++) {
    BNNSFilterApplyBatch(filter, M, a, K, c, N);
  }
  double avg = (NowSeconds() - start) / iterations;
  double ops = 2.0 * (double)M * K * N;
  printf("[%s] Avg: %.3f ms, Speed: %.4f TOPS, c[0] bits=0x%04x, c[mid] bits=0x%04x\n",
         label, avg * 1000.0, (ops / 1e12) / avg, c[0], c[M * N / 2]);

  BNNSFilterDestroy(filter);
  free(a); free(c); free(w);
}

int main(int argc, const char *argv[]) {
  size_t M = 1024, K = 1024, N = 1024;
  int iterations = 20;
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--size") == 0 && i + 1 < argc) {
      M = K = N = (size_t)atoi(argv[++i]);
    } else if (strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) {
      iterations = atoi(argv[++i]);
    }
  }
  printf("BNNSFilterCreateLayerFullyConnected direct probe: M=K=N=%zu, %d iterations\n", M, iterations);
  RunFC("FC FP16 weight", M, K, N, iterations, BNNSDataTypeFloat16, 1.0f);
  RunFC("FC INT8 weight (scale=1/32)", M, K, N, iterations, BNNSDataTypeInt8, 1.0f / 32.0f);
  return 0;
}
