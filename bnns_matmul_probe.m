// Standalone BNNSMatMul harness -- calls libBNNS.dylib's public, documented
// C API directly (Accelerate.framework/vecLib's bnns.h), with no CoreML, no
// MIL, and no protobuf involved at all. See
// docs/CPU_Matrix_Convolution_Capability.md Finding 10: the MIL-builder path
// (measure_matmul_coreml) can't reach a genuinely-INT8-typed BNNSMatMul call
// because the MIL compiler constant-folds a compile-time-constant
// dequantize() back to a plain FP16 weight before BNNS ever executes
// anything. This harness builds the BNNSNDArrayDescriptor tensors by hand,
// so there is no compiler in the way to fold anything -- whatever kernel
// BNNS picks for a real `BNNSDataTypeInt8` x `BNNSDataTypeInt8` call is the
// kernel it actually picks.
//
// `BNNSMatMul` itself is `__API_DEPRECATED("Use BNNSGraph* APIs", ...)` as of
// macOS 15/iOS 18, in favor of the BNNSGraph path CoreML uses internally
// (Finding 6). It remains present and callable; this probe exists
// specifically to look at what the *old*, directly-driven dense GEMM path
// does with genuine low-precision input, independent of whatever
// BNNSGraph's own kernel-selection logic (traced in Finding 6) does.

#import <Accelerate/Accelerate.h>
#import <Foundation/Foundation.h>
#include <mach/mach_time.h>
#include <stdint.h>
#include <string.h>

static double NowSeconds(void) {
  static mach_timebase_info_data_t timebase;
  if (timebase.denom == 0) mach_timebase_info(&timebase);
  uint64_t t = mach_absolute_time();
  return (double)t * timebase.numer / timebase.denom / 1e9;
}

// Deterministic, non-zero, non-cancelling fill -- same xorshift pattern used
// throughout this repo's other benchmarks (e.g. NativeCoreMLMatmulProbe).
static void FillInt8(int8_t *buf, size_t count, uint64_t seed) {
  uint64_t state = seed;
  for (size_t i = 0; i < count; i++) {
    state ^= state << 13; state ^= state >> 7; state ^= state << 17;
    buf[i] = (int8_t)(1 + (state % 15)); // 1..15, always positive and non-zero
  }
}

static void FillFloat16(uint16_t *buf, size_t count, uint64_t seed) {
  uint64_t state = seed;
  for (size_t i = 0; i < count; i++) {
    state ^= state << 13; state ^= state >> 7; state ^= state << 17;
    buf[i] = (state & 1) ? 0x2C00 : 0xAC00; // +-0.0625, matching the repo's convention
  }
}

static BNNSNDArrayDescriptor MakeDesc(BNNSDataType type, size_t rows, size_t cols,
                                       void *data, float scale, float bias) {
  BNNSNDArrayDescriptor d = {0};
  d.layout = BNNSDataLayoutRowMajorMatrix;
  d.size[0] = cols; // RowMajorMatrix: size[0] = columns
  d.size[1] = rows; // size[1] = rows
  d.data = data;
  d.data_type = type;
  d.data_scale = scale; // only consulted for Int<K>/UInt<K> data_type
  d.data_bias = bias;
  return d;
}

static void RunFloat16(size_t M, size_t K, size_t N, int iterations) {
  size_t aCount = M * K, bCount = K * N, cCount = M * N;
  uint16_t *a = malloc(aCount * sizeof(uint16_t));
  uint16_t *b = malloc(bCount * sizeof(uint16_t));
  uint16_t *c = calloc(cCount, sizeof(uint16_t));
  FillFloat16(a, aCount, 0x1234);
  FillFloat16(b, bCount, 0x5678);

  BNNSNDArrayDescriptor da = MakeDesc(BNNSDataTypeFloat16, M, K, a, 0, 0);
  BNNSNDArrayDescriptor db = MakeDesc(BNNSDataTypeFloat16, K, N, b, 0, 0);
  BNNSNDArrayDescriptor dc = MakeDesc(BNNSDataTypeFloat16, M, N, c, 0, 0);

  ssize_t wsSize = BNNSMatMulWorkspaceSize(false, false, 1.0f, &da, &db, &dc, NULL);
  void *workspace = wsSize > 0 ? malloc((size_t)wsSize) : NULL;

  // Warmup.
  for (int i = 0; i < 3; i++) {
    int rc = BNNSMatMul(false, false, 1.0f, &da, &db, &dc, workspace, NULL);
    if (rc != 0) { printf("[BNNSMatMul FP16] warmup failed, rc=%d\n", rc); return; }
  }

  double start = NowSeconds();
  for (int i = 0; i < iterations; i++) {
    BNNSMatMul(false, false, 1.0f, &da, &db, &dc, workspace, NULL);
  }
  double avg = (NowSeconds() - start) / iterations;
  double ops = 2.0 * (double)M * K * N;
  printf("[BNNSMatMul FP16] Avg: %.3f ms, Speed: %.4f TOPS\n", avg * 1000.0, (ops / 1e12) / avg);

  uint16_t sample0 = c[0], sample1 = c[cCount / 2];
  printf("   Output check: c[0] bits=0x%04x, c[mid] bits=0x%04x\n", sample0, sample1);

  free(a); free(b); free(c); free(workspace);
}

static void RunInt8(size_t M, size_t K, size_t N, int iterations, BNNSDataType outputType,
                     const char *outputName) {
  size_t aCount = M * K, bCount = K * N, cCount = M * N;
  int8_t *a = malloc(aCount);
  int8_t *b = malloc(bCount);
  FillInt8(a, aCount, 0x1234);
  FillInt8(b, bCount, 0x5678);

  // Real quantized inputs: int8 storage, with data_scale/data_bias giving the
  // affine mapping back to the "real" value BNNSMatMul uses internally
  // (real = data_scale * stored + data_bias), same convention as this repo's
  // other QDQ experiments (Finding 7/10) but now with the quantized tensor
  // actually reaching BNNS as BNNSDataTypeInt8, not folded away beforehand.
  const float scale = 1.0f / 32.0f;
  BNNSNDArrayDescriptor da = MakeDesc(BNNSDataTypeInt8, M, K, a, scale, 0.0f);
  BNNSNDArrayDescriptor db = MakeDesc(BNNSDataTypeInt8, K, N, b, scale, 0.0f);

  void *c = NULL;
  BNNSNDArrayDescriptor dc;
  if (outputType == BNNSDataTypeInt32) {
    c = calloc(cCount, sizeof(int32_t));
    dc = MakeDesc(BNNSDataTypeInt32, M, N, c, 1.0f, 0.0f);
  } else if (outputType == BNNSDataTypeFloat32) {
    c = calloc(cCount, sizeof(float));
    dc = MakeDesc(BNNSDataTypeFloat32, M, N, c, 0, 0);
  } else {
    c = calloc(cCount, sizeof(uint16_t));
    dc = MakeDesc(BNNSDataTypeFloat16, M, N, c, 0, 0);
  }

  ssize_t wsSize = BNNSMatMulWorkspaceSize(false, false, 1.0f, &da, &db, &dc, NULL);
  printf("[BNNSMatMul INT8->%s] workspace size: %zd\n", outputName, wsSize);
  void *workspace = wsSize > 0 ? malloc((size_t)wsSize) : NULL;

  int rc = BNNSMatMul(false, false, 1.0f, &da, &db, &dc, workspace, NULL);
  if (rc != 0) {
    printf("[BNNSMatMul INT8->%s] failed, rc=%d\n", outputName, rc);
    free(a); free(b); free(c); free(workspace);
    return;
  }

  for (int i = 0; i < 2; i++) {
    BNNSMatMul(false, false, 1.0f, &da, &db, &dc, workspace, NULL);
  }

  double start = NowSeconds();
  for (int i = 0; i < iterations; i++) {
    BNNSMatMul(false, false, 1.0f, &da, &db, &dc, workspace, NULL);
  }
  double avg = (NowSeconds() - start) / iterations;
  double ops = 2.0 * (double)M * K * N;
  printf("[BNNSMatMul INT8->%s] Avg: %.3f ms, Speed: %.4f TOPS\n", outputName, avg * 1000.0,
         (ops / 1e12) / avg);

  if (outputType == BNNSDataTypeInt32) {
    int32_t *ci = (int32_t *)c;
    printf("   Output check: c[0]=%d, c[mid]=%d\n", ci[0], ci[cCount / 2]);
  } else if (outputType == BNNSDataTypeFloat32) {
    float *cf = (float *)c;
    printf("   Output check: c[0]=%g, c[mid]=%g\n", cf[0], cf[cCount / 2]);
  } else {
    uint16_t *ch = (uint16_t *)c;
    printf("   Output check: c[0] bits=0x%04x, c[mid] bits=0x%04x\n", ch[0], ch[cCount / 2]);
  }

  free(a); free(b); free(c); free(workspace);
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
  printf("BNNSMatMul direct probe: M=K=N=%zu, %d iterations\n", M, iterations);
  RunFloat16(M, K, N, iterations);
  RunInt8(M, K, N, iterations, BNNSDataTypeInt32, "Int32");
  RunInt8(M, K, N, iterations, BNNSDataTypeFloat32, "Float32");
  RunInt8(M, K, N, iterations, BNNSDataTypeFloat16, "Float16");
  return 0;
}
