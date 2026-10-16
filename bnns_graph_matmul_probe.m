// Finding 11's natural follow-up (docs/CPU_Matrix_Convolution_Capability.md):
// executes a compiled .mlmodelc directly through BNNSGraphCompileFromFile +
// BNNSGraphContextExecute -- the same public, modern graph API CoreML's own
// E5RT backend calls internally (Finding 6 traced the live call chain down
// into exactly this function) -- bypassing MLModel/CoreML's Objective-C
// prediction path entirely, while still going through the normal CoreML
// *compiler* (MLModel.compileModelAtURL:) to produce the .mlmodelc, since
// BNNSGraphCompileFromFile's input is that compiled artifact, not raw MIL
// text.
//
// Unlike Finding 10's MILPrecisionINT8WeightOnly (where the weight is a
// compile-time `const`/`constexpr_blockwise_shift_scale` that the MIL
// compiler folds back to a plain FP16 constant before anything executes),
// MILBuildMatmulWeightInputSpec makes the weight a genuine function INPUT.
// There is no compile-time value to fold, and no load-time constant to
// expand -- the INT8 tensor only exists once BNNSGraphContextExecute() is
// actually called, every single time. If BNNSGraph's modern kernel
// selection has any genuinely-quantized GEMM path (as opposed to the legacy
// AMX one Finding 11 found through BNNSFilterCreateLayerFullyConnected),
// this is the configuration that would reach it.

#import <Accelerate/Accelerate.h>
#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>
#include <mach/mach_time.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "MILSpecBuilder.h"

static double NowSeconds(void) {
  static mach_timebase_info_data_t timebase;
  if (timebase.denom == 0) mach_timebase_info(&timebase);
  uint64_t t = mach_absolute_time();
  return (double)t * timebase.numer / timebase.denom / 1e9;
}

static void LogCallback(BNNSGraphMessageLevel level, const char *msg, const char *opInfo,
                         bnns_user_message_data_t *extra) {
  printf("   [bnns msg level=%u] %s%s%s\n", level, msg, opInfo ? " op_info=" : "", opInfo ?: "");
}

// The weight-input model declares its INT8 weight as an INT32 model input
// (see MILBuildMatmulWeightInputSpec's comment on CoreML's I/O type
// restriction) and `cast`s it to INT8 internally, so the caller fills an
// int32 buffer with small values that round-trip losslessly through that
// cast.
static void FillInt32AsInt8(int32_t *buf, size_t count, uint64_t seed) {
  uint64_t state = seed;
  for (size_t i = 0; i < count; i++) {
    state ^= state << 13; state ^= state >> 7; state ^= state << 17;
    buf[i] = (int32_t)(1 + (state % 15));
  }
}

static void FillFloat16(uint16_t *buf, size_t count, uint64_t seed) {
  uint64_t state = seed;
  for (size_t i = 0; i < count; i++) {
    state ^= state << 13; state ^= state >> 7; state ^= state << 17;
    buf[i] = (state & 1) ? 0x2C00 : 0xAC00;
  }
}

// Compiles `specData` (a serialized CoreML Model protobuf) to a .mlmodelc
// directory on disk and returns its path, or nil on failure. Mirrors
// measure_matmul_coreml.m's compileModel(), minus the timing/MLModelConfiguration
// bits this probe doesn't need (BNNSGraphCompileFromFile reads the .mlmodelc
// directly, with no MLModel ever involved).
static NSString *CompileSpecToPath(NSData *specData, NSError **error) {
  NSString *tempPath = [NSTemporaryDirectory()
      stringByAppendingPathComponent:[NSString stringWithFormat:@"bnns_graph_probe_%d.mlmodel",
                                                                 getpid()]];
  if (![specData writeToFile:tempPath options:0 error:error]) return nil;

  // MLModel.compileModelAtURL: derives its output .mlmodelc name from the
  // input .mlmodel's stem and places it directly in NSTemporaryDirectory()
  // by default -- the same directory/name this probe would otherwise pick
  // for a "destination" copy, which self-destructively deletes the compiler's
  // own output right before copying it. Simplest fix: just use the
  // compiler's own output path directly, no copy.
  dispatch_semaphore_t sem = dispatch_semaphore_create(0);
  __block NSString *compiledPath = nil;
  __block NSError *compileError = nil;
  [MLModel compileModelAtURL:[NSURL fileURLWithPath:tempPath]
           completionHandler:^(NSURL *compiledModelURL, NSError *err) {
             if (compiledModelURL) {
               compiledPath = compiledModelURL.path;
             } else {
               compileError = err;
             }
             dispatch_semaphore_signal(sem);
           }];
  dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);
  [[NSFileManager defaultManager] removeItemAtPath:tempPath error:nil];
  if (!compiledPath && error) *error = compileError;
  return compiledPath;
}

static void RunViaBNNSGraph(const char *label, MILMatmulWeightInputConfig config, int iterations) {
  NSError *error = nil;
  NSData *specData = MILBuildMatmulWeightInputSpec(config, &error);
  if (!specData) {
    printf("[%s] spec build failed: %s\n", label, error.localizedDescription.UTF8String);
    return;
  }

  NSString *mlmodelcPath = CompileSpecToPath(specData, &error);
  if (!mlmodelcPath) {
    printf("[%s] compile failed: %s\n", label, error.localizedDescription.UTF8String);
    return;
  }

  bnns_graph_t graph =
      BNNSGraphCompileFromFile(mlmodelcPath.UTF8String, NULL, (bnns_graph_compile_options_t){0});
  if (!graph.data) {
    printf("[%s] BNNSGraphCompileFromFile failed\n", label);
    [[NSFileManager defaultManager] removeItemAtPath:mlmodelcPath error:nil];
    return;
  }

  extern size_t BNNSGraphGetOperationCount(bnns_graph_t graph, const char *function);
  extern const char *BNNSGraphGetOperationTypeName(bnns_graph_t graph, const char *function, size_t op_id);
  size_t opCount = BNNSGraphGetOperationCount(graph, NULL);
  for (size_t op = 0; op < opCount; op++) {
    const char *opType = BNNSGraphGetOperationTypeName(graph, NULL, op);
    printf("[%s] Compiled Op[%zu]: %s\n", label, op, opType ?: "(null)");
  }

  bnns_graph_context_t context = BNNSGraphContextMake(graph);
  if (!context.data) {
    printf("[%s] BNNSGraphContextMake failed\n", label);
    free(graph.data);
    [[NSFileManager defaultManager] removeItemAtPath:mlmodelcPath error:nil];
    return;
  }
  BNNSGraphContextSetMessageLogCallback(context, LogCallback, NULL);
  BNNSGraphContextSetMessageLogMask(context, BNNSGraphMessageLevelInfo | BNNSGraphMessageLevelUnsupported |
                                                  BNNSGraphMessageLevelWarning | BNNSGraphMessageLevelError);

  size_t M = config.M, K = config.K, N = config.N;
  uint16_t *x = malloc(M * K * sizeof(uint16_t));
  FillFloat16(x, M * K, 0x1234);
  void *weight = NULL;
  if (config.weightIsInt8) {
    int32_t *w32 = malloc(K * N * sizeof(int32_t));
    FillInt32AsInt8(w32, K * N, 0x5678);
    weight = w32;
  } else {
    uint16_t *w16 = malloc(K * N * sizeof(uint16_t));
    FillFloat16(w16, K * N, 0x5678);
    weight = w16;
  }
  uint16_t *y = calloc(M * N, sizeof(uint16_t));

  const char *xName = MILMatmulWeightInputXName().UTF8String;
  const char *weightName = MILMatmulWeightInputWeightName().UTF8String;
  const char *outputName = MILMatmulWeightInputOutputName().UTF8String;

  size_t xPos = BNNSGraphGetArgumentPosition(graph, NULL, xName);
  size_t weightPos = BNNSGraphGetArgumentPosition(graph, NULL, weightName);
  size_t outPos = BNNSGraphGetArgumentPosition(graph, NULL, outputName);
  size_t argCount = BNNSGraphGetArgumentCount(graph, NULL);
  if (xPos == SIZE_MAX || weightPos == SIZE_MAX || outPos == SIZE_MAX) {
    printf("[%s] argument position lookup failed (x=%zu w=%zu y=%zu, count=%zu)\n", label, xPos,
           weightPos, outPos, argCount);
    goto cleanup;
  }

  bnns_graph_argument_t *args = calloc(argCount, sizeof(bnns_graph_argument_t));
  args[xPos].data_ptr = x;
  args[xPos].data_ptr_size = M * K * sizeof(uint16_t);
  args[weightPos].data_ptr = weight;
  args[weightPos].data_ptr_size = K * N * (config.weightIsInt8 ? sizeof(int32_t) : sizeof(uint16_t));
  args[outPos].data_ptr = y;
  args[outPos].data_ptr_size = M * N * sizeof(uint16_t);

  size_t workspaceSize = BNNSGraphContextGetWorkspaceSize(context, NULL);
  void *workspace = NULL;
  if (workspaceSize != SIZE_MAX && workspaceSize > 0) {
    // BNNSGraphContextExecute requires a page-aligned workspace -- plain
    // malloc() isn't guaranteed to be, and silently produces wrong (not
    // even NaN-flagged) output rather than failing loudly.
    posix_memalign(&workspace, 16384, workspaceSize);
  }

  int rc = BNNSGraphContextExecute(context, NULL, argCount, args, workspace ? workspaceSize : 0,
                                   workspace);
  if (rc != 0) {
    printf("[%s] BNNSGraphContextExecute failed, rc=%d\n", label, rc);
    free(args); free(workspace);
    goto cleanup;
  }

  for (int i = 0; i < 3; i++) {
    BNNSGraphContextExecute(context, NULL, argCount, args, workspace ? workspaceSize : 0, workspace);
  }

  double start = NowSeconds();
  for (int i = 0; i < iterations; i++) {
    BNNSGraphContextExecute(context, NULL, argCount, args, workspace ? workspaceSize : 0, workspace);
  }
  double avg = (NowSeconds() - start) / iterations;
  double ops = 2.0 * (double)M * K * N;

  size_t zeros = 0, nonFinite = 0;
  for (size_t i = 0; i < M * N; i++) {
    uint16_t bits = y[i];
    uint32_t exp = (bits >> 10) & 0x1F, mant = bits & 0x3FF;
    if (exp == 31) { nonFinite++; continue; }
    if (exp == 0 && mant == 0) zeros++;
  }
  printf("[%s] Avg: %.3f ms, Speed: %.4f TOPS, y[0] bits=0x%04x, y[mid] bits=0x%04x, zeros=%zu, "
         "non-finite=%zu\n",
         label, avg * 1000.0, (ops / 1e12) / avg, y[0], y[M * N / 2], zeros, nonFinite);

  free(args);
  free(workspace);

cleanup:
  free(x);
  free(weight);
  free(y);
  BNNSGraphContextDestroy(context);
  free(graph.data);
  [[NSFileManager defaultManager] removeItemAtPath:mlmodelcPath error:nil];
}

int main(int argc, const char *argv[]) {
  size_t M = 1024, K = 1024, N = 1024;
  int iterations = 20;
  const char *only = NULL;
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--size") == 0 && i + 1 < argc) {
      M = K = N = (size_t)atoi(argv[++i]);
    } else if (strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) {
      iterations = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--only") == 0 && i + 1 < argc) {
      only = argv[++i];
    }
  }
  printf("BNNSGraphCompileFromFile direct probe: M=K=N=%zu, %d iterations\n", (size_t)M, iterations);

  MILMatmulWeightInputConfig fp16Config = {.M = M, .K = K, .N = N, .weightIsInt8 = NO};
  MILMatmulWeightInputConfig int8Config = {.M = M, .K = K, .N = N, .weightIsInt8 = YES};
  MILMatmulWeightInputConfig reluConfig = {.M = M, .K = K, .N = N, .fuseRelu = YES};

  if (!only || strcmp(only, "fp16") == 0) {
    RunViaBNNSGraph("BNNSGraph FP16 weight-input", fp16Config, iterations);
  }
  if (!only || strcmp(only, "int8") == 0) {
    RunViaBNNSGraph("BNNSGraph INT8 weight-input", int8Config, iterations);
  }
  if (!only || strcmp(only, "relu") == 0) {
    RunViaBNNSGraph("BNNSGraph FP16+ReLU fused", reluConfig, iterations);
  }

  return 0;
}
