// Tests whether coremltools' native, single-op `scaled_dot_product_attention`
// (iOS18 opset) reaches BNNS's KERNEL_MATMUL_SOFTMAX_MATMUL_SME tier instead
// of the AMX path every bare `matmul` in this repo reaches (see
// docs/AMX_GEMM_Microkernel_Disassembly.md's "Clues toward actually reaching
// the SME2 path" -- the fused matmul+relu experiment there didn't trigger it
// because CoreML never actually fused those two ops; this op is a genuine
// single fused primitive confirmed via MLComputePlan, not an inferred one).
//
// Executes via BNNSGraphCompileFromFile + BNNSGraphContext*, same mechanism
// as bnns_graph_matmul_probe.m, with query/key/value as genuine function
// inputs (not consts) so there's nothing to constant-fold.

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

static void FillFloat16(uint16_t *buf, size_t count, uint64_t seed) {
  uint64_t state = seed;
  for (size_t i = 0; i < count; i++) {
    state ^= state << 13; state ^= state >> 7; state ^= state << 17;
    buf[i] = (state & 1) ? 0x2C00 : 0xAC00;
  }
}

static void LogCallback(BNNSGraphMessageLevel level, const char *msg, const char *opInfo,
                         bnns_user_message_data_t *extra) {
  printf("   [bnns msg level=%u] %s%s%s\n", level, msg, opInfo ? " op_info=" : "", opInfo ?: "");
}

static NSString *CompileSpecToPath(NSData *specData, NSError **error) {
  NSString *tempPath = [NSTemporaryDirectory()
      stringByAppendingPathComponent:[NSString stringWithFormat:@"bnns_graph_attn_%d.mlmodel",
                                                                 getpid()]];
  if (![specData writeToFile:tempPath options:0 error:error]) return nil;

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

int main(int argc, const char *argv[]) {
  size_t B = 1, L = 1024, S = 1024, E = 1024;
  int iterations = 20;
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--size") == 0 && i + 1 < argc) {
      L = S = E = (size_t)atoi(argv[++i]);
    } else if (strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) {
      iterations = atoi(argv[++i]);
    }
  }
  printf("BNNSGraph scaled_dot_product_attention probe: B=%zu L=S=E=%zu, %d iterations\n", B, L,
         iterations);

  MILAttentionConfig config = {.B = B, .L = L, .S = S, .E = E};
  NSError *error = nil;
  NSData *specData = MILBuildAttentionSpec(config, &error);
  if (!specData) {
    printf("spec build failed: %s\n", error.localizedDescription.UTF8String);
    return 1;
  }

  NSString *mlmodelcPath = CompileSpecToPath(specData, &error);
  if (!mlmodelcPath) {
    printf("compile failed: %s\n", error.localizedDescription.UTF8String);
    return 1;
  }

  bnns_graph_t graph =
      BNNSGraphCompileFromFile(mlmodelcPath.UTF8String, NULL, (bnns_graph_compile_options_t){0});
  if (!graph.data) {
    printf("BNNSGraphCompileFromFile failed\n");
    return 1;
  }

  bnns_graph_context_t context = BNNSGraphContextMake(graph);
  if (!context.data) {
    printf("BNNSGraphContextMake failed\n");
    free(graph.data);
    return 1;
  }
  BNNSGraphContextSetMessageLogCallback(context, LogCallback, NULL);
  BNNSGraphContextSetMessageLogMask(context, BNNSGraphMessageLevelInfo | BNNSGraphMessageLevelUnsupported |
                                                  BNNSGraphMessageLevelWarning | BNNSGraphMessageLevelError);

  uint16_t *q = malloc(B * L * E * sizeof(uint16_t));
  uint16_t *k = malloc(B * S * E * sizeof(uint16_t));
  uint16_t *v = malloc(B * S * E * sizeof(uint16_t));
  uint16_t *out = calloc(B * L * E, sizeof(uint16_t));
  FillFloat16(q, B * L * E, 0x1234);
  FillFloat16(k, B * S * E, 0x5678);
  FillFloat16(v, B * S * E, 0x9abc);

  const char *qName = MILAttentionQueryName().UTF8String;
  const char *kName = MILAttentionKeyName().UTF8String;
  const char *vName = MILAttentionValueName().UTF8String;
  const char *outName = MILAttentionOutputName().UTF8String;

  size_t qPos = BNNSGraphGetArgumentPosition(graph, NULL, qName);
  size_t kPos = BNNSGraphGetArgumentPosition(graph, NULL, kName);
  size_t vPos = BNNSGraphGetArgumentPosition(graph, NULL, vName);
  size_t outPos = BNNSGraphGetArgumentPosition(graph, NULL, outName);
  size_t argCount = BNNSGraphGetArgumentCount(graph, NULL);
  if (qPos == SIZE_MAX || kPos == SIZE_MAX || vPos == SIZE_MAX || outPos == SIZE_MAX) {
    printf("argument position lookup failed (q=%zu k=%zu v=%zu out=%zu, count=%zu)\n", qPos, kPos,
           vPos, outPos, argCount);
    return 1;
  }

  bnns_graph_argument_t *args = calloc(argCount, sizeof(bnns_graph_argument_t));
  args[qPos].data_ptr = q;
  args[qPos].data_ptr_size = B * L * E * sizeof(uint16_t);
  args[kPos].data_ptr = k;
  args[kPos].data_ptr_size = B * S * E * sizeof(uint16_t);
  args[vPos].data_ptr = v;
  args[vPos].data_ptr_size = B * S * E * sizeof(uint16_t);
  args[outPos].data_ptr = out;
  args[outPos].data_ptr_size = B * L * E * sizeof(uint16_t);

  size_t workspaceSize = BNNSGraphContextGetWorkspaceSize(context, NULL);
  void *workspace = NULL;
  if (workspaceSize != SIZE_MAX && workspaceSize > 0) {
    posix_memalign(&workspace, 16384, workspaceSize);
  }

  int rc = BNNSGraphContextExecute(context, NULL, argCount, args, workspace ? workspaceSize : 0, workspace);
  if (rc != 0) {
    printf("BNNSGraphContextExecute failed, rc=%d\n", rc);
    return 1;
  }

  for (int i = 0; i < 3; i++) {
    BNNSGraphContextExecute(context, NULL, argCount, args, workspace ? workspaceSize : 0, workspace);
  }

  double start = NowSeconds();
  for (int i = 0; i < iterations; i++) {
    BNNSGraphContextExecute(context, NULL, argCount, args, workspace ? workspaceSize : 0, workspace);
  }
  double avg = (NowSeconds() - start) / iterations;
  double ops = 2.0 * (double)B * L * S * E * 2.0; // QK^T + attn@V, each 2*B*L*S*E
  size_t zeros = 0, nonFinite = 0;
  for (size_t i = 0; i < B * L * E; i++) {
    uint16_t bits = out[i];
    uint32_t exp = (bits >> 10) & 0x1F, mant = bits & 0x3FF;
    if (exp == 31) { nonFinite++; continue; }
    if (exp == 0 && mant == 0) zeros++;
  }
  printf("[attention] Avg: %.3f ms, Speed: %.4f TOPS, out[0] bits=0x%04x, zeros=%zu, non-finite=%zu\n",
         avg * 1000.0, (ops / 1e12) / avg, out[0], zeros, nonFinite);

  free(q); free(k); free(v); free(out); free(args); free(workspace);
  BNNSGraphContextDestroy(context);
  free(graph.data);
  return 0;
}
