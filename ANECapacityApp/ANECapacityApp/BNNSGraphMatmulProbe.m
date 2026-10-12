//
//  BNNSGraphMatmulProbe.m
//  On-device port of the root repo's bnns_graph_matmul_probe.m (see
//  docs/CPU_Matrix_Convolution_Capability.md Finding 12).
//

#import "BNNSGraphMatmulProbe.h"
#import "MILSpecBuilder.h"
#import <Accelerate/Accelerate.h>
#import <CoreML/CoreML.h>
#include <mach/mach_time.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>

static double NowSeconds(void) {
    static mach_timebase_info_data_t timebase;
    if (timebase.denom == 0) mach_timebase_info(&timebase);
    uint64_t t = mach_absolute_time();
    return (double)t * timebase.numer / timebase.denom / 1e9;
}

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

// MLModel.compileModelAtURL:'s default output path and a self-computed
// "destination" path both land in NSTemporaryDirectory() with the same
// name-derivation rule, so copying there self-deletes the compiler's own
// output before the copy runs (see Finding 12). Just use the compiler's own
// output URL directly.
static NSString *CompileSpecToPath(NSData *specData, NSString *label, NSError **error) {
    NSString *tempPath = [NSTemporaryDirectory()
        stringByAppendingPathComponent:[NSString stringWithFormat:@"bnns_graph_probe_%@_%d.mlmodel",
                                                                   label, getpid()]];
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

static void LogCallback(BNNSGraphMessageLevel level, const char *msg, const char *opInfo,
                         bnns_user_message_data_t *extra) {
    printf("   [bnns msg] level=%u msg=%s op_info=%s\n", level, msg, opInfo ?: "(null)");
}

static void RunViaBNNSGraph(const char *label, MILMatmulWeightInputConfig config, int iterations) {
    NSError *error = nil;
    NSData *specData = MILBuildMatmulWeightInputSpec(config, &error);
    if (!specData) {
        printf("[%s] spec build failed: %s\n", label, error.localizedDescription.UTF8String);
        return;
    }

    NSString *mlmodelcPath = CompileSpecToPath(specData, [NSString stringWithUTF8String:label], &error);
    if (!mlmodelcPath) {
        printf("[%s] compile failed: %s\n", label, error.localizedDescription.UTF8String);
        return;
    }

    bnns_graph_t graph =
        BNNSGraphCompileFromFile(mlmodelcPath.UTF8String, NULL, (bnns_graph_compile_options_t){0});
    if (!graph.data) {
        printf("[%s] BNNSGraphCompileFromFile failed\n", label);
        return;
    }

    bnns_graph_context_t context = BNNSGraphContextMake(graph);
    if (!context.data) {
        printf("[%s] BNNSGraphContextMake failed\n", label);
        free(graph.data);
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

    {
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
            // BNNSGraphContextExecute requires a page-aligned workspace --
            // malloc() isn't guaranteed to be, and silently produces wrong
            // (not even NaN-flagged) output rather than failing loudly.
            posix_memalign(&workspace, 16384, workspaceSize);
        }

        int rc = BNNSGraphContextExecute(context, NULL, argCount, args, workspace ? workspaceSize : 0, workspace);
        if (rc != 0) {
            printf("[%s] BNNSGraphContextExecute failed, rc=%d\n", label, rc);
            free(args);
            free(workspace);
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
    }

cleanup:
    free(x);
    free(weight);
    free(y);
    BNNSGraphContextDestroy(context);
    free(graph.data);
}

void BNNSGraphMatmulProbeRun(void) {
    size_t M = 1024, K = 1024, N = 1024;
    int iterations = 2000;
    printf("BNNSGraphCompileFromFile on-device probe: M=K=N=%zu, %d iterations\n", M, iterations);

    MILMatmulWeightInputConfig fp16Config = {.M = M, .K = K, .N = N, .weightIsInt8 = NO};
    RunViaBNNSGraph("BNNSGraph FP16 weight-input", fp16Config, iterations);

    MILMatmulWeightInputConfig int8Config = {.M = M, .K = K, .N = N, .weightIsInt8 = YES};
    RunViaBNNSGraph("BNNSGraph INT8 weight-input", int8Config, iterations);
}
