#import <time.h>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

// Extend MPSGraphDevice to include the private API for ANE (Apple Neural
// Engine) device on older OS versions.
@interface MPSGraphDevice (ANE)
+ (instancetype)ANEDevice;
+ (instancetype)CPUDevice;
@end

typedef enum { kTargetGPU, kTargetANE, kTargetCPU } Target;

// Private property on MPSGraphCompilationDescriptor to target ANE on modern macOS/iOS.
@interface MPSGraphCompilationDescriptor (Private)
@property (nonatomic, assign) unsigned long long preferredDevice;
@end

// Fill buffer with small non-zero values to prevent hardware zero-skipping on H17/H18.
// Uses deterministic xorshift64 random non-canceling signs to avoid reduction cancellation.
static void fillNonZeroDataWithSeed(void *buffer, size_t byteCount, MPSDataType dataType, uint64_t seed) {
  if (!buffer || byteCount == 0) return;
  uint64_t state = seed;
  if (dataType == MPSDataTypeFloat16) {
    uint16_t *p = (uint16_t *)buffer;
    size_t count = byteCount / sizeof(uint16_t);
    for (size_t i = 0; i < count; i++) {
      state ^= state << 13;
      state ^= state >> 7;
      state ^= state << 17;
      p[i] = (state & 1) ? 0xA800 : 0x2800; // -0.03125, +0.03125
    }
  } else {
    int8_t *p = (int8_t *)buffer;
    for (size_t i = 0; i < byteCount; i++) {
      state ^= state << 13;
      state ^= state >> 7;
      state ^= state << 17;
      p[i] = (state & 1) ? -1 : 1;
    }
  }
}

static void fillNonZeroData(void *buffer, size_t byteCount, MPSDataType dataType) {
  fillNonZeroDataWithSeed(buffer, byteCount, dataType, 0x5EED5EED5EED5EEDULL);
}

/**
 * Runs the Matrix Multiplication (GEMM) benchmark on the specified device with the given data type
 * using the native MPSGraph `matrixMultiplicationWithPrimaryTensor:secondaryTensor:` API.
 *
 * NOTE ON INT8 IN MPSGRAPH:
 * Unlike MPSGraph convolution (which allows INT8 operands directly), MPSGraph's underlying
 * `mps.matmul` MLIR operator strictly requires floating-point operands:
 *   "error: 'mps.matmul' op operand #0 must be tensor of floating point values or tensor of complex values"
 * Therefore, in MPSGraph, INT8 matrix multiplication requires the standard Quantize-Dequantize
 * flow: INT8 inputs are dequantized to FP16 before matmul, and requantized to INT8 afterwards.
 */
static NSUInteger g_size = 1024;
static NSUInteger g_layers = 20;
static NSUInteger g_optLevel = 0;

void run_bench(id<MTLDevice> device, Target target, MPSDataType dataType,
               NSString *name) {
  @autoreleasepool {
    // Default device wrapper
    MPSGraphDevice *mDev = [MPSGraphDevice deviceWithMTLDevice:device];

    MPSGraph *graph = [MPSGraph new];

    // High-throughput GEMM dimensions: 4D tensors [B, 1, M, K] x [B, 1, K, N]
    NSUInteger B = 1, M = g_size, K = g_size, N = g_size, L = g_layers;
    NSArray *inShape = @[ @(B), @1, @(M), @(K) ];
    NSArray *wShape = @[ @(B), @1, @(K), @(N) ];

    MPSGraphTensor *input = [graph placeholderWithShape:inShape
                                               dataType:dataType
                                                   name:@"in"];
    MPSGraphTensor *cur = input;

    // Weights allocation (non-zero initialized to prevent hardware zero-skipping on H17+)
    // MPSGraph matrixMultiplication requires floating point operands.
    NSMutableData *wData =
        [NSMutableData dataWithLength:B * K * N * sizeof(uint16_t)];
    fillNonZeroDataWithSeed(wData.mutableBytes, wData.length, MPSDataTypeFloat16, 0x5EED5EED5EED5EEDULL);
    MPSGraphTensor *w = [graph constantWithData:wData
                                          shape:wShape
                                       dataType:MPSDataTypeFloat16];

    for (int i = 0; i < L; i++) {
      MPSGraphTensor *lhs = cur;
      // MPSGraph's mps.matmul requires floating-point operands. For INT8 mode,
      // cast INT8 to FP16 before matrixMultiplication and requantize back to INT8.
      if (dataType == MPSDataTypeInt8) {
        lhs = [graph castTensor:cur toType:MPSDataTypeFloat16 name:@"dequant"];
      }

      NSString *matmulName = [NSString stringWithFormat:@"matmul_%d", i];
      MPSGraphTensor *matmulOut = [graph matrixMultiplicationWithPrimaryTensor:lhs
                                                              secondaryTensor:w
                                                                         name:matmulName];

      if (dataType == MPSDataTypeInt8) {
        cur = [graph castTensor:matmulOut toType:MPSDataTypeInt8 name:@"requant"];
      } else {
        cur = matmulOut;
      }
    }

    NSDictionary *feeds = @{
      input : [[MPSGraphShapedType alloc] initWithShape:inShape
                                               dataType:dataType]
    };

    MPSGraphCompilationDescriptor *cd = [MPSGraphCompilationDescriptor new];
    cd.optimizationLevel = (target == kTargetANE) ? MPSGraphOptimizationLevel1 : (MPSGraphOptimization)g_optLevel;

    // Safe ANE device targeting across OS versions:
    // On macOS 15+ / 26+, preferredDevice = 2 targets ANE while retaining MTLDevice context.
    // On older platforms, fallback to [MPSGraphDevice ANEDevice].
    if (target == kTargetANE) {
      if ([cd respondsToSelector:@selector(setPreferredDevice:)]) {
        cd.preferredDevice = 2; // MPSGraphDeviceTypeANE
      } else if ([MPSGraphDevice respondsToSelector:@selector(ANEDevice)]) {
        mDev = [MPSGraphDevice ANEDevice];
      } else {
        NSLog(@"[%@] Skipped: ANE device not supported on this platform/OS version.", name);
        return;
      }
    } else if (target == kTargetCPU) {
      // Private SPI, parallel to +[MPSGraphDevice ANEDevice]; no corresponding
      // preferredDevice enum value is known, so route through the device
      // object directly rather than guessing one.
      if ([MPSGraphDevice respondsToSelector:@selector(CPUDevice)]) {
        mDev = [MPSGraphDevice CPUDevice];
      } else {
        NSLog(@"[%@] Skipped: CPU device not supported on this platform/OS version.", name);
        return;
      }
    }

    MPSGraphExecutable *exe = [graph compileWithDevice:mDev
                                                 feeds:feeds
                                         targetTensors:@[ cur ]
                                      targetOperations:nil
                                 compilationDescriptor:cd];
    if (!exe) {
      NSLog(@"[%@] Failed to compile graph.", name);
      return;
    }

    NSUInteger inElementSize = (dataType == MPSDataTypeFloat16) ? sizeof(uint16_t) : sizeof(int8_t);
    id<MTLBuffer> iBuf =
        [device newBufferWithLength:B * M * K * inElementSize options:0];
    fillNonZeroDataWithSeed(iBuf.contents, iBuf.length, dataType, 0x9E3779B97F4A7C15ULL);
    MPSGraphTensorData *iData =
        [[MPSGraphTensorData alloc] initWithMTLBuffer:iBuf
                                                shape:inShape
                                             dataType:dataType];

    id<MTLBuffer> oBuf =
        [device newBufferWithLength:B * M * N * inElementSize options:0];
    MPSGraphTensorData *oData =
        [[MPSGraphTensorData alloc] initWithMTLBuffer:oBuf
                                                shape:inShape
                                             dataType:dataType];

    id<MTLCommandQueue> q = [device newCommandQueue];
    MPSGraphExecutableExecutionDescriptor *ed =
        [MPSGraphExecutableExecutionDescriptor new];
    ed.waitUntilCompleted = YES;

    // Warmup
    [exe runWithMTLCommandQueue:q
                    inputsArray:@[ iData ]
                   resultsArray:@[ oData ]
            executionDescriptor:ed];

    NSUInteger iterations = 20;
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < iterations; i++) {
      [exe runWithMTLCommandQueue:q
                      inputsArray:@[ iData ]
                     resultsArray:@[ oData ]
              executionDescriptor:ed];
    }
    clock_gettime(CLOCK_MONOTONIC, &end);

    double duration =
        (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    double avg = duration / iterations;

    // 2.0 * B * M * K * N FLOPs per layer * L layers
    double tops = (2.0 * B * M * K * N * L) / (avg * 1e12);

    size_t totalElem = B * M * N;
    size_t zeroCount = 0;
    if (dataType == MPSDataTypeFloat16) {
      uint16_t *outPtr = (uint16_t *)oBuf.contents;
      for (size_t k = 0; k < totalElem; k++) {
        if (outPtr[k] == 0x0000 || outPtr[k] == 0x8000) zeroCount++;
      }
      NSLog(@"[%@] Avg: %.2f ms, Speed: %.4f TOPS | Check: [0x%04x, 0x%04x, 0x%04x, 0x%04x] (zeros: %lu/%lu)",
            name, avg * 1000.0, tops, outPtr[0], outPtr[1], outPtr[2], outPtr[3],
            (unsigned long)zeroCount, (unsigned long)totalElem);
    } else {
      int8_t *outPtr = (int8_t *)oBuf.contents;
      for (size_t k = 0; k < totalElem; k++) {
        if (outPtr[k] == 0) zeroCount++;
      }
      NSLog(@"[%@] Avg: %.2f ms, Speed: %.4f TOPS | Check: [%d, %d, %d, %d] (zeros: %lu/%lu)",
            name, avg * 1000.0, tops, outPtr[0], outPtr[1], outPtr[2], outPtr[3],
            (unsigned long)zeroCount, (unsigned long)totalElem);
    }
  }
}

int main(int argc, char *argv[]) {
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  if (!device) {
    NSLog(@"Failed to get default Metal device.");
    return 1;
  }

  // The CPU target's compile path has been observed to throw an
  // NSInvalidArgumentException from a background GCD thread inside
  // -[MPSGraphExecutable compileWithDevice:...] itself, which @try/@catch on
  // the calling thread cannot intercept -- it aborts the whole process. Run
  // each target as its own subprocess (via --target) so a CPU-target crash
  // doesn't take out the GPU/ANE results in the same invocation.
  NSString *onlyTarget = nil;
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--target") == 0 && i + 1 < argc) {
      onlyTarget = [NSString stringWithUTF8String:argv[++i]];
    } else if (strcmp(argv[i], "--size") == 0 && i + 1 < argc) {
      g_size = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--layers") == 0 && i + 1 < argc) {
      g_layers = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--opt") == 0 && i + 1 < argc) {
      g_optLevel = atoi(argv[++i]);
    }
  }

  NSLog(@"Benchmarking GEMM: Size = %lu x %lu, Layers = %lu, OptLevel = %lu", 
        (unsigned long)g_size, (unsigned long)g_size, (unsigned long)g_layers, (unsigned long)g_optLevel);

  if (!onlyTarget || [onlyTarget isEqualToString:@"gpu"]) {
    run_bench(device, kTargetGPU, MPSDataTypeFloat16, @"GPU FP16");
    run_bench(device, kTargetGPU, MPSDataTypeInt8, @"GPU INT8 (QDQ)");
  }
  if (!onlyTarget || [onlyTarget isEqualToString:@"ane"]) {
    run_bench(device, kTargetANE, MPSDataTypeFloat16, @"ANE FP16");
    run_bench(device, kTargetANE, MPSDataTypeInt8, @"ANE INT8");
  }
  if (!onlyTarget || [onlyTarget isEqualToString:@"cpu"]) {
    run_bench(device, kTargetCPU, MPSDataTypeFloat16, @"CPU FP16");
    run_bench(device, kTargetCPU, MPSDataTypeInt8, @"CPU INT8");
  }

  return 0;
}
