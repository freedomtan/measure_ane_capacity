#import <time.h>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

// Extend MPSGraphDevice to include the private API for ANE (Apple Neural
// Engine) device on older OS versions.
@interface MPSGraphDevice (ANE)
+ (instancetype)ANEDevice;
@end

// Private property on MPSGraphCompilationDescriptor to target ANE on modern macOS/iOS.
@interface MPSGraphCompilationDescriptor (Private)
@property (nonatomic, assign) unsigned long long preferredDevice;
@end

// Fill buffer with small non-zero values to prevent hardware zero-skipping on H17/H18.
static void fillNonZeroData(void *buffer, size_t byteCount, MPSDataType dataType) {
  if (!buffer || byteCount == 0) return;
  if (dataType == MPSDataTypeFloat16) {
    uint16_t *p = (uint16_t *)buffer;
    size_t count = byteCount / sizeof(uint16_t);
    static const uint16_t fp16_pattern[4] = {0x2C00, 0xAC00, 0x2800, 0xA800};
    for (size_t i = 0; i < count; i++) {
      p[i] = fp16_pattern[i % 4];
    }
  } else {
    int8_t *p = (int8_t *)buffer;
    static const int8_t int8_pattern[4] = {1, -1, 2, -2};
    for (size_t i = 0; i < byteCount; i++) {
      p[i] = int8_pattern[i % 4];
    }
  }
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
void run_bench(id<MTLDevice> device, bool useANE, MPSDataType dataType,
               NSString *name) {
  @autoreleasepool {
    // Default device wrapper
    MPSGraphDevice *mDev = [MPSGraphDevice deviceWithMTLDevice:device];

    MPSGraph *graph = [MPSGraph new];

    // High-throughput GEMM dimensions: B x M x K multiplied by B x K x N
    // Output shape matches input shape (B x M x N where M == N) allowing seamless chaining of L layers.
    NSUInteger B = 1, M = 1024, K = 1024, N = 1024, L = 20;
    NSArray *inShape = @[ @(B), @(M), @(K) ];
    NSArray *wShape = @[ @(B), @(K), @(N) ];

    MPSGraphTensor *input = [graph placeholderWithShape:inShape
                                               dataType:dataType
                                                   name:@"in"];
    MPSGraphTensor *cur = input;

    // Weights allocation (non-zero initialized to prevent hardware zero-skipping on H17+)
    // MPSGraph matrixMultiplication requires floating point operands.
    NSMutableData *wData =
        [NSMutableData dataWithLength:B * K * N * sizeof(uint16_t)];
    fillNonZeroData(wData.mutableBytes, wData.length, MPSDataTypeFloat16);
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

      cur = [graph matrixMultiplicationWithPrimaryTensor:lhs
                                         secondaryTensor:w
                                                    name:nil];

      if (dataType == MPSDataTypeInt8) {
        cur = [graph castTensor:cur toType:MPSDataTypeInt8 name:@"requant"];
      }
    }

    NSDictionary *feeds = @{
      input : [[MPSGraphShapedType alloc] initWithShape:inShape
                                               dataType:dataType]
    };

    MPSGraphCompilationDescriptor *cd = [MPSGraphCompilationDescriptor new];
    cd.optimizationLevel =
        useANE ? MPSGraphOptimizationLevel1 : MPSGraphOptimizationLevel0;

    // Safe ANE device targeting across OS versions:
    // On macOS 15+ / 26+, preferredDevice = 2 targets ANE while retaining MTLDevice context.
    // On older platforms, fallback to [MPSGraphDevice ANEDevice].
    if (useANE) {
      if ([cd respondsToSelector:@selector(setPreferredDevice:)]) {
        cd.preferredDevice = 2; // MPSGraphDeviceTypeANE
      } else if ([MPSGraphDevice respondsToSelector:@selector(ANEDevice)]) {
        mDev = [MPSGraphDevice ANEDevice];
      } else {
        NSLog(@"[%@] Skipped: ANE device not supported on this platform/OS version.", name);
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
    fillNonZeroData(iBuf.contents, iBuf.length, dataType);
    MPSGraphTensorData *iData =
        [[MPSGraphTensorData alloc] initWithMTLBuffer:iBuf
                                                shape:inShape
                                             dataType:dataType];

    id<MTLCommandQueue> q = [device newCommandQueue];
    MPSGraphExecutableExecutionDescriptor *ed =
        [MPSGraphExecutableExecutionDescriptor new];
    ed.waitUntilCompleted = YES;

    // Warmup
    [exe runWithMTLCommandQueue:q
                    inputsArray:@[ iData ]
                   resultsArray:nil
            executionDescriptor:ed];

    NSUInteger iterations = 20;
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < iterations; i++) {
      [exe runWithMTLCommandQueue:q
                      inputsArray:@[ iData ]
                     resultsArray:nil
              executionDescriptor:ed];
    }
    clock_gettime(CLOCK_MONOTONIC, &end);

    double duration =
        (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    double avg = duration / iterations;

    // 2.0 * B * M * K * N FLOPs per layer * L layers
    double tops = (2.0 * B * M * K * N * L) / (avg * 1e12);
    NSLog(@"[%@] Avg: %.2f ms, Speed: %.4f TOPS", name, avg * 1000.0, tops);
  }
}

int main(int argc, char *argv[]) {
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  if (!device) {
    NSLog(@"Failed to get default Metal device.");
    return 1;
  }

  // Benchmark FP16
  run_bench(device, false, MPSDataTypeFloat16, @"GPU FP16");
  run_bench(device, true, MPSDataTypeFloat16, @"ANE FP16");

  // Benchmark INT8 (Signed Int 8)
  // run_bench(device, false, MPSDataTypeInt8, @"GPU INT8");
  run_bench(device, true, MPSDataTypeInt8, @"ANE INT8");

  return 0;
}
