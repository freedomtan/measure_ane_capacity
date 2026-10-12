#import <time.h>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

// Extend MPSGraphDevice to include the private API for ANE (Apple Neural
// Engine) device. This interface allows us to use the private API *if* it is
// available.
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
 * Runs the Conv2D benchmark on the specified device with the given data type.
 */
void run_bench(id<MTLDevice> device, Target target, MPSDataType dataType,
               NSString *name) {
  @autoreleasepool {
    // Default device wrapper
    MPSGraphDevice *mDev = [MPSGraphDevice deviceWithMTLDevice:device];

    MPSGraph *graph = [MPSGraph new];

    // Settings for high-throughput
    NSUInteger B = 1, H = 256, W = 256, Ci = 128, Co = 128, K = 3, L = 20;
    NSArray *inShape = @[ @(B), @(Ci), @(H), @(W) ];
    NSArray *wShape = @[ @(Co), @(Ci), @(K), @(K) ];

    MPSGraphTensor *input = [graph placeholderWithShape:inShape
                                               dataType:dataType
                                                   name:@"in"];
    MPSGraphTensor *cur = input;

    // Determine element size based on data type
    NSUInteger elementSize = (dataType == MPSDataTypeFloat16) ? 2 : 1;

    // Weights allocation (non-zero initialized to prevent hardware zero-skipping on H17+)
    NSMutableData *wData =
        [NSMutableData dataWithLength:Co * Ci * K * K * elementSize];
    fillNonZeroDataWithSeed(wData.mutableBytes, wData.length, dataType, 0x5EED5EED5EED5EEDULL);
    MPSGraphTensor *w = [graph constantWithData:wData
                                          shape:wShape
                                       dataType:dataType];

    for (int i = 0; i < L; i++) {
      MPSGraphConvolution2DOpDescriptor *d = [MPSGraphConvolution2DOpDescriptor
          descriptorWithStrideInX:1
                        strideInY:1
                  dilationRateInX:1
                  dilationRateInY:1
                           groups:1
                     paddingStyle:MPSGraphPaddingStyleTF_SAME
                       dataLayout:MPSGraphTensorNamedDataLayoutNCHW
                    weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];
      cur = [graph convolution2DWithSourceTensor:cur
                                   weightsTensor:w
                                      descriptor:d
                                            name:nil];
      // Realistic Quantized flow: Int8 -> (Conv) -> Int32/FP16 ->
      // (Select/Scale) -> Int8
      if (dataType == MPSDataTypeInt8) {
        MPSGraphTensor *fp = [graph castTensor:cur
                                        toType:MPSDataTypeFloat16
                                          name:@"dequant"];
        cur = [graph castTensor:fp toType:MPSDataTypeInt8 name:@"requant"];
      }
    }

    NSDictionary *feeds = @{
      input : [[MPSGraphShapedType alloc] initWithShape:inShape
                                               dataType:dataType]
    };

    MPSGraphCompilationDescriptor *cd = [MPSGraphCompilationDescriptor new];
    cd.optimizationLevel =
        (target == kTargetGPU) ? MPSGraphOptimizationLevel0 : MPSGraphOptimizationLevel1;

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

    id<MTLBuffer> iBuf =
        [device newBufferWithLength:B * H * W * Ci * elementSize options:0];
    fillNonZeroDataWithSeed(iBuf.contents, iBuf.length, dataType, 0x9E3779B97F4A7C15ULL);
    MPSGraphTensorData *iData =
        [[MPSGraphTensorData alloc] initWithMTLBuffer:iBuf
                                                shape:inShape
                                             dataType:dataType];

    id<MTLBuffer> oBuf =
        [device newBufferWithLength:B * H * W * Co * elementSize options:0];
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

    double tops = (2.0 * B * H * W * Ci * Co * K * K * L) / (avg * 1e12);

    size_t totalElem = B * H * W * Co;
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
  // -[MPSGraphExecutable compileWithDevice:...] itself (see
  // measure_matmul_universal.m for the full stack trace), which @try/@catch
  // on the calling thread cannot intercept -- it aborts the whole process.
  // Run each target as its own subprocess (via --target) so a CPU-target
  // crash doesn't take out the GPU/ANE results in the same invocation.
  NSString *onlyTarget = nil;
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--target") == 0 && i + 1 < argc) {
      onlyTarget = [NSString stringWithUTF8String:argv[i + 1]];
    }
  }

  if (!onlyTarget || [onlyTarget isEqualToString:@"gpu"]) {
    run_bench(device, kTargetGPU, MPSDataTypeFloat16, @"GPU FP16");
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
