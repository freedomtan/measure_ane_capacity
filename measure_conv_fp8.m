#import <time.h>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

// Extend MPSGraphDevice to include the private API for ANE (Apple Neural
// Engine) device. This interface allows us to use the private API *if* it is
// available.
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
  // Non-zero 8-bit pattern suitable for FP8 or INT8
  uint8_t *p = (uint8_t *)buffer;
  static const uint8_t fp8_pattern[4] = {0x38, 0x3C, 0x30, 0x40}; // ~1.0 in FP8 formats
  for (size_t i = 0; i < byteCount; i++) {
    p[i] = fp8_pattern[i % 4];
  }
}

/**
 * Runs the native FP8 Conv2D benchmark on the specified device with the given data type.
 */
void run_bench_fp8(id<MTLDevice> device, bool useANE, MPSDataType dataType,
                   NSString *name) {
  @autoreleasepool {
    MPSGraphDevice *mDev = [MPSGraphDevice deviceWithMTLDevice:device];

    MPSGraph *graph = [MPSGraph new];

    // Settings for high-throughput (same dimensions as measure_conv_universal.m)
    NSUInteger B = 1, H = 256, W = 256, Ci = 128, Co = 128, K = 3, L = 20;
    NSArray *inShape = @[ @(B), @(Ci), @(H), @(W) ];
    NSArray *wShape = @[ @(Co), @(Ci), @(K), @(K) ];

    MPSGraphTensor *input = [graph placeholderWithShape:inShape
                                               dataType:dataType
                                                   name:@"in"];
    MPSGraphTensor *cur = input;

    // Determine element size based on data type (FP8 = 1 byte)
    NSUInteger elementSize = 1;

    // Weights allocation
    NSMutableData *wData =
        [NSMutableData dataWithLength:Co * Ci * K * K * elementSize];
    fillNonZeroData(wData.mutableBytes, wData.length, dataType);
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
      @try {
        cur = [graph convolution2DWithSourceTensor:cur
                                     weightsTensor:w
                                        descriptor:d
                                              name:nil];
      } @catch (NSException *e) {
        NSLog(@"[%@] Graph construction failed at layer %d: %@", name, i, e);
        return;
      }
    }

    NSDictionary *feeds = @{
      input : [[MPSGraphShapedType alloc] initWithShape:inShape
                                               dataType:dataType]
    };

    MPSGraphCompilationDescriptor *cd = [MPSGraphCompilationDescriptor new];
    cd.optimizationLevel =
        useANE ? MPSGraphOptimizationLevel1 : MPSGraphOptimizationLevel0;

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

    NSLog(@"[%@] Compiling graph...", name);
    MPSGraphExecutable *exe = nil;
    @try {
      exe = [graph compileWithDevice:mDev
                               feeds:feeds
                       targetTensors:@[ cur ]
                    targetOperations:nil
               compilationDescriptor:cd];
    } @catch (NSException *e) {
      NSLog(@"[%@] Compilation threw exception: %@", name, e);
      return;
    }

    if (!exe) {
      NSLog(@"[%@] Failed to compile graph (unsupported operation/type on this device/OS).", name);
      return;
    }

    NSLog(@"[%@] Compilation succeeded! Running benchmark...", name);

    id<MTLBuffer> iBuf =
        [device newBufferWithLength:B * H * W * Ci * elementSize options:0];
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
    @try {
      [exe runWithMTLCommandQueue:q
                      inputsArray:@[ iData ]
                     resultsArray:nil
              executionDescriptor:ed];
    } @catch (NSException *e) {
      NSLog(@"[%@] Execution failed during warmup: %@", name, e);
      return;
    }

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

    double tops = (2.0 * B * H * W * Ci * Co * K * K * L) / (avg * 1e12);
    NSLog(@"[%@] Avg: %.2f ms, Speed: %.4f TOPS", name, avg * 1000.0, tops);
  }
}

int main(int argc, char *argv[]) {
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  if (!device) {
    NSLog(@"Failed to get default Metal device.");
    return 1;
  }

  NSLog(@"=== Native FP8 (E4M3) Convolutions Benchmark ===");
  run_bench_fp8(device, false, MPSDataTypeFloat8e4m3, @"GPU FP8 (E4M3)");
  run_bench_fp8(device, true, MPSDataTypeFloat8e4m3, @"ANE FP8 (E4M3)");

  NSLog(@"\n=== Native FP8 (E5M2) Convolutions Benchmark ===");
  run_bench_fp8(device, false, MPSDataTypeFloat8e5m2, @"GPU FP8 (E5M2)");
  run_bench_fp8(device, true, MPSDataTypeFloat8e5m2, @"ANE FP8 (E5M2)");

  return 0;
}
