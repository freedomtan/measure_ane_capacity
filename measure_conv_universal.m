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

/**
 * Runs the Conv2D benchmark on the specified device with the given data type.
 */
void run_bench(id<MTLDevice> device, bool useANE, MPSDataType dataType,
               NSString *name) {
  @autoreleasepool {
    // --- Safe ANE Device Retrieval ---
    MPSGraphDevice *mDev = nil;

    if (useANE) {
      if ([MPSGraphDevice respondsToSelector:@selector(ANEDevice)]) {
        mDev = [MPSGraphDevice ANEDevice];
      } else {
        NSLog(@"[%@] Skipped: ANE device not supported on this platform/OS "
              @"version.",
              name);
        return;
      }
    } else {
      mDev = [MPSGraphDevice deviceWithMTLDevice:device];
    }

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

    // Weights allocation (content irrelevant for perf)
    NSMutableData *wData =
        [NSMutableData dataWithLength:Co * Ci * K * K * elementSize];
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
        useANE ? MPSGraphOptimizationLevel1 : MPSGraphOptimizationLevel0;

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

  // Benchmark FP16
  run_bench(device, false, MPSDataTypeFloat16, @"GPU FP16");
  run_bench(device, true, MPSDataTypeFloat16, @"ANE FP16");

  // Benchmark INT8 (Signed Int 8)
  // Check runtime environment or compile macros if specific GPU skipping is
  // needed for iOS vs macOS. Generally, ANE supports INT8. GPU support depends
  // on the specific GPU family. For safety, we keep the previous logic: GPU
  // INT8 is likely unsupported on Apple GPUs for MPSGraph convolution.

  // run_bench(device, false, MPSDataTypeInt8, @"GPU INT8");
  run_bench(device, true, MPSDataTypeInt8, @"ANE INT8");
}
