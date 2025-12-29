#import <time.h>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

@interface MPSGraphDevice (ANE)
+ (instancetype)ANEDevice;
@end

/**
 * Runs the Conv2D benchmark on the specified device with the given data type.
 */
void run_bench(id<MTLDevice> device, bool useANE, MPSDataType dataType,
               NSString *name) {
  @autoreleasepool {
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
    }

    MPSGraphDevice *mDev = useANE ? [MPSGraphDevice ANEDevice]
                                  : [MPSGraphDevice deviceWithMTLDevice:device];
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
    if (!exe)
      return;

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

  // Benchmark FP16
  run_bench(device, false, MPSDataTypeFloat16, @"GPU FP16");
  run_bench(device, true, MPSDataTypeFloat16, @"ANE FP16");

  // Benchmark INT8 (Signed Int 8)
  // GPU INT8 is not supported by MPSGraphConvolution on this
  // device/configuration run_bench(device, false, MPSDataTypeInt8, @"GPU
  // INT8");
  run_bench(device, true, MPSDataTypeInt8, @"ANE INT8");
}
