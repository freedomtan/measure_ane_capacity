#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#import <time.h>

// Extend MPSGraphDevice to include the private API for ANE (Apple Neural
// Engine) device.
@interface MPSGraphDevice (ANE)
+ (instancetype)ANEDevice;
@end

/**
 * Runs the Conv2D benchmark with QDQ (Quantization/Dequantization) on the
 * specified device. Input is INT8, dequantized to FP16 for FP16 convolution,
 * then quantized back to INT8.
 */
void run_bench_qdq(id<MTLDevice> device, bool useANE) {
  @autoreleasepool {
    MPSGraph *graph = [MPSGraph new];

    // --- Benchmark Configuration ---
    NSUInteger B = 1, H = 256, W = 256, Ci = 128, Co = 128, K = 3, L = 20;
    NSArray *inShape = @[ @(B), @(Ci), @(H), @(W) ];
    NSArray *wShape = @[ @(Co), @(Ci), @(K), @(K) ];

    // Input tensor: INT8
    MPSGraphTensor *input = [graph placeholderWithShape:inShape
                                               dataType:MPSDataTypeInt8
                                                   name:@"in"];

    // Scale and Zero Point for quantization (per-channel or scalar)
    // We'll use scalars for simplicity, but broadcasted to Ci/Co axis
    MPSGraphTensor *scale = [graph constantWithScalar:1.0
                                             dataType:MPSDataTypeFloat32];
    MPSGraphTensor *zp = [graph constantWithScalar:0.0
                                          dataType:MPSDataTypeInt32];

    MPSGraphTensor *cur = input;

    // Weights tensor: FP16
    NSMutableData *wData = [NSMutableData dataWithLength:Co * Ci * K * K * 2];
    MPSGraphTensor *w = [graph constantWithData:wData
                                          shape:wShape
                                       dataType:MPSDataTypeFloat16];

    // --- Graph Construction with QDQ ---
    // Pattern: Dequantize -> [Chain of Conv] -> Quantize
    // Or: Chain of [Dequantize -> Conv -> Quantize]
    // Usually, "QDQ" in benchmarking refers to simulating quantized precision
    // with FP16 math.

    for (int i = 0; i < L; i++) {
      // Dequantize to FP16 before convolution
      cur = [graph dequantizeTensor:cur
                        scaleTensor:scale
                    zeroPointTensor:zp
                           dataType:MPSDataTypeFloat16
                               axis:1
                               name:nil];

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

      // Quantize back to INT8 after convolution
      cur = [graph quantizeTensor:cur
                      scaleTensor:scale
                  zeroPointTensor:zp
                         dataType:MPSDataTypeInt8
                             axis:1
                             name:nil];
    }

    // --- Compilation ---
    MPSGraphDevice *mDev = useANE ? [MPSGraphDevice ANEDevice]
                                  : [MPSGraphDevice deviceWithMTLDevice:device];

    NSDictionary *feeds = @{
      input : [[MPSGraphShapedType alloc] initWithShape:inShape
                                               dataType:MPSDataTypeInt8]
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
      NSLog(@"Failed to compile graph.");
      return;
    }

    // --- Execution ---
    id<MTLBuffer> iBuf = [device newBufferWithLength:B * H * W * Ci options:0];
    MPSGraphTensorData *iData =
        [[MPSGraphTensorData alloc] initWithMTLBuffer:iBuf
                                                shape:inShape
                                             dataType:MPSDataTypeInt8];

    id<MTLCommandQueue> q = [device newCommandQueue];
    MPSGraphExecutableExecutionDescriptor *ed =
        [MPSGraphExecutableExecutionDescriptor new];
    ed.waitUntilCompleted = YES;

    // Warmup
    [exe runWithMTLCommandQueue:q
                    inputsArray:@[ iData ]
                   resultsArray:nil
            executionDescriptor:ed];

    // Benchmark
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

    NSLog(@"[%@] QDQ(Int8->FP16->Int8) Avg: %.2f ms, Speed: %.4f TOPS",
          useANE ? @"ANE" : @"GPU", avg * 1000.0, tops);
  }
}

int main(int argc, char *argv[]) {
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();

  // Run on GPU
  run_bench_qdq(device, false);

  // Run on ANE
  run_bench_qdq(device, true);

  return 0;
}
