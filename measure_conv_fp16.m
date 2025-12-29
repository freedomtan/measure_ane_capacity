#import <time.h>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

// Extend MPSGraphDevice to include the private API for ANE (Apple Neural Engine) device.
// This allows us to target the ANE explicitly for benchmarking.
@interface MPSGraphDevice (ANE)
// ANEDevice is not exported in public headers but is available at runtime.
+ (instancetype)ANEDevice;
@end

/**
 * Runs the Conv2D benchmark on the specified device.
 *
 * @param device The Metal device to use (for buffer allocation).
 * @param useANE Whether to target the Apple Neural Engine (true) or GPU (false).
 */
void run_bench(id<MTLDevice> device, bool useANE) {
  @autoreleasepool {
    // Create a new Metal Performance Shaders Graph instance
    MPSGraph *graph = [MPSGraph new];

    // --- Benchmark Configuration ---
    // Settings for high-throughput convolution
    // B: Batch size
    // H: Height of input
    // W: Width of input
    // Ci: Input channels
    // Co: Output channels
    // K: Kernel size (KxK)
    // L: Number of convolution layers chained together
    NSUInteger B = 1, H = 256, W = 256, Ci = 128, Co = 128, K = 3, L = 20;

    // Define shapes for input tensor and weights
    // Input layout: NCHW (Batch, Channels, Height, Width)
    NSArray *inShape = @[ @(B), @(Ci), @(H), @(W) ];
    // Weights layout: OIHW (Output Channels, Input Channels, Height, Width)
    NSArray *wShape = @[ @(Co), @(Ci), @(K), @(K) ];

    // Create the input placeholder tensor
    MPSGraphTensor *input = [graph placeholderWithShape:inShape
                                               dataType:MPSDataTypeFloat16
                                                   name:@"in"];
    
    // 'cur' tracks the output tensor of the current layer, forming a chain
    MPSGraphTensor *cur = input;

    // Use FP16 (2 bytes per element) for better performance on ANE/GPU
    NSUInteger elementSize = 2;

    // Create constant weights tensor filled with zeros (allocation only, content doesn't matter for perf)
    NSMutableData *wData = [NSMutableData dataWithLength:Co * Ci * K * K * elementSize];
    MPSGraphTensor *w = [graph constantWithData:wData shape:wShape dataType:MPSDataTypeFloat16];

    // --- Graph Construction ---
    // Chain L convolution layers
    for (int i = 0; i < L; i++) {
      // Define the convolution descriptor
      MPSGraphConvolution2DOpDescriptor *d = [MPSGraphConvolution2DOpDescriptor
          descriptorWithStrideInX:1
                        strideInY:1
                  dilationRateInX:1
                  dilationRateInY:1
                           groups:1
                     paddingStyle:MPSGraphPaddingStyleTF_SAME // TensorFlow 'SAME' padding
                       dataLayout:MPSGraphTensorNamedDataLayoutNCHW
                    weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];
      
      // Add convolution operation to the graph
      cur = [graph convolution2DWithSourceTensor:cur weightsTensor:w descriptor:d name:nil];
    }

    // --- Compilation ---
    // Select the target device: ANE or GPU
    MPSGraphDevice *mDev =
        useANE ? [MPSGraphDevice ANEDevice] : [MPSGraphDevice deviceWithMTLDevice:device];
    
    // Define feeding dictionary for compilation (shape and type info)
    NSDictionary *feeds =
        @{input : [[MPSGraphShapedType alloc] initWithShape:inShape dataType:MPSDataTypeFloat16]};

    MPSGraphCompilationDescriptor *cd = [MPSGraphCompilationDescriptor new];
    // Optimization Level: ANE usually benefits from Level 1
    cd.optimizationLevel = useANE ? MPSGraphOptimizationLevel1 : MPSGraphOptimizationLevel0;

    // Compile the graph into an executable
    MPSGraphExecutable *exe = [graph compileWithDevice:mDev
                                                 feeds:feeds
                                         targetTensors:@[ cur ]
                                      targetOperations:nil
                                 compilationDescriptor:cd];
    if (!exe) return; // Exit if compilation failed

    // --- Execution Setup ---
    // Allocate input buffer on the GPU/Shared memory
    id<MTLBuffer> iBuf = [device newBufferWithLength:B * H * W * Ci * elementSize options:0];
    // Wrap buffer in MPSGraphTensorData
    MPSGraphTensorData *iData = [[MPSGraphTensorData alloc] initWithMTLBuffer:iBuf
                                                                        shape:inShape
                                                                     dataType:MPSDataTypeFloat16];

    id<MTLCommandQueue> q = [device newCommandQueue];
    MPSGraphExecutableExecutionDescriptor *ed = [MPSGraphExecutableExecutionDescriptor new];
    // Wait for previous execution to complete before measuring (synchronous execution for timing)
    ed.waitUntilCompleted = YES;

    // --- Warmup ---
    // Run once to prime the caches and stabilize the device state
    [exe runWithMTLCommandQueue:q inputsArray:@[ iData ] resultsArray:nil executionDescriptor:ed];

    // --- Benchmarking ---
    NSUInteger iterations = 20;
    struct timespec start, end;
    
    // Start timer
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < iterations; i++) {
      // Execute the graph
      [exe runWithMTLCommandQueue:q inputsArray:@[ iData ] resultsArray:nil executionDescriptor:ed];
    }
    // Stop timer
    clock_gettime(CLOCK_MONOTONIC, &end);

    // Calculate average duration per iteration in seconds
    double duration = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    double avg = duration / iterations;

    // --- Calculate TOPS ---
    // Generic formula for Conv2D FLOPS (approximate): 2 * B * H * W * Ci * Co * K * K
    // Multiplied by Number of Layers (L)
    // TOPS = Total Operations / (Time in seconds * 10^12)
    double tops = (2.0 * B * H * W * Ci * Co * K * K * L) / (avg * 1e12);
    
    if (!useANE) {
      NSLog(@"[%@] Avg: %.2f ms, Speed: %.4f TOPS", @"GPU FP16", avg * 1000.0, tops);
    } else {
      NSLog(@"[%@] Avg: %.2f ms, Speed: %.4f TOPS", @"ANE FP16", avg * 1000.0, tops);
    }
  }
}

int main(int argc, char *argv[]) {
  // Get system default Metal device (usually the GPU)
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  
  // Run benchmark on GPU
  run_bench(device, false);
  
  // Run benchmark on ANE
  run_bench(device, true);
}
