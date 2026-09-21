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

    // Create constant weights tensor filled with non-zero values (required on H17+ to prevent hardware zero-skipping)
    NSMutableData *wData = [NSMutableData dataWithLength:Co * Ci * K * K * elementSize];
    fillNonZeroDataWithSeed(wData.mutableBytes, wData.length, MPSDataTypeFloat16, 0x5EED5EED5EED5EEDULL);
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
    MPSGraphDevice *mDev = [MPSGraphDevice deviceWithMTLDevice:device];
    
    // Define feeding dictionary for compilation (shape and type info)
    NSDictionary *feeds =
        @{input : [[MPSGraphShapedType alloc] initWithShape:inShape dataType:MPSDataTypeFloat16]};

    MPSGraphCompilationDescriptor *cd = [MPSGraphCompilationDescriptor new];
    // Optimization Level: ANE usually benefits from Level 1
    cd.optimizationLevel = useANE ? MPSGraphOptimizationLevel1 : MPSGraphOptimizationLevel0;

    if (useANE) {
      if ([cd respondsToSelector:@selector(setPreferredDevice:)]) {
        cd.preferredDevice = 2; // MPSGraphDeviceTypeANE
      } else if ([MPSGraphDevice respondsToSelector:@selector(ANEDevice)]) {
        mDev = [MPSGraphDevice ANEDevice];
      }
    }

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
    fillNonZeroDataWithSeed(iBuf.contents, iBuf.length, MPSDataTypeFloat16, 0x9E3779B97F4A7C15ULL);
    // Wrap buffer in MPSGraphTensorData
    MPSGraphTensorData *iData = [[MPSGraphTensorData alloc] initWithMTLBuffer:iBuf
                                                                        shape:inShape
                                                                     dataType:MPSDataTypeFloat16];

    id<MTLBuffer> oBuf = [device newBufferWithLength:B * H * W * Co * elementSize options:0];
    MPSGraphTensorData *oData = [[MPSGraphTensorData alloc] initWithMTLBuffer:oBuf
                                                                        shape:inShape
                                                                     dataType:MPSDataTypeFloat16];

    id<MTLCommandQueue> q = [device newCommandQueue];
    MPSGraphExecutableExecutionDescriptor *ed = [MPSGraphExecutableExecutionDescriptor new];
    // Wait for previous execution to complete before measuring (synchronous execution for timing)
    ed.waitUntilCompleted = YES;

    // --- Warmup ---
    // Run once to prime the caches and stabilize the device state
    [exe runWithMTLCommandQueue:q inputsArray:@[ iData ] resultsArray:@[ oData ] executionDescriptor:ed];

    // --- Benchmarking ---
    NSUInteger iterations = 20;
    struct timespec start, end;
    
    // Start timer
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < iterations; i++) {
      // Execute the graph
      [exe runWithMTLCommandQueue:q inputsArray:@[ iData ] resultsArray:@[ oData ] executionDescriptor:ed];
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
    
    NSString *tag = useANE ? @"ANE FP16" : @"GPU FP16";
    size_t totalElem = B * H * W * Co;
    size_t zeroCount = 0;
    uint16_t *outPtr = (uint16_t *)oBuf.contents;
    for (size_t k = 0; k < totalElem; k++) {
      if (outPtr[k] == 0x0000 || outPtr[k] == 0x8000) zeroCount++;
    }
    NSLog(@"[%@] Avg: %.2f ms, Speed: %.4f TOPS | Check: [0x%04x, 0x%04x, 0x%04x, 0x%04x] (zeros: %lu/%lu)",
          tag, avg * 1000.0, tops, outPtr[0], outPtr[1], outPtr[2], outPtr[3],
          (unsigned long)zeroCount, (unsigned long)totalElem);
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
