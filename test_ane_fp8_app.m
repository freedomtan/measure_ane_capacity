#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

@interface MPSGraphCompilationDescriptor (Private)
@property (nonatomic, assign) unsigned long long preferredDevice;
@end

static void fillMultiTierFP8(void *buffer, size_t byteCount, uint64_t seed) {
  uint64_t state = seed;
  const uint8_t posTiers[4] = {0x28, 0x30, 0x38, 0x40}; // 0.25, 0.5, 0.75, 1.0
  const uint8_t negTiers[4] = {0xA8, 0xB0, 0xB8, 0xC0}; // -0.25, -0.5, -0.75, -1.0
  uint8_t *ptr = (uint8_t *)buffer;
  for (size_t i = 0; i < byteCount; i++) {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    int tier = (int)((state >> 1) & 3);
    ptr[i] = (state & 1) ? negTiers[tier] : posTiers[tier];
  }
}

int main(int argc, char *argv[]) {
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> q = [device newCommandQueue];

    NSUInteger B = 1, H = 256, W = 256, C = 128, L = 20;

    for (int i = 1; i < argc; i++) {
      NSString *arg = [NSString stringWithUTF8String:argv[i]];
      if ([arg isEqualToString:@"--channels"] && i + 1 < argc) C = atoi(argv[++i]);
      else if ([arg isEqualToString:@"--layers"] && i + 1 < argc) L = atoi(argv[++i]);
      else if ([arg isEqualToString:@"--size"] && i + 1 < argc) { H = atoi(argv[++i]); W = H; }
      else if ([arg isEqualToString:@"--batch"] && i + 1 < argc) B = atoi(argv[++i]);
    }

    NSUInteger Ci = C, Co = C, K = 3;
    double fp8Scale = 1.0;

    printf("========================================================================\n");
    printf(" MPSGraph FP8 QDQ Benchmark on ANE: [B=%lu, %lux%lu, C=%lu, L=%lu]\n",
           (unsigned long)B, (unsigned long)H, (unsigned long)W, (unsigned long)C, (unsigned long)L);
    printf("========================================================================\n");

    NSArray *inShape = @[@(B), @(Ci), @(H), @(W)];
    NSArray *wShape = @[@(Co), @(Ci), @(K), @(K)];
    MPSDataType fp8Type = (MPSDataType)0x10430008; // MPSDataTypeFloat8e4m3

    MPSGraph *graph = [MPSGraph new];
    MPSGraphTensor *input = [graph placeholderWithShape:inShape dataType:fp8Type name:@"in"];
    MPSGraphTensor *cur = input;

    size_t wBytes = Co * Ci * K * K;
    NSMutableData *wData = [NSMutableData dataWithLength:wBytes];
    fillMultiTierFP8(wData.mutableBytes, wBytes, 0x5EED5EED5EED5EEDULL);
    MPSGraphTensor *wFP8 = [graph constantWithData:wData shape:wShape dataType:fp8Type];
    MPSGraphTensor *w = [graph dequantizeTensor:wFP8 scale:fp8Scale zeroPoint:0.0 dataType:MPSDataTypeFloat16 name:@"w_dequant"];

    MPSGraphConvolution2DOpDescriptor *d = [MPSGraphConvolution2DOpDescriptor
        descriptorWithStrideInX:1 strideInY:1
        dilationRateInX:1 dilationRateInY:1
        groups:1 paddingStyle:MPSGraphPaddingStyleTF_SAME
        dataLayout:MPSGraphTensorNamedDataLayoutNCHW
        weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];

    for (NSUInteger l = 0; l < L; l++) {
      MPSGraphTensor *actFP16 = [graph dequantizeTensor:cur scale:fp8Scale zeroPoint:0.0 dataType:MPSDataTypeFloat16 name:nil];
      MPSGraphTensor *outFP16 = [graph convolution2DWithSourceTensor:actFP16 weightsTensor:w descriptor:d name:nil];
      cur = [graph quantizeTensor:outFP16 scale:fp8Scale zeroPoint:0.0 dataType:fp8Type name:nil];
    }

    MPSGraphCompilationDescriptor *cd = [MPSGraphCompilationDescriptor new];
    cd.preferredDevice = 2; // ANE
    cd.optimizationLevel = MPSGraphOptimizationLevel1;

    printf("⚡️ Compiling MPSGraph for Apple Neural Engine...\n");
    uint64_t tComp0 = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
    MPSGraphExecutable *exe = [graph compileWithDevice:[MPSGraphDevice deviceWithMTLDevice:device]
                                                 feeds:@{input: [[MPSGraphShapedType alloc] initWithShape:inShape dataType:fp8Type]}
                                         targetTensors:@[cur]
                                      targetOperations:nil
                                 compilationDescriptor:cd];
    uint64_t dtComp = clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tComp0;
    if (!exe) {
      printf("❌ Compilation failed!\n");
      return 1;
    }
    printf("   Compilation completed in %.2f ms\n", (double)dtComp / 1e6);

    size_t inBytes = B * Ci * H * W;
    size_t outBytes = B * Co * H * W;
    id<MTLBuffer> iBuf = [device newBufferWithLength:inBytes options:0];
    id<MTLBuffer> oBuf = [device newBufferWithLength:outBytes options:0];
    fillMultiTierFP8(iBuf.contents, inBytes, 0x9E3779B97F4A7C15ULL);
    memset(oBuf.contents, 0, outBytes);

    MPSGraphTensorData *iData = [[MPSGraphTensorData alloc] initWithMTLBuffer:iBuf shape:inShape dataType:fp8Type];
    MPSGraphTensorData *oData = [[MPSGraphTensorData alloc] initWithMTLBuffer:oBuf shape:inShape dataType:fp8Type];

    MPSGraphExecutableExecutionDescriptor *ed = [MPSGraphExecutableExecutionDescriptor new];
    ed.waitUntilCompleted = YES;

    // Warmup
    [exe runWithMTLCommandQueue:q inputsArray:@[iData] resultsArray:@[oData] executionDescriptor:ed];

    int iters = 20;
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < iters; i++) {
      [exe runWithMTLCommandQueue:q inputsArray:@[iData] resultsArray:@[oData] executionDescriptor:ed];
    }
    clock_gettime(CLOCK_MONOTONIC, &end);

    double sec = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    double avgMs = (sec / iters) * 1000.0;
    double totalOps = 2.0 * B * H * W * Ci * Co * K * K * L;
    double tops = (totalOps / 1e12) / (sec / iters);

    uint8_t *outPtr = (uint8_t *)oBuf.contents;
    size_t zeroCount = 0;
    for (size_t i = 0; i < outBytes; i++) {
      if (outPtr[i] == 0x00 || outPtr[i] == 0x80) zeroCount++;
    }

    printf("📊 Results:\n");
    printf("   Latency:   %6.2f ms\n", avgMs);
    printf("   Framerate: %6.1f FPS\n", 1000.0 / avgMs);
    printf("   Speed:     %7.2f TOPS\n", tops);
    printf("   Zero Audit: %lu / %lu zeros (%.3f%%)\n", zeroCount, outBytes, (double)zeroCount * 100.0 / outBytes);
    printf("   Sample Bytes: [0x%02x, 0x%02x, 0x%02x, 0x%02x]\n", outPtr[0], outPtr[1], outPtr[2], outPtr[3]);
    printf("========================================================================\n\n");
    return 0;
  }
}
