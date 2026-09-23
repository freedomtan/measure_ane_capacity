/**
 * measure_matmul_fp8.m - MPSGraph FP8 (Float8E4M3) QDQ Matrix Multiplication Benchmark
 *
 * Companion to measure_conv_fp8.m, same QDQ rationale, applied to
 * matrixMultiplicationWithPrimaryTensor:secondaryTensor: instead of
 * convolution2DWithSourceTensor:weightsTensor:. mps.matmul (like mps.conv_2d)
 * requires MPS-native operand types, so raw FP8 tensors are rejected and QDQ
 * is mandatory:
 *   - W8A8 QDQ: FP8 in/out activations, FP8 weights, dequantize -> matmul -> quantize.
 *   - Weight-Only QDQ: FP8 weights dequantized to FP16 once, FP16 activations.
 * As with conv, the ANE's MAC arrays only have FP16/INT8 ALUs, so ANECCompile
 * rejects FP8 MLIR and MPSGraph silently falls back to Metal GPU.
 */

#import <time.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

@interface MPSGraphDevice (ANE)
+ (instancetype)ANEDevice;
@end

@interface MPSGraphCompilationDescriptor (Private)
@property (nonatomic, assign) unsigned long long preferredDevice;
@end

typedef NS_ENUM(NSInteger, FP8BenchMode) {
  FP8BenchModeFullQDQ = 0,     // W8A8 QDQ: FP8 in/out activations, FP8 weights
  FP8BenchModeWeightOnly = 1,  // Weight-only: FP8 weights dequantized to FP16, FP16 activations
};

static const char *formatName(MPSDataType dataType) {
  if (dataType == MPSDataTypeFloat8e4m3) return "FP8 E4M3";
  return "Unknown";
}

// Random-sign, magnitude 1/32 (E4M3 0x10 = +0.03125, 0x90 = -0.03125). Chained
// through L=20 layers of a K=1024 GEMM, a same-sign, magnitude-~1 fill (the
// pattern this file started with) grows by ~K per layer: by layer 2 the
// accumulator already exceeds both FP16 (max 65504) and FP8 E4M3 (max 448),
// and every later layer just propagates Inf/NaN -- the benchmark stops
// measuring real arithmetic after one layer. Random sign at 1/sqrt(K) makes
// the expected per-layer RMS gain 1.0 (sqrt(1024)/32 == 1024/32/32), so
// magnitude stays bounded across the whole chain. Same convention as
// MILSpecBuilder's MILWeightModeDense and fillDenseFloat16 elsewhere in this
// repo. Deterministic (fixed-seed xorshift64) so runs are reproducible.
static void fillFP8Random(void *buffer, size_t byteCount, uint64_t seed) {
  if (!buffer || byteCount == 0) return;
  uint8_t *p = (uint8_t *)buffer;
  uint64_t state = seed;
  for (size_t i = 0; i < byteCount; i++) {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    p[i] = (state & 1) ? 0xB0 : 0x30;  // -0.5 : +0.5 (physical magnitude to avoid H18 zero cliff)
  }
}

static void fillFP16Random(void *buffer, size_t byteCount, uint64_t seed) {
  if (!buffer || byteCount == 0) return;
  uint16_t *p = (uint16_t *)buffer;
  size_t count = byteCount / sizeof(uint16_t);
  uint64_t state = seed;
  for (size_t i = 0; i < count; i++) {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    p[i] = (state & 1) ? 0xA800 : 0x2800;  // -0.03125 : +0.03125
  }
}

static void run_bench_matmul_fp8_qdq(id<MTLDevice> device, bool useANE,
                                     MPSDataType fp8Type, FP8BenchMode mode) {
  @autoreleasepool {
    const char *target = useANE ? "ANE" : "GPU";
    const char *fName = formatName(fp8Type);
    const char *mName = (mode == FP8BenchModeFullQDQ) ? "W8A8 QDQ" : "Weight-Only QDQ";

    printf("--> Testing [%s | %s | %s]...\n", target, fName, mName);
    fflush(stdout);

    @try {
      // Map GEMM to 1x1 2D Convolution on ANE:
      //   Input:   [B, K, H, W] where H * W = M (32 x 32 = 1024)
      //   Weights: [N, K, 1, 1]
      //   Output:  [B, N, H, W]
      // This maps reduction across the 64-wide ANE channel MAC array and validates QDQ without GPU fallback.
      NSUInteger B = 1, M = 1024, K = 1024, N = 1024, L = 20;
      NSUInteger H = 32, W = 32;
      NSArray *inShape = @[ @(B), @(K), @(H), @(W) ];
      NSArray *wShape = @[ @(N), @(K), @1, @1 ];

      MPSGraph *graph = [MPSGraph new];
      MPSDataType actType = (mode == FP8BenchModeFullQDQ) ? fp8Type : MPSDataTypeFloat16;
      double fp8Scale = 0.0625; // 2^-4 scale avoiding H18/H19 underflow cliff

      MPSGraphTensor *input = [graph placeholderWithShape:inShape
                                                 dataType:actType
                                                     name:@"in"];
      MPSGraphTensor *cur = input;

      // Weights stored in FP8 (1 byte per element)
      NSMutableData *wData = [NSMutableData dataWithLength:N * K * 1 * 1 * sizeof(uint8_t)];
      fillFP8Random(wData.mutableBytes, wData.length, 0x5EED5EED5EED5EEDULL);
      MPSGraphTensor *wFP8 = [graph constantWithData:wData shape:wShape dataType:fp8Type];

      // Dequantize weights from FP8 to FP16
      MPSGraphTensor *w = [graph dequantizeTensor:wFP8
                                            scale:fp8Scale
                                        zeroPoint:0.0
                                         dataType:MPSDataTypeFloat16
                                             name:@"w_dequant"];

      MPSGraphConvolution2DOpDescriptor *convDesc = [MPSGraphConvolution2DOpDescriptor
          descriptorWithStrideInX:1 strideInY:1 dilationRateInX:1 dilationRateInY:1
                           groups:1 paddingStyle:MPSGraphPaddingStyleTF_SAME
                       dataLayout:MPSGraphTensorNamedDataLayoutNCHW
                    weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];

      for (int i = 0; i < (int)L; i++) {
        if (mode == FP8BenchModeFullQDQ) {
          // Dequantize activation to FP16
          MPSGraphTensor *inFP16 = [graph dequantizeTensor:cur
                                                     scale:fp8Scale
                                                 zeroPoint:0.0
                                                  dataType:MPSDataTypeFloat16
                                                      name:nil];
          // 1x1 Convolution in FP16
          MPSGraphTensor *outFP16 = [graph convolution2DWithSourceTensor:inFP16
                                                           weightsTensor:w
                                                              descriptor:convDesc
                                                                    name:nil];
          // Quantize back to FP8
          cur = [graph quantizeTensor:outFP16
                                scale:fp8Scale
                            zeroPoint:0.0
                             dataType:fp8Type
                                 name:nil];
        } else {
          // Weight-only: multiply FP16 activations with dequantized FP8 weights
          cur = [graph convolution2DWithSourceTensor:cur
                                       weightsTensor:w
                                          descriptor:convDesc
                                                name:nil];
        }
      }

      MPSGraphDevice *mDev = [MPSGraphDevice deviceWithMTLDevice:device];
      MPSGraphCompilationDescriptor *cd = [MPSGraphCompilationDescriptor new];
      cd.optimizationLevel =
          useANE ? MPSGraphOptimizationLevel1 : MPSGraphOptimizationLevel0;

      if (useANE) {
        if ([cd respondsToSelector:@selector(setPreferredDevice:)]) {
          cd.preferredDevice = 2;  // MPSGraphDeviceTypeANE
        } else if ([MPSGraphDevice respondsToSelector:@selector(ANEDevice)]) {
          mDev = [MPSGraphDevice ANEDevice];
        }
      }

      MPSGraphExecutable *exe = [graph compileWithDevice:mDev
                                                   feeds:@{input : [[MPSGraphShapedType alloc] initWithShape:inShape dataType:actType]}
                                           targetTensors:@[ cur ]
                                        targetOperations:nil
                                   compilationDescriptor:cd];
      if (!exe) {
        NSLog(@"[%s | %s | %s] Compilation failed.", target, fName, mName);
        return;
      }

      size_t inBytes = B * M * K * ((mode == FP8BenchModeFullQDQ) ? sizeof(uint8_t) : sizeof(uint16_t));
      id<MTLBuffer> iBuf = [device newBufferWithLength:inBytes options:0];
      if (mode == FP8BenchModeFullQDQ) {
        fillFP8Random(iBuf.contents, inBytes, 0x9E3779B97F4A7C15ULL);
      } else {
        fillFP16Random(iBuf.contents, inBytes, 0x9E3779B97F4A7C15ULL);
      }
      MPSGraphTensorData *iData = [[MPSGraphTensorData alloc] initWithMTLBuffer:iBuf
                                                                          shape:inShape
                                                                       dataType:actType];

      size_t outBytes = B * M * N * ((mode == FP8BenchModeFullQDQ) ? sizeof(uint8_t) : sizeof(uint16_t));
      id<MTLBuffer> oBuf = [device newBufferWithLength:outBytes options:0];
      MPSGraphTensorData *oData = [[MPSGraphTensorData alloc] initWithMTLBuffer:oBuf
                                                                          shape:inShape
                                                                       dataType:actType];

      id<MTLCommandQueue> q = [device newCommandQueue];
      MPSGraphExecutableExecutionDescriptor *ed = [MPSGraphExecutableExecutionDescriptor new];
      ed.waitUntilCompleted = YES;

      // Warmup pass
      [exe runWithMTLCommandQueue:q
                      inputsArray:@[ iData ]
                     resultsArray:@[ oData ]
              executionDescriptor:ed];

      NSUInteger iterations = 20;
      struct timespec start, end;
      clock_gettime(CLOCK_MONOTONIC, &start);
      for (int i = 0; i < (int)iterations; i++) {
        [exe runWithMTLCommandQueue:q
                        inputsArray:@[ iData ]
                       resultsArray:@[ oData ]
                executionDescriptor:ed];
      }
      clock_gettime(CLOCK_MONOTONIC, &end);

      double duration = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
      double avg = duration / iterations;
      // 2.0 * B * M * K * N FLOPs per layer * L layers, same as measure_matmul_universal.m.
      double tops = (2.0 * B * M * K * N * L) / (avg * 1e12);

      if (mode == FP8BenchModeFullQDQ) {
        uint8_t *outPtr = (uint8_t *)oBuf.contents;
        size_t totalElem = B * M * N;
        size_t zeroCount = 0;
        for (size_t k = 0; k < totalElem; k++) {
          if (outPtr[k] == 0) zeroCount++;
        }
        NSLog(@"[%s | %-8s | %-15s] Avg: %6.2f ms, Speed: %7.4f TOPS | Check: [0x%02x, 0x%02x, 0x%02x, 0x%02x] (zeros: %lu/%lu)",
              target, fName, mName, avg * 1000.0, tops,
              outPtr[0], outPtr[1], outPtr[2], outPtr[3],
              (unsigned long)zeroCount, (unsigned long)totalElem);
      } else {
        uint16_t *outPtr = (uint16_t *)oBuf.contents;
        size_t totalElem = B * M * N;
        size_t zeroCount = 0;
        for (size_t k = 0; k < totalElem; k++) {
          if (outPtr[k] == 0) zeroCount++;
        }
        NSLog(@"[%s | %-8s | %-15s] Avg: %6.2f ms, Speed: %7.4f TOPS | Check: [0x%04x, 0x%04x, 0x%04x, 0x%04x] (zeros: %lu/%lu)",
              target, fName, mName, avg * 1000.0, tops,
              outPtr[0], outPtr[1], outPtr[2], outPtr[3],
              (unsigned long)zeroCount, (unsigned long)totalElem);
      }

    } @catch (NSException *e) {
      NSLog(@"[%s | %s | %s] Caught exception: %@", target, fName, mName, e.reason);
    }
  }
}

int main(int argc, char *argv[]) {
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    NSLog(@"Starting MPSGraph FP8 QDQ Matrix Multiplication Benchmark on %s\n", device.name.UTF8String);

    printf("========================================================\n");
    printf(" 1. Metal GPU FP8 QDQ Benchmark\n");
    printf("========================================================\n");
    run_bench_matmul_fp8_qdq(device, false, MPSDataTypeFloat8e4m3, FP8BenchModeWeightOnly);
    run_bench_matmul_fp8_qdq(device, false, MPSDataTypeFloat8e4m3, FP8BenchModeFullQDQ);

    printf("\n========================================================\n");
    printf(" 2. Apple Neural Engine (ANE) FP8 QDQ Benchmark\n");
    printf("    (Note: ANE lacks native FP8 ALUs; ANECCompile falls back to GPU)\n");
    printf("========================================================\n");
    run_bench_matmul_fp8_qdq(device, true, MPSDataTypeFloat8e4m3, FP8BenchModeWeightOnly);
    run_bench_matmul_fp8_qdq(device, true, MPSDataTypeFloat8e4m3, FP8BenchModeFullQDQ);

    return 0;
  }
}
