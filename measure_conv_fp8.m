/**
 * measure_conv_fp8.m - MPSGraph FP8 (Float8E4M3) QDQ Convolution Benchmark
 *
 * TECHNICAL BACKGROUND:
 * ---------------------
 * 1. MPSGraph Conv2D Operand Restriction:
 *    Attempting to pass raw FP8 tensors directly into `convolution2DWithSourceTensor:weightsTensor:`
 *    causes an MLIR compiler failure:
 *      "'mps.conv_2d' op operand #0 must be tensor of mps native type values, but got 'tensor<...xf8E4M3FN>'"
 *    Underlying Metal hardware convolution engines operate on FP16 or FP32. Therefore, FP8
 *    convolution in MPSGraph requires Quantize-Dequantize (QDQ) flow.
 *
 * 2. QDQ Patterns Supported:
 *    - W8A8 QDQ (Full Activation & Weight QDQ):
 *        Input & weights are stored in FP8.
 *        Activations and weights are dequantized to FP16 before each convolution layer,
 *        and output activations are requantized to FP8.
 *    - Weight-Only QDQ (FP8 Weights, FP16 Activations):
 *        Weights are compressed as 8-bit FP8 (cutting weight storage/bandwidth by 2x),
 *        dequantized to FP16 once, and convolutions proceed at full FP16 speed.
 *
 * 3. Supported FP8 Formats (macOS 27+ / iOS 27+):
 *    - MPSDataTypeFloat8e4m3 (E4M3: 1 sign, 4 exp, 3 mantissa; IEEE 754-style for weights/activations)
 *    Note: MPSGraph dequantize/quantize passes only support E4M3. E5M2 is rejected
 *    with "Unsupported quantization scheme".
 *
 * 4. Apple Neural Engine (ANE) Behavior:
 *    The physical ANE MAC arrays (H13-H16) only possess arithmetic ALUs for FP16 and INT8.
 *    ANECCompile returns an internal error when encountering FP8 MLIR operations:
 *      "MLIR MPS to ANEC conversion failed"
 *    MPSGraph catches this and automatically falls back to Metal GPU execution.
 *
 * 5. Weight/Input Fill Pattern:
 *    Random-sign, magnitude 1/32, RMS-scaled so the expected gain per layer
 *    (sqrt(Ci*K*K)/32 ~= 1.06 for Ci=128, K=3) stays near 1 across the 20
 *    chained layers -- same convention as MILSpecBuilder's MILWeightModeDense.
 *    An earlier same-sign, magnitude-~1 fill grew by ~1152x per layer and
 *    saturated to NaN/448 by layer 2, silently benchmarking overflow instead
 *    of real arithmetic for 18 of the 20 layers.
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
  FP8BenchModeFullQDQ = 0,       // W8A8 QDQ: FP8 in/out activations, FP8 weights
  FP8BenchModeWeightOnly = 1,    // Weight-only: FP8 weights dequantized to FP16, FP16 activations
};

static const char *formatName(MPSDataType dataType) {
  if (dataType == MPSDataTypeFloat8e4m3) return "FP8 E4M3";
  return "Unknown";
}

// Random-sign, magnitude 2^shift (default shift=-5, i.e. 1/32; E4M3 0x10 =
// +0.03125, 0x90 = -0.03125). Chained through L=20 conv layers with a
// Ci*K*K=1152 reduction, a same-sign, magnitude-~1 fill (the pattern this
// file started with) grows by ~1152x per layer: by layer 2 the accumulator
// already exceeds both FP16 (max 65504) and FP8 E4M3 (max 448), and the
// Check: values below were consequently NaN (0x7e00) or saturated (0x7e/448)
// -- every later layer just propagated that, not measuring real arithmetic.
// Random sign at 1/32 makes the expected per-layer RMS gain sqrt(1152)/32 ~=
// 1.06, so magnitude stays bounded across the whole chain on GPU (verified:
// finite, small, non-degenerate output). --shift exists because 1/32 sits
// only one exponent step above E4M3's minimum normal (0.015625) -- on
// hardware with a real ANE FP8 datapath (e.g. iPhone 18 Pro's H18, unlike
// this file's "falls back to GPU" assumption which only holds up through
// H16g), that headroom may not be enough and could underflow to hardware
// zero, which then triggers H17+ zero-skip and reports an inflated,
// degenerate TOPS number -- exactly the failure mode this repo has
// documented for FP16/INT8 zero tensors, now suspected for FP8. Same
// convention as MILSpecBuilder's MILWeightModeDense and fillDenseFloat16
// elsewhere in this repo. Deterministic (fixed-seed xorshift64) so runs are
// reproducible.
static void fillFP8Random(void *buffer, size_t byteCount, uint64_t seed, int shift) {
  if (!buffer || byteCount == 0) return;
  int storedExp = shift + 7;  // E4M3 bias = 7
  if (storedExp < 1) storedExp = 1;
  if (storedExp > 14) storedExp = 14;
  uint8_t posByte = (uint8_t)(storedExp << 3);
  uint8_t negByte = (uint8_t)(0x80 | posByte);
  uint8_t *p = (uint8_t *)buffer;
  uint64_t state = seed;
  for (size_t i = 0; i < byteCount; i++) {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    p[i] = (state & 1) ? negByte : posByte;
  }
}

static void fillFP16Random(void *buffer, size_t byteCount, uint64_t seed, int shift) {
  if (!buffer || byteCount == 0) return;
  int storedExp = shift + 15;  // FP16 bias = 15
  if (storedExp < 1) storedExp = 1;
  if (storedExp > 30) storedExp = 30;
  uint16_t posWord = (uint16_t)(storedExp << 10);
  uint16_t negWord = (uint16_t)(0x8000 | posWord);
  uint16_t *p = (uint16_t *)buffer;
  size_t count = byteCount / sizeof(uint16_t);
  uint64_t state = seed;
  for (size_t i = 0; i < count; i++) {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    p[i] = (state & 1) ? negWord : posWord;
  }
}

static void run_bench_fp8_qdq(id<MTLDevice> device, bool useANE,
                              MPSDataType fp8Type, FP8BenchMode mode,
                              NSUInteger layers, int shift) {
  @autoreleasepool {
    const char *target = useANE ? "ANE" : "GPU";
    const char *fName = formatName(fp8Type);
    const char *mName = (mode == FP8BenchModeFullQDQ) ? "W8A8 QDQ" : "Weight-Only QDQ";

    printf("--> Testing [%s | %s | %s] (layers=%lu, magnitude=2^%d)...\n",
           target, fName, mName, (unsigned long)layers, shift);
    fflush(stdout);

    @try {
      NSUInteger B = 1, H = 256, W = 256, Ci = 128, Co = 128, K = 3, L = layers;
      NSArray *inShape = @[ @(B), @(Ci), @(H), @(W) ];
      NSArray *wShape = @[ @(Co), @(Ci), @(K), @(K) ];

      MPSGraph *graph = [MPSGraph new];
      MPSDataType actType = (mode == FP8BenchModeFullQDQ) ? fp8Type : MPSDataTypeFloat16;

      MPSGraphTensor *input = [graph placeholderWithShape:inShape
                                                 dataType:actType
                                                     name:@"in"];
      MPSGraphTensor *cur = input;

      // Weights stored in FP8 (1 byte per element)
      NSMutableData *wData = [NSMutableData dataWithLength:Co * Ci * K * K * sizeof(uint8_t)];
      fillFP8Random(wData.mutableBytes, wData.length, 0x5EED5EED5EED5EEDULL, shift);
      MPSGraphTensor *wFP8 = [graph constantWithData:wData shape:wShape dataType:fp8Type];

      // Dequantize weights from FP8 to FP16 (scale = 1.0, zeroPoint = 0.0)
      MPSGraphTensor *w = [graph dequantizeTensor:wFP8
                                            scale:1.0
                                        zeroPoint:0.0
                                         dataType:MPSDataTypeFloat16
                                             name:@"w_dequant"];

      MPSGraphConvolution2DOpDescriptor *d = [MPSGraphConvolution2DOpDescriptor
          descriptorWithStrideInX:1
                        strideInY:1
                  dilationRateInX:1
                  dilationRateInY:1
                           groups:1
                     paddingStyle:MPSGraphPaddingStyleTF_SAME
                       dataLayout:MPSGraphTensorNamedDataLayoutNCHW
                    weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];

      for (int i = 0; i < (int)L; i++) {
        if (mode == FP8BenchModeFullQDQ) {
          // Dequantize activation to FP16
          MPSGraphTensor *inFP16 = [graph dequantizeTensor:cur
                                                     scale:1.0
                                                 zeroPoint:0.0
                                                  dataType:MPSDataTypeFloat16
                                                      name:nil];
          // Convolve in FP16
          MPSGraphTensor *outFP16 = [graph convolution2DWithSourceTensor:inFP16
                                                           weightsTensor:w
                                                              descriptor:d
                                                                    name:nil];
          // Quantize back to FP8
          cur = [graph quantizeTensor:outFP16
                                scale:1.0
                            zeroPoint:0.0
                             dataType:fp8Type
                                 name:nil];
        } else {
          // Weight-only: Convolve FP16 activations with dequantized FP8 weights
          cur = [graph convolution2DWithSourceTensor:cur
                                       weightsTensor:w
                                          descriptor:d
                                                name:nil];
        }
      }

      MPSGraphDevice *mDev = [MPSGraphDevice deviceWithMTLDevice:device];
      MPSGraphCompilationDescriptor *cd = [MPSGraphCompilationDescriptor new];
      cd.optimizationLevel =
          useANE ? MPSGraphOptimizationLevel1 : MPSGraphOptimizationLevel0;

      if (useANE) {
        if ([cd respondsToSelector:@selector(setPreferredDevice:)]) {
          cd.preferredDevice = 2; // MPSGraphDeviceTypeANE
        } else if ([MPSGraphDevice respondsToSelector:@selector(ANEDevice)]) {
          mDev = [MPSGraphDevice ANEDevice];
        }
      }

      MPSGraphExecutable *exe = [graph compileWithDevice:mDev
                                                   feeds:@{input: [[MPSGraphShapedType alloc] initWithShape:inShape dataType:actType]}
                                           targetTensors:@[ cur ]
                                        targetOperations:nil
                                   compilationDescriptor:cd];
      if (!exe) {
        NSLog(@"[%s | %s | %s] Compilation failed.", target, fName, mName);
        return;
      }

      size_t inBytes = B * Ci * H * W * ((mode == FP8BenchModeFullQDQ) ? sizeof(uint8_t) : sizeof(uint16_t));
      id<MTLBuffer> iBuf = [device newBufferWithLength:inBytes options:0];
      if (mode == FP8BenchModeFullQDQ) {
        fillFP8Random(iBuf.contents, inBytes, 0x9E3779B97F4A7C15ULL, shift);
      } else {
        fillFP16Random(iBuf.contents, inBytes, 0x9E3779B97F4A7C15ULL, shift);
      }
      MPSGraphTensorData *iData = [[MPSGraphTensorData alloc] initWithMTLBuffer:iBuf
                                                                          shape:inShape
                                                                       dataType:actType];

      size_t outBytes = B * Co * H * W * ((mode == FP8BenchModeFullQDQ) ? sizeof(uint8_t) : sizeof(uint16_t));
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
      double totalOps = 2.0 * B * H * W * Ci * Co * K * K * L;
      double tops = (totalOps / 1e12) / avg;

      if (mode == FP8BenchModeFullQDQ) {
        uint8_t *outPtr = (uint8_t *)oBuf.contents;
        size_t totalElem = B * Co * H * W;
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
        size_t totalElem = B * Co * H * W;
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
    NSUInteger layers = 20;
    int shift = -5;  // magnitude 2^shift; default 1/32

    for (int i = 1; i < argc; i++) {
      NSString *arg = [NSString stringWithUTF8String:argv[i]];
      if ([arg isEqualToString:@"--layers"] && i + 1 < argc) {
        layers = (NSUInteger)atoi(argv[++i]);
      } else if ([arg isEqualToString:@"--shift"] && i + 1 < argc) {
        shift = atoi(argv[++i]);
      } else if ([arg isEqualToString:@"--help"] || [arg isEqualToString:@"-h"]) {
        printf("Usage: %s [--layers N] [--shift E]\n", argv[0]);
        printf("  --layers N  chained conv layers (default: 20). Use 1 to isolate\n");
        printf("              whether a single QDQ round-trip already underflows.\n");
        printf("  --shift E   weight/input magnitude = 2^E, random sign (default: -5,\n");
        printf("              i.e. 1/32). Try a larger E (e.g. 0, for magnitude 1.0)\n");
        printf("              if output is suspiciously all-zero on real ANE FP8\n");
        printf("              hardware -- see the fillFP8Random comment.\n");
        return 0;
      } else {
        fprintf(stderr, "error: unknown argument '%s' (try --help)\n", arg.UTF8String);
        return 1;
      }
    }
    if (layers == 0) {
      fprintf(stderr, "error: --layers must be at least 1\n");
      return 1;
    }

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    NSLog(@"Starting MPSGraph FP8 QDQ Benchmark on %s\n", device.name.UTF8String);

    printf("========================================================\n");
    printf(" 1. Metal GPU FP8 QDQ Benchmark\n");
    printf("========================================================\n");
    run_bench_fp8_qdq(device, false, MPSDataTypeFloat8e4m3, FP8BenchModeWeightOnly, layers, shift);
    run_bench_fp8_qdq(device, false, MPSDataTypeFloat8e4m3, FP8BenchModeFullQDQ, layers, shift);

    printf("\n========================================================\n");
    printf(" 2. Apple Neural Engine (ANE) FP8 QDQ Benchmark\n");
    printf("    (Note: on H16g this falls back to GPU -- ANECCompile rejects FP8\n");
    printf("    MLIR. Newer silicon (H17+) may have a real ANE FP8 datapath; if\n");
    printf("    Check: is all-zero here with an implausibly high TOPS, that's\n");
    printf("    the classic hardware zero-skip signature -- try --shift 0.)\n");
    printf("========================================================\n");
    run_bench_fp8_qdq(device, true, MPSDataTypeFloat8e4m3, FP8BenchModeWeightOnly, layers, shift);
    run_bench_fp8_qdq(device, true, MPSDataTypeFloat8e4m3, FP8BenchModeFullQDQ, layers, shift);

    return 0;
  }
}
