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

#import <limits.h>
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

// Random-sign, magnitude 2^shift. Chained through L=20 conv layers with a
// Ci*K*K=1152 reduction, a same-sign, magnitude-~1 fill (the pattern this
// file started with) grows by ~1152x per layer: by layer 2 the accumulator
// already exceeds both FP16 (max 65504) and FP8 E4M3 (max 448), and the
// Check: values were consequently NaN (0x7e00) or saturated (0x7e/448) --
// every later layer just propagated that, not measuring real arithmetic.
// Random sign at magnitude 1/32 makes the expected per-layer RMS gain
// sqrt(1152)/32 ~= 1.06, so magnitude stays bounded across the whole chain
// (verified directly across all 20 layers on GPU: 1/32 is the only magnitude
// among {1/32, 1/16, 1/8, 1/4, 1/2} that stays unsaturated by layer 20 -- see
// kLogicalShiftDefault/kPhysicalShiftDefault above run_bench_fp8_qdq for why
// the physical FP8-byte magnitude and the logical FP16-math magnitude are
// deliberately different values on hardware with a real ANE FP8 datapath.
// Same convention as MILSpecBuilder's MILWeightModeDense and
// fillDenseFloat16 elsewhere in this repo. Deterministic (fixed-seed
// xorshift64) so runs are reproducible.
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

// W8A8 QDQ requantizes every layer with scale=1.0, but that does NOT reset
// activation magnitude to a constant: layer N+1's input is layer N's actual
// output, so magnitude still compounds across the chain exactly like
// Weight-Only mode's does. Measured directly on this GPU across the full
// 20-layer chain: 1/32 is the ONLY magnitude of {1/32, 1/16, 1/8, 1/4, 1/2}
// that stays unsaturated by layer 20 (matches the sqrt(Ci*K*K)/32 ~= 1.06
// per-layer-gain derivation -- it is not an arbitrary choice). So the
// magnitude the FP16 *math* needs to see (the "logical" magnitude) is fixed
// at 1/32 regardless of hardware quirks.
//
// The problem: on iPhone 18 Pro (H18), storing FP8 bytes at literally 1/32
// and quantizing/dequantizing at scale=1.0 hits a hard underflow-to-
// hardware-zero cliff on the ANE (confirmed: all 8388608/8388608 elements
// exactly zero at --layers 1, i.e. after a single round-trip, paired with an
// implausible ~65 TOPS -- H17+ zero-skip inflating a degenerate result).
// 1/16 through 1/2 all measured clean on that hardware (no cliff, no
// saturation, exact linear response) -- but per the paragraph above, storing
// bytes at those magnitudes with scale=1.0 breaks growth-stability across 20
// layers instead.
//
// Fix: decouple what's physically stored in FP8 (kPhysicalShift, chosen to
// avoid the ANE cliff) from what the FP16 math sees (kLogicalShift, chosen
// for 20-layer stability) via quantize/dequantize's scale parameter:
// dequantized_fp16 = stored_fp8_byte_value * scale, so
// scale = 2^(kLogicalShift - kPhysicalShift) makes bytes physically encoded
// at a hardware-safe magnitude while the downstream arithmetic still runs at
// the growth-stable magnitude.
static const int kLogicalShiftDefault = -5;   // FP16-domain magnitude for stability
static const int kPhysicalShiftDefault = -1;  // FP8-domain magnitude, safe on H18

static void run_bench_fp8_qdq(id<MTLDevice> device, bool useANE,
                              MPSDataType fp8Type, FP8BenchMode mode,
                              NSUInteger layers, int logicalShift,
                              int physicalShift) {
  @autoreleasepool {
    const char *target = useANE ? "ANE" : "GPU";
    const char *fName = formatName(fp8Type);
    const char *mName = (mode == FP8BenchModeFullQDQ) ? "W8A8 QDQ" : "Weight-Only QDQ";
    double scale = exp2((double)(logicalShift - physicalShift));

    printf("--> Testing [%s | %s | %s] (layers=%lu, logical=2^%d, "
           "physical=2^%d, scale=%.6f)...\n",
           target, fName, mName, (unsigned long)layers, logicalShift,
           physicalShift, scale);
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

      // Weights stored in FP8 (1 byte per element) at the hardware-safe
      // physical magnitude; scale brings the dequantized FP16 value to the
      // growth-stable logical magnitude.
      NSMutableData *wData = [NSMutableData dataWithLength:Co * Ci * K * K * sizeof(uint8_t)];
      fillFP8Random(wData.mutableBytes, wData.length, 0x5EED5EED5EED5EEDULL, physicalShift);
      MPSGraphTensor *wFP8 = [graph constantWithData:wData shape:wShape dataType:fp8Type];

      MPSGraphTensor *w = [graph dequantizeTensor:wFP8
                                            scale:scale
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
          // Dequantize activation to FP16 (physical -> logical magnitude)
          MPSGraphTensor *inFP16 = [graph dequantizeTensor:cur
                                                     scale:scale
                                                 zeroPoint:0.0
                                                  dataType:MPSDataTypeFloat16
                                                      name:nil];
          // Convolve in FP16
          MPSGraphTensor *outFP16 = [graph convolution2DWithSourceTensor:inFP16
                                                           weightsTensor:w
                                                              descriptor:d
                                                                    name:nil];
          // Quantize back to FP8 (logical -> physical magnitude)
          cur = [graph quantizeTensor:outFP16
                                scale:scale
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
        // Physical magnitude: this buffer is real FP8 bytes fed to hardware.
        fillFP8Random(iBuf.contents, inBytes, 0x9E3779B97F4A7C15ULL, physicalShift);
      } else {
        // Weight-Only mode's activations are plain FP16 the entire chain --
        // never encoded as FP8, so there is no hardware-cliff risk here and
        // no scale trick is needed; fill directly at the logical magnitude.
        fillFP16Random(iBuf.contents, inBytes, 0x9E3779B97F4A7C15ULL, logicalShift);
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
    int logicalShift = kLogicalShiftDefault;
    int physicalShift = kPhysicalShiftDefault;

    for (int i = 1; i < argc; i++) {
      NSString *arg = [NSString stringWithUTF8String:argv[i]];
      if ([arg isEqualToString:@"--layers"] && i + 1 < argc) {
        layers = (NSUInteger)atoi(argv[++i]);
      } else if ([arg isEqualToString:@"--logical-shift"] && i + 1 < argc) {
        logicalShift = atoi(argv[++i]);
      } else if ([arg isEqualToString:@"--physical-shift"] && i + 1 < argc) {
        physicalShift = atoi(argv[++i]);
      } else if ([arg isEqualToString:@"--help"] || [arg isEqualToString:@"-h"]) {
        printf("Usage: %s [--layers N] [--logical-shift E] [--physical-shift E]\n", argv[0]);
        printf("  --layers N          chained conv layers (default: 20). Use 1 to\n");
        printf("                      isolate whether a single QDQ round-trip\n");
        printf("                      already underflows.\n");
        printf("  --logical-shift E   FP16-math magnitude = 2^E (default: -5, i.e.\n");
        printf("                      1/32). Chosen for 20-layer growth stability --\n");
        printf("                      verified directly: only -5 stays unsaturated\n");
        printf("                      by layer 20 among {-5..-1}. Do not raise this\n");
        printf("                      without re-checking the full chain.\n");
        printf("  --physical-shift E  FP8-byte magnitude = 2^E (default: -1, i.e.\n");
        printf("                      1/2). What's literally stored/rounded to FP8\n");
        printf("                      and fed to hardware; quantize/dequantize scale\n");
        printf("                      = 2^(logical-physical) bridges the two. Sweep\n");
        printf("                      this on real ANE FP8 hardware to bracket an\n");
        printf("                      all-zero underflow cliff or 0x7e/0xfe\n");
        printf("                      saturation (confirmed on iPhone 18 Pro at -5).\n");
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
    run_bench_fp8_qdq(device, false, MPSDataTypeFloat8e4m3, FP8BenchModeWeightOnly, layers, logicalShift, physicalShift);
    run_bench_fp8_qdq(device, false, MPSDataTypeFloat8e4m3, FP8BenchModeFullQDQ, layers, logicalShift, physicalShift);

    printf("\n========================================================\n");
    printf(" 2. Apple Neural Engine (ANE) FP8 QDQ Benchmark\n");
    printf("    (Note: on H16g this falls back to GPU -- ANECCompile rejects FP8\n");
    printf("    MLIR. Newer silicon (H17+) may have a real ANE FP8 datapath; on\n");
    printf("    iPhone 18 Pro, storing FP8 bytes at the logical 1/32 magnitude\n");
    printf("    directly (physical-shift == logical-shift, scale=1.0) hit a\n");
    printf("    confirmed all-zero underflow cliff on W8A8 QDQ's ANE path. The\n");
    printf("    default --physical-shift -1 avoids it by construction (scale\n");
    printf("    bridges physical and logical magnitude). If Check: is still\n");
    printf("    all-zero here with an implausibly high TOPS, that's the classic\n");
    printf("    hardware zero-skip signature -- sweep --physical-shift to\n");
    printf("    rebracket the cliff on this device.)\n");
    printf("========================================================\n");
    run_bench_fp8_qdq(device, true, MPSDataTypeFloat8e4m3, FP8BenchModeWeightOnly, layers, logicalShift, physicalShift);
    run_bench_fp8_qdq(device, true, MPSDataTypeFloat8e4m3, FP8BenchModeFullQDQ, layers, logicalShift, physicalShift);

    return 0;
  }
}
