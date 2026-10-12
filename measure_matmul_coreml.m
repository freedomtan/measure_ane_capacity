// CoreML matrix-multiplication capacity benchmark.
//
// Companion to measure_matmul_universal.m, which measures the same workload
// (L chained [B,1,M,K] x [B,1,K,N] matmuls) through MPSGraph. This binary
// drives the model via CoreML instead, so the two paths can be compared on
// identical arithmetic -- and, in particular, so CPU has a working path at
// all: MPSGraph's private CPUDevice SPI crashes unconditionally (see
// docs/CPU_Matrix_Convolution_Capability.md), where CoreML's CPU compute
// unit just works.
//
// The MIL program is built at runtime by MILSpecBuilder's matmul chain
// support, using the CoreML::Specification classes generated from
// coremltools' own .proto schema -- no Python, no .mlpackage. The serialized
// spec is written to a temporary .mlmodel file and compiled with
// compileModelAtURL: (measured ~28% faster than handing CoreML the spec in
// memory via MLModelAsset for the conv benchmark; assumed to hold here too),
// then the temp file is deleted.
//
// --precision fp16 (default), int8 (W8A8 QDQ), or fp8 (Float8E4M3 QDQ) --
// MILSpecBuilder's matmul chain builder mirrors its conv chain counterpart's
// quantize/dequantize pattern for all three.

#import <locale.h>
#import <time.h>

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

#import "MILSpecBuilder.h"

typedef struct {
  NSUInteger B, M, K, N, L;
  NSUInteger iterations;
  NSUInteger warmup;
  NSString *savePath;
  MLComputeUnits units;
  NSString *unitsName;
  MILWeightMode weightMode;
  MILPrecision precision;
  BOOL showPlan;
  BOOL showMIL;
  BOOL checkOutput;
  BOOL verbose;
} BenchConfig;

// Random-sign +/-0.0625 (fp16 0x2C00 / 0xAC00), matching
// measure_conv_coreml.m's fillDenseFloat16 -- non-cancelling under the
// matmul reduction, deterministic (fixed-seed xorshift64) for reproducible
// runs.
static void fillDenseFloat16(void *buffer, size_t byteCount) {
  if (!buffer || byteCount == 0) return;
  uint16_t *p = (uint16_t *)buffer;
  size_t count = byteCount / sizeof(uint16_t);
  uint64_t state = 0x9E3779B97F4A7C15ULL;
  for (size_t i = 0; i < count; i++) {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    p[i] = (state & 1) ? 0xAC00 : 0x2C00;
  }
}

static double nowSeconds(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec / 1e9;
}

static NSString *deviceLabel(id<MLComputeDeviceProtocol> device) {
  if (!device) return @"unknown";
  if ([device isKindOfClass:[MLNeuralEngineComputeDevice class]]) return @"ANE";
  if ([device isKindOfClass:[MLGPUComputeDevice class]]) return @"GPU";
  if ([device isKindOfClass:[MLCPUComputeDevice class]]) return @"CPU";
  return NSStringFromClass([device class]);
}

// Compile a .mlpackage/.mlmodel to .mlmodelc. Identical to
// measure_conv_coreml.m's compileModel().
static NSURL *compileModel(NSString *path, double *compileMs, NSError **error) {
  double start = nowSeconds();
  dispatch_semaphore_t sem = dispatch_semaphore_create(0);
  __block NSURL *compiled = nil;
  __block NSError *compileError = nil;
  [MLModel compileModelAtURL:[NSURL fileURLWithPath:path]
           completionHandler:^(NSURL *compiledModelURL, NSError *err) {
             if (compiledModelURL) {
               NSString *dest = [NSTemporaryDirectory()
                   stringByAppendingPathComponent:
                       [NSString stringWithFormat:@"measure_matmul_coreml_%@",
                                                  compiledModelURL.lastPathComponent]];
               NSURL *destURL = [NSURL fileURLWithPath:dest];
               NSFileManager *fm = [NSFileManager defaultManager];
               [fm removeItemAtURL:destURL error:nil];
               NSError *copyError = nil;
               if ([fm copyItemAtURL:compiledModelURL toURL:destURL error:&copyError]) {
                 compiled = destURL;
               } else {
                 compileError = copyError;
               }
             } else {
               compileError = err;
             }
             dispatch_semaphore_signal(sem);
           }];
  dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);
  *compileMs = (nowSeconds() - start) * 1000.0;

  if (!compiled && error) *error = compileError;
  return compiled;
}

static void dumpComputePlan(NSURL *compiledURL, MLModelConfiguration *config,
                            BenchConfig cfg) {
  if (![MLComputePlan respondsToSelector:@selector(loadContentsOfURL:
                                                        configuration:
                                                    completionHandler:)]) {
    printf("   Compute plan unavailable on this OS version (needs macOS 14.4+).\n");
    return;
  }

  dispatch_semaphore_t sem = dispatch_semaphore_create(0);
  __block MLComputePlan *plan = nil;
  __block NSError *planError = nil;
  [MLComputePlan loadContentsOfURL:compiledURL
                    configuration:config
                completionHandler:^(MLComputePlan *loaded, NSError *err) {
                  plan = loaded;
                  planError = err;
                  dispatch_semaphore_signal(sem);
                }];
  dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);

  if (!plan) {
    printf("   Failed to load compute plan: %s\n",
           planError.localizedDescription.UTF8String);
    return;
  }

  MLModelStructureProgram *program = plan.modelStructure.program;
  if (!program) {
    printf("   Model structure is not an ML program; cannot walk operations.\n");
    return;
  }

  printf("\n--- Compute Plan (%s) ---\n", cfg.unitsName.UTF8String);
  NSUInteger matmulTotal = 0, matmulOnTarget = 0;
  NSMutableDictionary<NSString *, NSNumber *> *byDevice = [NSMutableDictionary new];

  for (NSString *fnName in program.functions) {
    MLModelStructureProgramFunction *fn = program.functions[fnName];
    for (MLModelStructureProgramOperation *op in fn.block.operations) {
      if ([op.operatorName hasSuffix:@"const"]) continue;

      MLComputePlanDeviceUsage *usage = [plan computeDeviceUsageForMLProgramOperation:op];
      NSString *label = deviceLabel(usage.preferredComputeDevice);
      MLComputePlanCost *cost = [plan estimatedCostOfMLProgramOperation:op];

      byDevice[label] = @(byDevice[label].unsignedIntegerValue + 1);
      if ([op.operatorName hasSuffix:@"matmul"]) {
        matmulTotal++;
        if ([label isEqualToString:cfg.unitsName]) matmulOnTarget++;
      }

      if (cfg.verbose) {
        NSString *outName = op.outputs.firstObject.name ?: @"?";
        NSMutableArray *devNames = [NSMutableArray array];
        for (id dev in usage.supportedComputeDevices) {
          [devNames addObject:deviceLabel(dev)];
        }
        printf("   %-24s %-20s -> %-4s (supported: %s, cost weight %.4f)\n",
               outName.UTF8String, op.operatorName.UTF8String, label.UTF8String,
               [devNames componentsJoinedByString:@", "].UTF8String,
               cost ? cost.weight : 0.0);
      }
    }
  }

  for (NSString *label in [byDevice.allKeys sortedArrayUsingSelector:@selector(compare:)]) {
    printf("   ops on %-4s: %lu\n", label.UTF8String,
           (unsigned long)byDevice[label].unsignedIntegerValue);
  }
  printf("   matmul ops on %s: %lu/%lu%s\n", cfg.unitsName.UTF8String,
         (unsigned long)matmulOnTarget, (unsigned long)matmulTotal,
         (matmulTotal > 0 && matmulOnTarget == matmulTotal) ? "" : "  <-- not fully on target");
  printf("\n");
}

static void printUsage(const char *argv0) {
  printf("Usage: %s [options]\n\n", argv0);
  printf("CoreML matmul capacity benchmark. The MIL program is built in\n");
  printf("process at runtime, so every dimension below is just a flag.\n\n");
  printf("Options:\n");
  printf("  --units <target>   ane | gpu | cpu | all (default: ane)\n");
  printf("  --precision <mode> fp16 (default), int8 (W8A8 QDQ), or fp8 (Float8E4M3 QDQ)\n");
  printf("  --batch <B>        batch dimension (default: 1)\n");
  printf("  --size <N>         square M=K=N matmul dimension (default: 1024)\n");
  printf("  --layers <L>       number of chained matmul layers (default: 20)\n\n");
  printf("  --weights <mode>   dense (random-sign, non-cancelling) | repeat "
         "(tiled pattern matching the MPSGraph binaries) (default: dense)\n");
  printf("  --iterations <N>   timed prediction count (default: 20)\n");
  printf("  --warmup <N>       warmup prediction count (default: 3)\n");
  printf("  --plan             dump per-operation compute device placement\n");
  printf("  --check            print output samples to verify finite, non-zero results\n");
  printf("  --dump-mil         print the MIL program before running\n");
  printf("  --save <path>      write the generated spec as a .mlmodel\n");
  printf("  --verbose          per-operation detail in the compute plan\n");
  printf("  --help             this message\n");
}

static BOOL runBenchmark(BenchConfig cfg) {
  @autoreleasepool {
    NSError *error = nil;

    MILMatmulChainConfig spec = {
        .batch = cfg.B, .M = cfg.M, .K = cfg.K, .N = cfg.N, .layers = cfg.L,
        .weightMode = cfg.weightMode, .precision = cfg.precision,
    };
    if (cfg.showMIL) printf("\n%s\n", MILMatmulChainText(spec).UTF8String);

    NSData *specData = MILBuildMatmulChainSpec(spec, &error);
    if (!specData) {
      fprintf(stderr, "error: failed to build MIL spec: %s\n",
              error.localizedDescription.UTF8String);
      return NO;
    }
    if (cfg.verbose) {
      printf("Spec: %.1f KB (weights inline)\n", specData.length / 1024.0);
    }
    if (cfg.savePath) {
      if (![specData writeToFile:cfg.savePath options:0 error:&error]) {
        fprintf(stderr, "error: failed to write %s: %s\n",
                cfg.savePath.UTF8String, error.localizedDescription.UTF8String);
        return NO;
      }
      printf("Wrote %s\n", cfg.savePath.UTF8String);
    }
    if (cfg.iterations == 0 && !cfg.showPlan) {
      return YES;
    }

    NSString *tempPath = [NSTemporaryDirectory()
        stringByAppendingPathComponent:
            [NSString stringWithFormat:@"matmul_spec_%d.mlmodel", getpid()]];
    if (![specData writeToFile:tempPath options:0 error:&error]) {
      fprintf(stderr, "error: failed to write temporary model: %s\n",
              error.localizedDescription.UTF8String);
      return NO;
    }
    double prepMs = 0.0;
    NSURL *compiledURL = compileModel(tempPath, &prepMs, &error);
    [[NSFileManager defaultManager] removeItemAtPath:tempPath error:nil];
    if (!compiledURL) {
      fprintf(stderr, "error: failed to compile generated model: %s\n",
              error.localizedDescription.UTF8String);
      return NO;
    }

    MLModelConfiguration *config = [MLModelConfiguration new];
    config.computeUnits = cfg.units;
    if ([config respondsToSelector:@selector(optimizationHints)]) {
      MLOptimizationHints *hints = [MLOptimizationHints new];
      if ([hints respondsToSelector:@selector(setReshapeFrequency:)]) {
        hints.reshapeFrequency = MLReshapeFrequencyHintInfrequent;
      }
      if ([hints respondsToSelector:@selector(setSpecializationStrategy:)]) {
        hints.specializationStrategy = MLSpecializationStrategyFastPrediction;
      }
      config.optimizationHints = hints;
    }

    double loadStart = nowSeconds();
    MLModel *model = [MLModel modelWithContentsOfURL:compiledURL
                                        configuration:config
                                                error:&error];
    double loadMs = (nowSeconds() - loadStart) * 1000.0;
    if (!model) {
      fprintf(stderr, "error: failed to load model: %s\n",
              error.localizedDescription.UTF8String);
      return NO;
    }

    printf("Workload: [%lu, 1, %lu, %lu] x [%lu, 1, %lu, %lu] x %lu layers, "
           "%.2f GOPs/pass, %lu iterations\n",
           (unsigned long)cfg.B, (unsigned long)cfg.M, (unsigned long)cfg.K,
           (unsigned long)cfg.B, (unsigned long)cfg.K, (unsigned long)cfg.N,
           (unsigned long)cfg.L,
           2.0 * cfg.B * cfg.M * cfg.K * cfg.N * cfg.L / 1e9,
           (unsigned long)cfg.iterations);

    NSString *inputName = MILMatmulChainInputName();
    NSString *outputName = MILMatmulChainOutputName(spec);

    if (cfg.showPlan) dumpComputePlan(compiledURL, config, cfg);

    NSArray<NSNumber *> *shape = @[ @(cfg.B), @1, @(cfg.M), @(cfg.K) ];
    MLMultiArray *input = [[MLMultiArray alloc] initWithShape:shape
                                                    dataType:MLMultiArrayDataTypeFloat16
                                                       error:&error];
    if (!input) {
      fprintf(stderr, "error: failed to allocate input: %s\n",
              error.localizedDescription.UTF8String);
      return NO;
    }
    [input getMutableBytesWithHandler:^(void *bytes, NSInteger size,
                                        NSArray<NSNumber *> *strides) {
      fillDenseFloat16(bytes, (size_t)size);
    }];

    MLDictionaryFeatureProvider *features =
        [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{inputName : input}
                                                         error:&error];
    if (!features) {
      fprintf(stderr, "error: failed to build feature provider: %s\n",
              error.localizedDescription.UTF8String);
      return NO;
    }

    MLPredictionOptions *options = [MLPredictionOptions new];
    MLMultiArray *outputBacking = nil;
    if ([options respondsToSelector:@selector(setOutputBackings:)]) {
      MLMultiArrayConstraint *outConstraint =
          model.modelDescription.outputDescriptionsByName[outputName].multiArrayConstraint;
      outputBacking = [[MLMultiArray alloc] initWithShape:outConstraint.shape
                                                dataType:outConstraint.dataType
                                                   error:&error];
      if (outputBacking) {
        options.outputBackings = @{outputName : outputBacking};
      }
    }

    for (NSUInteger i = 0; i < cfg.warmup; i++) {
      if (![model predictionFromFeatures:features options:options error:&error]) {
        fprintf(stderr, "error: warmup prediction failed: %s\n",
                error.localizedDescription.UTF8String);
        return NO;
      }
    }

    id<MLFeatureProvider> result = nil;
    double start = nowSeconds();
    for (NSUInteger i = 0; i < cfg.iterations; i++) {
      result = [model predictionFromFeatures:features options:options error:&error];
      if (!result) {
        fprintf(stderr, "error: prediction failed: %s\n",
                error.localizedDescription.UTF8String);
        return NO;
      }
    }
    double duration = nowSeconds() - start;

    double avg = duration / cfg.iterations;
    double totalOps = 2.0 * cfg.B * cfg.M * cfg.K * cfg.N * cfg.L;
    double tops = (totalOps / 1e12) / avg;

    printf("[CoreML %s %s] Compile: %.2f ms, Load: %.2f ms, Avg: %.2f ms, "
           "Speed: %.4f TOPS (%.1f FPS)\n",
           cfg.unitsName.UTF8String, MILPrecisionName(cfg.precision).UTF8String,
           prepMs, loadMs, avg * 1000.0, tops, 1.0 / avg);

    if (cfg.checkOutput) {
      MLMultiArray *out = [result featureValueForName:outputName].multiArrayValue;
      if (!out) {
        printf("   Output check: no multi-array output named %s\n",
               outputName.UTF8String);
      } else {
        __block double minAbs = INFINITY, maxAbs = 0.0;
        __block NSUInteger nonFinite = 0, zeros = 0, total = 0;
        __block NSMutableArray<NSString *> *samples = [NSMutableArray array];
        [out getBytesWithHandler:^(const void *bytes, NSInteger size) {
          const _Float16 *p = (const _Float16 *)bytes;
          total = (NSUInteger)(size / sizeof(_Float16));
          for (NSUInteger i = 0; i < total; i++) {
            double v = (double)p[i];
            if (!isfinite(v)) {
              nonFinite++;
              continue;
            }
            if (v == 0.0) zeros++;
            double a = fabs(v);
            if (a < minAbs && a > 0.0) minAbs = a;
            if (a > maxAbs) maxAbs = a;
            if (i < 6) [samples addObject:[NSString stringWithFormat:@"%g", v]];
          }
        }];
        printf("   Output check: %lu elements, first: [%s]\n", (unsigned long)total,
               [samples componentsJoinedByString:@", "].UTF8String);
        printf("   Output check: |v| range [%g, %g], zeros: %lu, non-finite: %lu%s\n",
               isfinite(minAbs) ? minAbs : 0.0, maxAbs, (unsigned long)zeros,
               (unsigned long)nonFinite,
               nonFinite > 0 ? "  <-- NaN/Inf across the matmul chain" : "");
      }
    }

    return YES;
  }
}

int main(int argc, char *argv[]) {
  setlocale(LC_NUMERIC, "");

  BenchConfig cfg = {
      .B = 1,
      .M = 1024,
      .K = 1024,
      .N = 1024,
      .L = 20,
      .iterations = 20,
      .warmup = 3,
      .savePath = nil,
      .units = MLComputeUnitsCPUAndNeuralEngine,
      .unitsName = @"ANE",
      .weightMode = MILWeightModeDense,
      .precision = MILPrecisionFP16,
      .showPlan = NO,
      .showMIL = NO,
      .checkOutput = NO,
      .verbose = NO,
  };

  for (int i = 1; i < argc; i++) {
    NSString *arg = [NSString stringWithUTF8String:argv[i]];
    if ([arg isEqualToString:@"--precision"] && i + 1 < argc) {
      NSString *v = [[NSString stringWithUTF8String:argv[++i]] lowercaseString];
      if ([v isEqualToString:@"fp16"]) {
        cfg.precision = MILPrecisionFP16;
      } else if ([v isEqualToString:@"int8"]) {
        cfg.precision = MILPrecisionINT8;
      } else if ([v isEqualToString:@"fp8"]) {
        cfg.precision = MILPrecisionFP8;
      } else if ([v isEqualToString:@"int8w"]) {
        cfg.precision = MILPrecisionINT8WeightOnly;
      } else {
        fprintf(stderr, "error: unknown --precision value '%s' (expected fp16, int8, fp8, or int8w)\n", v.UTF8String);
        return 1;
      }
    } else if ([arg isEqualToString:@"--units"] && i + 1 < argc) {
      NSString *u = [[NSString stringWithUTF8String:argv[++i]] lowercaseString];
      if ([u isEqualToString:@"ane"]) {
        cfg.units = MLComputeUnitsCPUAndNeuralEngine;
        cfg.unitsName = @"ANE";
      } else if ([u isEqualToString:@"gpu"]) {
        cfg.units = MLComputeUnitsCPUAndGPU;
        cfg.unitsName = @"GPU";
      } else if ([u isEqualToString:@"cpu"]) {
        cfg.units = MLComputeUnitsCPUOnly;
        cfg.unitsName = @"CPU";
      } else if ([u isEqualToString:@"all"]) {
        cfg.units = MLComputeUnitsAll;
        cfg.unitsName = @"All";
      } else {
        fprintf(stderr, "error: unknown --units value '%s'\n", u.UTF8String);
        return 1;
      }
    } else if ([arg isEqualToString:@"--batch"] && i + 1 < argc) {
      cfg.B = atoi(argv[++i]);
    } else if ([arg isEqualToString:@"--size"] && i + 1 < argc) {
      cfg.M = cfg.K = cfg.N = atoi(argv[++i]);
    } else if ([arg isEqualToString:@"--layers"] && i + 1 < argc) {
      cfg.L = atoi(argv[++i]);
    } else if ([arg isEqualToString:@"--iterations"] && i + 1 < argc) {
      cfg.iterations = atoi(argv[++i]);
    } else if ([arg isEqualToString:@"--warmup"] && i + 1 < argc) {
      cfg.warmup = atoi(argv[++i]);
    } else if ([arg isEqualToString:@"--weights"] && i + 1 < argc) {
      NSString *v = [[NSString stringWithUTF8String:argv[++i]] lowercaseString];
      if ([v isEqualToString:@"dense"]) {
        cfg.weightMode = MILWeightModeDense;
      } else if ([v isEqualToString:@"repeat"]) {
        cfg.weightMode = MILWeightModeRepeat;
      } else {
        fprintf(stderr, "error: unknown --weights value '%s'\n", v.UTF8String);
        return 1;
      }
    } else if ([arg isEqualToString:@"--save"] && i + 1 < argc) {
      cfg.savePath = [NSString stringWithUTF8String:argv[++i]];
    } else if ([arg isEqualToString:@"--dump-mil"]) {
      cfg.showMIL = YES;
    } else if ([arg isEqualToString:@"--plan"]) {
      cfg.showPlan = YES;
    } else if ([arg isEqualToString:@"--check"]) {
      cfg.checkOutput = YES;
    } else if ([arg isEqualToString:@"--verbose"]) {
      cfg.verbose = YES;
    } else if ([arg isEqualToString:@"--help"] || [arg isEqualToString:@"-h"]) {
      printUsage(argv[0]);
      return 0;
    } else {
      fprintf(stderr, "error: unknown argument '%s'\n", arg.UTF8String);
      printUsage(argv[0]);
      return 1;
    }
  }

  if (cfg.iterations == 0 && !cfg.savePath && !cfg.showMIL) {
    fprintf(stderr, "error: --iterations must be at least 1\n");
    return 1;
  }

  printf("CoreML Matmul Capacity Benchmark (%s)\n", MILPrecisionName(cfg.precision).UTF8String);
  printf("Model: built at runtime by MILSpecBuilder (weights=%s)\n",
         MILWeightModeName(cfg.weightMode).UTF8String);

  return runBenchmark(cfg) ? 0 : 1;
}
