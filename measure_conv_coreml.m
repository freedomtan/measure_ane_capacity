// CoreML convolution capacity benchmark.
//
// Companion to measure_conv_universal.m, which measures the same workload
// (L chained KxK convolutions over [B, Ci, H, W], NCHW, SAME padding) through
// MPSGraph. This binary drives the model via CoreML instead, so the two paths
// can be compared on identical arithmetic.
//
// The MIL program is built at runtime by MILSpecBuilder, in process, using the
// CoreML::Specification classes generated from coremltools' own .proto schema.
// There is no Python step, no .mlpackage, and no coremlcompiler invocation: the
// serialized spec goes straight to MLModelAsset (macOS 13 / iOS 16). Every
// dimension is therefore a plain command-line flag, and the whole path ports to
// iOS, where coremltools cannot run.
//
// Unlike MPSGraph, which needs the private preferredDevice property and gives no
// confirmation of where the work landed, CoreML exposes MLComputePlan: --plan
// reports the preferred compute device for every operation in the program.

#import <locale.h>
#import <time.h>

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

#import "MILSpecBuilder.h"

typedef struct {
  NSUInteger B, H, W, Ci, Co, K, L;
  NSUInteger iterations;
  NSUInteger warmup;
  // When set, benchmark this existing .mlpackage/.mlmodelc instead of building a
  // spec. Lets a coremltools-produced model be cross-checked against ours.
  NSString *modelPath;
  NSString *savePath;
  MLComputeUnits units;
  NSString *unitsName;
  MILWeightMode weightMode;
  MILPrecision precision;
  BOOL showPlan;
  BOOL showMIL;
  BOOL checkOutput;
  BOOL verbose;
  BOOL denseInput;
  // Which dimensions the user set explicitly. With --model, anything left unset
  // is taken from the model's own metadata.
  BOOL setB, setSize, setChannels, setKernel, setLayers;
} BenchConfig;

// Fill buffer with the tiled non-zero pattern used by measure_conv_universal.m,
// so CoreML and MPSGraph runs see bit-for-bit the same input.
static void fillRepeatFloat16(void *buffer, size_t byteCount) {
  if (!buffer || byteCount == 0) return;
  uint16_t *p = (uint16_t *)buffer;
  size_t count = byteCount / sizeof(uint16_t);
  static const uint16_t fp16_pattern[4] = {0x2C00, 0xAC00, 0x2800, 0xA800};
  for (size_t i = 0; i < count; i++) {
    p[i] = fp16_pattern[i % 4];
  }
}

// Random-sign +/-0.0625 (fp16 0x2C00 / 0xAC00). The tiled pattern above is
// structured enough to cancel exactly under the convolution reduction, which
// hands every layer after the first an all-zero tensor -- precisely what H17+
// zero-skipping detects. Random signs keep the reduction non-degenerate.
// Deterministic (fixed-seed xorshift64) so runs are reproducible.
static void fillDenseFloat16(void *buffer, size_t byteCount) {
  if (!buffer || byteCount == 0) return;
  uint16_t *p = (uint16_t *)buffer;
  size_t count = byteCount / sizeof(uint16_t);
  uint64_t state = 0x9E3779B97F4A7C15ULL;
  for (size_t i = 0; i < count; i++) {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    // Sign bit only: magnitude stays constant at 0.0625.
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

// Compile a .mlpackage to .mlmodelc. Already-compiled inputs pass through.
static NSURL *compileModel(NSString *path, double *compileMs, NSError **error) {
  NSURL *url = [NSURL fileURLWithPath:path];
  if ([path hasSuffix:@".mlmodelc"]) {
    *compileMs = 0.0;
    return url;
  }

  double start = nowSeconds();
  dispatch_semaphore_t sem = dispatch_semaphore_create(0);
  __block NSURL *compiled = nil;
  __block NSError *compileError = nil;
  [MLModel compileModelAtURL:url
           completionHandler:^(NSURL *compiledModelURL, NSError *err) {
             // The compiled model lives in a temporary location that is torn
             // down once this handler returns, so copy it somewhere stable.
             if (compiledModelURL) {
               NSString *dest = [NSTemporaryDirectory()
                   stringByAppendingPathComponent:
                       [NSString stringWithFormat:@"measure_conv_coreml_%@",
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

// The generator names tensors from the MIL graph (input "x", output "conv_<L-1>"),
// so read the names off the model rather than hardcoding them.
static NSString *soleInputName(MLModel *model) {
  return model.modelDescription.inputDescriptionsByName.allKeys.firstObject;
}

static NSString *soleOutputName(MLModel *model) {
  return model.modelDescription.outputDescriptionsByName.allKeys.firstObject;
}

// Adopt the workload dimensions stamped into the model by tools/gen_conv_mil.py.
// K and L cannot be recovered from the input shape, so without this a stale
// --kernel or --layers would silently scale the reported TOPS. Flags the user set
// explicitly are honoured but must agree with the model.
static BOOL resolveDimensions(MLModel *model, BenchConfig *cfg, BOOL *fromMetadata) {
  NSDictionary *meta =
      model.modelDescription.metadata[MLModelCreatorDefinedKey];
  *fromMetadata = NO;
  if (![meta isKindOfClass:[NSDictionary class]] || !meta[@"workload.layers"]) {
    return YES;  // Foreign model: fall back to flags plus the shape check below.
  }
  *fromMetadata = YES;

  struct {
    NSString *key;
    NSUInteger *slot;
    BOOL wasSet;
    const char *flag;
  } fields[] = {
      {@"workload.batch", &cfg->B, cfg->setB, "--batch"},
      {@"workload.channels_in", &cfg->Ci, cfg->setChannels, "--channels"},
      {@"workload.height", &cfg->H, cfg->setSize, "--size"},
      {@"workload.width", &cfg->W, cfg->setSize, "--size"},
      {@"workload.kernel", &cfg->K, cfg->setKernel, "--kernel"},
      {@"workload.layers", &cfg->L, cfg->setLayers, "--layers"},
  };

  for (size_t i = 0; i < sizeof(fields) / sizeof(fields[0]); i++) {
    NSString *raw = meta[fields[i].key];
    if (!raw) continue;
    NSUInteger value = (NSUInteger)raw.integerValue;
    if (fields[i].wasSet && *fields[i].slot != value) {
      fprintf(stderr,
              "error: %s=%lu disagrees with the model's own %s=%lu.\n"
              "       Omit the flag to use the model's value, or generate a "
              "matching model.\n",
              fields[i].flag, (unsigned long)*fields[i].slot,
              fields[i].key.UTF8String, (unsigned long)value);
      return NO;
    }
    *fields[i].slot = value;
  }

  NSString *co = meta[@"workload.channels_out"];
  cfg->Co = co ? (NSUInteger)co.integerValue : cfg->Ci;

  NSString *prec = meta[@"workload.precision"];
  if (prec) {
    if ([prec isEqualToString:@"INT8"]) {
      cfg->precision = MILPrecisionINT8;
    } else {
      cfg->precision = MILPrecisionFP16;
    }
  }
  return YES;
}

// A dimension mismatch between the model and the --size/--channels/--layers
// flags would yield a plausible-looking but wrong TOPS figure, so refuse to run.
static BOOL validateShape(MLModel *model, BenchConfig cfg) {
  MLFeatureDescription *desc =
      model.modelDescription.inputDescriptionsByName[soleInputName(model)];
  NSArray<NSNumber *> *shape = desc.multiArrayConstraint.shape;
  NSArray<NSNumber *> *expected = @[ @(cfg.B), @(cfg.Ci), @(cfg.H), @(cfg.W) ];
  if (![shape isEqualToArray:expected]) {
    // NSArray's own description is multi-line; keep this on one readable line.
    NSString *got = [shape componentsJoinedByString:@", "];
    NSString *want = [expected componentsJoinedByString:@", "];
    fprintf(stderr,
            "error: model input shape [%s] does not match requested dimensions "
            "[%s].\n"
            "       Drop --model to build a matching model instead.\n",
            got.UTF8String, want.UTF8String);
    return NO;
  }
  if (desc.multiArrayConstraint.dataType != MLMultiArrayDataTypeFloat16) {
    fprintf(stderr,
            "error: model input is not Float16; an FP32 boundary would add a "
            "per-prediction CPU cast and invalidate the measurement.\n");
    return NO;
  }
  return YES;
}

// Report the preferred compute device per operation. This is what MPSGraph
// cannot tell us: proof the convolutions actually ran on the ANE.
//
// Pass either the in-memory asset or a compiled .mlmodelc URL. Handed a
// .mlpackage URL, MLComputePlan throws an uncatchable C++ ios_base::failure
// rather than returning an error, so the caller must compile first.
static void dumpComputePlan(MLModelAsset *asset, NSURL *compiledURL,
                            MLModelConfiguration *config, BenchConfig cfg) {
  if (![MLComputePlan respondsToSelector:@selector(loadContentsOfURL:
                                                        configuration:
                                                    completionHandler:)]) {
    printf("   Compute plan unavailable on this OS version (needs macOS 14.4+).\n");
    return;
  }

  dispatch_semaphore_t sem = dispatch_semaphore_create(0);
  __block MLComputePlan *plan = nil;
  __block NSError *planError = nil;
  void (^handler)(MLComputePlan *, NSError *) =
      ^(MLComputePlan *loaded, NSError *err) {
        plan = loaded;
        planError = err;
        dispatch_semaphore_signal(sem);
      };
  if (asset) {
    [MLComputePlan loadModelAsset:asset
                    configuration:config
                completionHandler:handler];
  } else {
    [MLComputePlan loadContentsOfURL:compiledURL
                      configuration:config
                  completionHandler:handler];
  }
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
  NSUInteger convTotal = 0, convOnANE = 0;
  NSMutableDictionary<NSString *, NSNumber *> *byDevice = [NSMutableDictionary new];

  for (NSString *fnName in program.functions) {
    MLModelStructureProgramFunction *fn = program.functions[fnName];
    for (MLModelStructureProgramOperation *op in fn.block.operations) {
      // const ops carry no compute and would drown out the real work.
      // Names may be opset-qualified ("ios18.const") or bare.
      if ([op.operatorName hasSuffix:@"const"]) continue;

      MLComputePlanDeviceUsage *usage = [plan computeDeviceUsageForMLProgramOperation:op];
      NSString *label = deviceLabel(usage.preferredComputeDevice);
      MLComputePlanCost *cost = [plan estimatedCostOfMLProgramOperation:op];

      byDevice[label] = @(byDevice[label].unsignedIntegerValue + 1);
      // Compiled programs report opset-qualified names, e.g. "ios18.conv".
      if ([op.operatorName hasSuffix:@"conv"]) {
        convTotal++;
        if ([label isEqualToString:@"ANE"]) convOnANE++;
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
  printf("   conv ops on ANE: %lu/%lu%s\n", (unsigned long)convOnANE,
         (unsigned long)convTotal,
         (convTotal > 0 && convOnANE == convTotal) ? "" : "  <-- not fully on ANE");
  printf("\n");
}

static void printUsage(const char *argv0) {
  printf("Usage: %s [options]\n\n", argv0);
  printf("CoreML convolution capacity benchmark. The MIL program is built in\n");
  printf("process at runtime, so every dimension below is just a flag.\n\n");
  printf("Options:\n");
  printf("  --units <target>   ane | gpu | cpu | all (default: ane)\n");
  printf("  --precision <mode> fp16 (default) or int8 (W8A8 simulated QDQ)\n");
  printf("  --batch <B>        batch dimension (default: 1)\n");
  printf("  --size <H>         spatial height and width (default: 256)\n");
  printf("  --channels <C>     input and output channels (default: 128)\n");
  printf("  --kernel <K>       kernel size (default: 3)\n");
  printf("  --layers <L>       number of chained conv layers (default: 20)\n\n");
  printf("  --weights <mode>   dense (random-sign, non-cancelling) | repeat "
         "(tiled pattern matching the MPSGraph binaries) (default: dense)\n");
  printf("  --input <mode>     same choices, for the input tensor "
         "(default: dense)\n");
  printf("  --iterations <N>   timed prediction count (default: 20)\n");
  printf("  --warmup <N>       warmup prediction count (default: 3)\n");
  printf("  --sweep <axis>     channels | spatial | depth | kernel: measure "
         "every point along an axis\n");
  printf("  --sweep-values <a,b,c>  override the default sweep points\n");
  printf("  --plan             dump per-operation compute device placement\n");
  printf("  --check            print output samples to verify finite, non-zero results\n");
  printf("  --dump-mil         print the MIL program before running\n");
  printf("  --save <path>      write the generated spec as a .mlmodel\n");
  printf("  --model <path>     benchmark an existing .mlpackage/.mlmodelc "
         "instead of building one; dimensions come from its metadata\n");
  printf("  --verbose          per-operation detail in the compute plan\n");
  printf("  --help             this message\n");
}

static BOOL runBenchmark(BenchConfig cfg) {
  @autoreleasepool {
    NSError *error = nil;
    // "Build" is spec construction for a generated model, "Compile" is
    // coremlcompiler for an external one. They are not comparable, so the label
    // says which happened.
    const char *prepLabel = "Build";
    double prepMs = 0.0;
    MLModelAsset *asset = nil;
    NSURL *compiledURL = nil;

    if (cfg.modelPath) {
      prepLabel = "Compile";
      if (![[NSFileManager defaultManager] fileExistsAtPath:cfg.modelPath]) {
        fprintf(stderr, "error: model not found at %s\n",
                cfg.modelPath.UTF8String);
        return NO;
      }
      compiledURL = compileModel(cfg.modelPath, &prepMs, &error);
      if (!compiledURL) {
        fprintf(stderr, "error: failed to compile model: %s\n",
                error.localizedDescription.UTF8String);
        return NO;
      }
    } else {
      MILConvChainConfig spec = {
          .batch = cfg.B,
          .channelsIn = cfg.Ci,
          .channelsOut = cfg.Co,
          .height = cfg.H,
          .width = cfg.W,
          .kernel = cfg.K,
          .layers = cfg.L,
          .weightMode = cfg.weightMode,
          .precision = cfg.precision,
      };
      if (cfg.showMIL) printf("\n%s\n", MILConvChainText(spec).UTF8String);

      double buildStart = nowSeconds();
      NSData *specData = MILBuildConvChainSpec(spec, &error);
      if (!specData) {
        fprintf(stderr, "error: failed to build MIL spec: %s\n",
                error.localizedDescription.UTF8String);
        return NO;
      }
      if (cfg.precision == MILPrecisionINT8) {
        NSString *tempPath = [NSTemporaryDirectory()
            stringByAppendingPathComponent:[NSString stringWithFormat:@"spec_int8_%d.mlmodel", getpid()]];
        if (![specData writeToFile:tempPath options:0 error:&error]) {
          fprintf(stderr, "error: failed to write temporary model: %s\n",
                  error.localizedDescription.UTF8String);
          return NO;
        }
        prepLabel = "Compile";
        compiledURL = compileModel(tempPath, &prepMs, &error);
        [[NSFileManager defaultManager] removeItemAtPath:tempPath error:nil];
        if (!compiledURL) {
          fprintf(stderr, "error: failed to compile INT8 model: %s\n",
                  error.localizedDescription.UTF8String);
          return NO;
        }
      } else {
        // No file ever hits disk for FP16: the serialized spec goes straight into CoreML.
        asset = [MLModelAsset modelAssetWithSpecificationData:specData
                                                       error:&error];
        prepMs = (nowSeconds() - buildStart) * 1000.0;
        if (!asset) {
          fprintf(stderr, "error: CoreML rejected the generated spec: %s\n",
                  error.localizedDescription.UTF8String);
          return NO;
        }
      }
      if (cfg.verbose) {
        printf("Spec: %.1f KB (weights inline)\n", specData.length / 1024.0);
      }
      if (cfg.savePath && ![specData writeToFile:cfg.savePath options:0
                                           error:&error]) {
        fprintf(stderr, "error: failed to write %s: %s\n",
                cfg.savePath.UTF8String, error.localizedDescription.UTF8String);
        return NO;
      } else if (cfg.savePath) {
        printf("Wrote %s\n", cfg.savePath.UTF8String);
      }
    }

    MLModelConfiguration *config = [MLModelConfiguration new];
    config.computeUnits = cfg.units;
    // Guarded so this source still builds and runs against older SDKs/OSes,
    // matching how the MPSGraph benchmarks probe for private API.
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

    // Model load is where the ANE program is compiled and cached by the OS, so
    // it is timed separately from inference.
    double loadStart = nowSeconds();
    MLModel *model = nil;
    if (asset) {
      dispatch_semaphore_t sem = dispatch_semaphore_create(0);
      __block MLModel *loaded = nil;
      __block NSError *loadError = nil;
      [MLModel loadModelAsset:asset
                configuration:config
            completionHandler:^(MLModel *m, NSError *err) {
              loaded = m;
              loadError = err;
              dispatch_semaphore_signal(sem);
            }];
      dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);
      model = loaded;
      error = loadError;
    } else {
      model = [MLModel modelWithContentsOfURL:compiledURL
                               configuration:config
                                       error:&error];
    }
    double loadMs = (nowSeconds() - loadStart) * 1000.0;
    if (!model) {
      fprintf(stderr, "error: failed to load model: %s\n",
              error.localizedDescription.UTF8String);
      return NO;
    }

    BOOL fromMetadata = NO;
    // Only an external model can disagree with the flags; for a generated one
    // the flags are what built it. validateShape still runs either way -- it is
    // a genuine check that the spec we emitted has the shape we intended.
    if (cfg.modelPath && !resolveDimensions(model, &cfg, &fromMetadata)) return NO;
    if (!validateShape(model, cfg)) return NO;

    printf("Workload: [%lu, %lu, %lu, %lu] x %lu conv%lux%lu (Co=%lu), "
           "%.2f GOPs/pass, %lu iterations%s\n",
           (unsigned long)cfg.B, (unsigned long)cfg.Ci, (unsigned long)cfg.H,
           (unsigned long)cfg.W, (unsigned long)cfg.L, (unsigned long)cfg.K,
           (unsigned long)cfg.K, (unsigned long)cfg.Co,
           2.0 * cfg.B * cfg.H * cfg.W * cfg.Ci * cfg.Co * cfg.K * cfg.K * cfg.L / 1e9,
           (unsigned long)cfg.iterations,
           fromMetadata ? " (dimensions from model metadata)" : "");

    NSString *inputName = soleInputName(model);
    NSString *outputName = soleOutputName(model);

    if (cfg.showPlan) dumpComputePlan(asset, compiledURL, config, cfg);

    NSArray<NSNumber *> *shape = @[ @(cfg.B), @(cfg.Ci), @(cfg.H), @(cfg.W) ];
    MLMultiArray *input = [[MLMultiArray alloc] initWithShape:shape
                                                    dataType:MLMultiArrayDataTypeFloat16
                                                       error:&error];
    if (!input) {
      fprintf(stderr, "error: failed to allocate input: %s\n",
              error.localizedDescription.UTF8String);
      return NO;
    }
    BOOL dense = cfg.denseInput;
    [input getMutableBytesWithHandler:^(void *bytes, NSInteger size,
                                        NSArray<NSNumber *> *strides) {
      if (dense) {
        fillDenseFloat16(bytes, (size_t)size);
      } else {
        fillRepeatFloat16(bytes, (size_t)size);
      }
    }];

    MLDictionaryFeatureProvider *features =
        [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{inputName : input}
                                                         error:&error];
    if (!features) {
      fprintf(stderr, "error: failed to build feature provider: %s\n",
              error.localizedDescription.UTF8String);
      return NO;
    }

    // Reuse one output buffer across predictions; otherwise every call allocates
    // a fresh multi-megabyte MLMultiArray and we end up timing malloc.
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
    double totalOps =
        2.0 * cfg.B * cfg.H * cfg.W * cfg.Ci * cfg.Co * cfg.K * cfg.K * cfg.L;
    double tops = (totalOps / 1e12) / avg;

    printf("[CoreML %s %s] %s: %.2f ms, Load: %.2f ms, Avg: %.2f ms, "
           "Speed: %.4f TOPS (%.1f FPS)\n",
           cfg.unitsName.UTF8String, MILPrecisionName(cfg.precision).UTF8String,
           prepLabel, prepMs, loadMs, avg * 1000.0,
           tops, 1.0 / avg);

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
               nonFinite > 0 ? "  <-- NaN/Inf across the conv chain" : "");
      }
    }

    return YES;
  }
}

// Sweep axes mirror SweepType in ANECapacityApp/BenchmarkModels.swift. Because
// models are now built in process, a sweep measures every point in one run
// instead of pre-generating a directory of packages.
static BOOL runSweep(BenchConfig cfg, NSString *axis, NSString *valuesArg) {
  NSDictionary<NSString *, NSArray<NSNumber *> *> *defaults = @{
    @"channels" : @[ @32, @64, @128, @192, @256 ],
    @"spatial" : @[ @64, @128, @192, @256 ],
    @"depth" : @[ @1, @5, @10, @20, @50 ],
    @"kernel" : @[ @1, @3, @5, @7 ],
  };
  NSArray<NSNumber *> *values = defaults[axis];
  if (!values) {
    fprintf(stderr,
            "error: unknown --sweep axis '%s' (expected channels, spatial, "
            "depth or kernel)\n",
            axis.UTF8String);
    return NO;
  }
  if (valuesArg) {
    NSMutableArray<NSNumber *> *parsed = [NSMutableArray array];
    for (NSString *part in [valuesArg componentsSeparatedByString:@","]) {
      NSInteger v = part.integerValue;
      if (v <= 0) {
        fprintf(stderr, "error: bad --sweep-values entry '%s'\n",
                part.UTF8String);
        return NO;
      }
      [parsed addObject:@(v)];
    }
    values = parsed;
  }

  printf("\nSweeping %s over [%s]\n", axis.UTF8String,
         [values componentsJoinedByString:@", "].UTF8String);

  BOOL allOK = YES;
  for (NSNumber *value in values) {
    NSUInteger v = value.unsignedIntegerValue;
    BenchConfig point = cfg;
    if ([axis isEqualToString:@"channels"]) {
      point.Ci = point.Co = v;
    } else if ([axis isEqualToString:@"spatial"]) {
      point.H = point.W = v;
    } else if ([axis isEqualToString:@"depth"]) {
      point.L = v;
    } else {
      point.K = v;
    }
    printf("\n");
    if (!runBenchmark(point)) allOK = NO;
  }
  return allOK;
}

int main(int argc, char *argv[]) {
  setlocale(LC_NUMERIC, "");

  BenchConfig cfg = {
      .B = 1,
      .H = 256,
      .W = 256,
      .Ci = 128,
      .Co = 128,
      .K = 3,
      .L = 20,
      .iterations = 20,
      .warmup = 3,
      .modelPath = nil,
      .savePath = nil,
      .units = MLComputeUnitsCPUAndNeuralEngine,
      .unitsName = @"ANE",
      .weightMode = MILWeightModeDense,
      .precision = MILPrecisionFP16,
      .showPlan = NO,
      .showMIL = NO,
      .checkOutput = NO,
      .verbose = NO,
      .denseInput = YES,
  };
  NSString *sweepAxis = nil;
  NSString *sweepValues = nil;

  for (int i = 1; i < argc; i++) {
    NSString *arg = [NSString stringWithUTF8String:argv[i]];
    if ([arg isEqualToString:@"--model"] && i + 1 < argc) {
      cfg.modelPath = [NSString stringWithUTF8String:argv[++i]];
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
    } else if ([arg isEqualToString:@"--precision"] && i + 1 < argc) {
      NSString *v = [[NSString stringWithUTF8String:argv[++i]] lowercaseString];
      if ([v isEqualToString:@"fp16"]) {
        cfg.precision = MILPrecisionFP16;
      } else if ([v isEqualToString:@"int8"]) {
        cfg.precision = MILPrecisionINT8;
      } else {
        fprintf(stderr, "error: unknown --precision value '%s' (expected fp16 or int8)\n", v.UTF8String);
        return 1;
      }
    } else if ([arg isEqualToString:@"--batch"] && i + 1 < argc) {
      cfg.B = atoi(argv[++i]);
      cfg.setB = YES;
    } else if ([arg isEqualToString:@"--size"] && i + 1 < argc) {
      cfg.H = cfg.W = atoi(argv[++i]);
      cfg.setSize = YES;
    } else if ([arg isEqualToString:@"--channels"] && i + 1 < argc) {
      cfg.Ci = cfg.Co = atoi(argv[++i]);
      cfg.setChannels = YES;
    } else if ([arg isEqualToString:@"--kernel"] && i + 1 < argc) {
      cfg.K = atoi(argv[++i]);
      cfg.setKernel = YES;
    } else if ([arg isEqualToString:@"--layers"] && i + 1 < argc) {
      cfg.L = atoi(argv[++i]);
      cfg.setLayers = YES;
    } else if ([arg isEqualToString:@"--iterations"] && i + 1 < argc) {
      cfg.iterations = atoi(argv[++i]);
    } else if ([arg isEqualToString:@"--warmup"] && i + 1 < argc) {
      cfg.warmup = atoi(argv[++i]);
    } else if ([arg isEqualToString:@"--input"] && i + 1 < argc) {
      NSString *v = [[NSString stringWithUTF8String:argv[++i]] lowercaseString];
      if ([v isEqualToString:@"dense"]) {
        cfg.denseInput = YES;
      } else if ([v isEqualToString:@"repeat"]) {
        cfg.denseInput = NO;
      } else {
        fprintf(stderr, "error: unknown --input value '%s'\n", v.UTF8String);
        return 1;
      }
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
    } else if ([arg isEqualToString:@"--sweep"] && i + 1 < argc) {
      sweepAxis = [[NSString stringWithUTF8String:argv[++i]] lowercaseString];
    } else if ([arg isEqualToString:@"--sweep-values"] && i + 1 < argc) {
      sweepValues = [NSString stringWithUTF8String:argv[++i]];
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

  if (cfg.iterations == 0) {
    fprintf(stderr, "error: --iterations must be at least 1\n");
    return 1;
  }
  if (sweepAxis && cfg.modelPath) {
    fprintf(stderr, "error: --sweep builds its own models; drop --model\n");
    return 1;
  }

  printf("CoreML Conv2D Capacity Benchmark (%s)\n",
         MILPrecisionName(cfg.precision).UTF8String);
  if (cfg.modelPath) {
    printf("Model: %s\n", cfg.modelPath.UTF8String);
  } else {
    printf("Model: built at runtime by MILSpecBuilder (weights=%s)\n",
           MILWeightModeName(cfg.weightMode).UTF8String);
  }
  printf("Input: %s\n", cfg.denseInput ? "dense (random-sign)" : "repeat (tiled)");

  if (!sweepAxis) return runBenchmark(cfg) ? 0 : 1;
  return runSweep(cfg, sweepAxis, sweepValues) ? 0 : 1;
}
