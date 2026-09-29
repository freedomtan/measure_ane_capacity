//
//  MILCapacityEngineBridge.mm
//  ANECapacityWithMILApp
//
//  Implementation of CoreML and _ANEClient bridge for MIL capacity measurements.
//

#import "MILCapacityEngineBridge.h"
#import <time.h>
#import <IOSurface/IOSurfaceRef.h>

#define kANEFModelTypeKey            @"kANEFModelType"
#define kANEFModelMILValue           @"kANEFModelMIL"
#define kANEFPerformanceStatsMaskKey @"kANEFPerformanceStatsMask"

__attribute__((objc_runtime_visible))
@interface _ANEIOSurfaceObject : NSObject
+ (instancetype)objectWithIOSurface:(IOSurfaceRef)surface;
@end

__attribute__((objc_runtime_visible))
@interface _ANEPerformanceStatsIOSurface : NSObject
+ (instancetype)objectWithIOSurface:(_ANEIOSurfaceObject *)ioSurface statType:(int)statType;
@end

__attribute__((objc_runtime_visible))
@interface _ANEPerformanceStats : NSObject
@property (nonatomic, readonly) NSData *perfCounterData;
- (NSString *)stringForPerfCounter:(int)index;
@end

__attribute__((objc_runtime_visible))
@interface _ANERequest : NSObject
+ (instancetype)requestWithInputs:(NSArray *)inputs
                     inputIndices:(NSArray *)inputIndices
                          outputs:(NSArray *)outputs
                    outputIndices:(NSArray *)outputIndices
                        perfStats:(NSArray *)perfStats
                   procedureIndex:(NSNumber *)procedureIndex;
@property (nonatomic, readonly) _ANEPerformanceStats *perfStats;
@end

__attribute__((objc_runtime_visible))
@interface _ANEModel : NSObject
+ (instancetype)modelAtURL:(NSURL *)url key:(nullable NSString *)key;
@end

__attribute__((objc_runtime_visible))
@interface _ANEClient : NSObject
+ (instancetype)sharedConnection;
- (BOOL)compileModel:(_ANEModel *)model options:(NSDictionary *)options qos:(unsigned int)qos error:(NSError **)error;
- (BOOL)loadModel:(_ANEModel *)model options:(NSDictionary *)options qos:(unsigned int)qos error:(NSError **)error;
- (BOOL)unloadModel:(_ANEModel *)model options:(NSDictionary *)options qos:(unsigned int)qos error:(NSError **)error;
- (BOOL)evaluateWithModel:(_ANEModel *)model options:(NSDictionary *)options request:(_ANERequest *)request qos:(unsigned int)qos error:(NSError **)error;
@end

@implementation MILBenchmarkExecutionResult
@end

@implementation MILCapacityEngineBridge

static double nowSeconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

static void fillDenseFloat16(void *buffer, size_t count) {
    if (!buffer || count == 0) return;
    uint16_t *p = (uint16_t *)buffer;
    uint64_t state = 0x9E3779B97F4A7C15ULL;
    for (size_t i = 0; i < count; i++) {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        // Non-canceling +/- 0.03125 (0x2800 / 0xA800)
        p[i] = (state & 1) ? 0xA800 : 0x2800;
    }
}

+ (MILBenchmarkExecutionResult *)compileAndEvaluateSpecData:(NSData *)specData
                                                      batch:(NSUInteger)B
                                                   channels:(NSUInteger)C
                                                     height:(NSUInteger)H
                                                      width:(NSUInteger)W
                                                     kernel:(NSUInteger)K
                                                     layers:(NSUInteger)L
                                                  precision:(NSString *)precision
                                               computeUnits:(MLComputeUnits)units
                                                 iterations:(NSUInteger)iterations
                                                     warmup:(NSUInteger)warmup
                                                     usePMU:(BOOL)usePMU
                                            progressHandler:(nullable void (^)(NSString *log))progress {
    MILBenchmarkExecutionResult *res = [MILBenchmarkExecutionResult new];
    NSError *error = nil;

    NSString *tempPath = [NSTemporaryDirectory() stringByAppendingPathComponent:[NSString stringWithFormat:@"model_%d.mlmodel", getpid()]];
    if (![specData writeToFile:tempPath options:0 error:&error]) {
        res.success = NO;
        res.statusMessage = [NSString stringWithFormat:@"Failed to write temp spec: %@", error.localizedDescription];
        return res;
    }

    if (progress) progress([NSString stringWithFormat:@"Compiling ML Program spec on device..."]);
    double startCompile = nowSeconds();

    dispatch_semaphore_t sem = dispatch_semaphore_create(0);
    __block NSURL *compiledURL = nil;
    __block NSError *compileErr = nil;

    [MLModel compileModelAtURL:[NSURL fileURLWithPath:tempPath] completionHandler:^(NSURL *url, NSError *err) {
        if (url) {
            NSString *dest = [NSTemporaryDirectory() stringByAppendingPathComponent:[NSString stringWithFormat:@"compiled_%@", url.lastPathComponent]];
            NSURL *destURL = [NSURL fileURLWithPath:dest];
            [[NSFileManager defaultManager] removeItemAtURL:destURL error:nil];
            if ([[NSFileManager defaultManager] copyItemAtURL:url toURL:destURL error:nil]) {
                compiledURL = destURL;
            } else {
                compiledURL = url;
            }
        } else {
            compileErr = err;
        }
        dispatch_semaphore_signal(sem);
    }];
    dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);
    [[NSFileManager defaultManager] removeItemAtPath:tempPath error:nil];

    if (!compiledURL) {
        res.success = NO;
        res.statusMessage = [NSString stringWithFormat:@"CoreML compilation failed: %@", compileErr.localizedDescription];
        return res;
    }

    res.compileTimeMs = (nowSeconds() - startCompile) * 1000.0;
    return [self evaluateModelAtURL:compiledURL batch:B channels:C height:H width:W kernel:K layers:L precision:precision computeUnits:units iterations:iterations warmup:warmup usePMU:usePMU progressHandler:progress];
}

+ (MILBenchmarkExecutionResult *)evaluateModelAtURL:(NSURL *)compiledModelURL
                                              batch:(NSUInteger)B
                                           channels:(NSUInteger)C
                                             height:(NSUInteger)H
                                              width:(NSUInteger)W
                                             kernel:(NSUInteger)K
                                             layers:(NSUInteger)L
                                          precision:(NSString *)precision
                                       computeUnits:(MLComputeUnits)units
                                         iterations:(NSUInteger)iterations
                                             warmup:(NSUInteger)warmup
                                             usePMU:(BOOL)usePMU
                                    progressHandler:(nullable void (^)(NSString *log))progress {
    MILBenchmarkExecutionResult *res = [MILBenchmarkExecutionResult new];
    NSError *error = nil;

    double totalOps = 2.0 * B * H * W * C * C * K * K * L;
    res.totalGOPs = totalOps / 1e9;

    MLModelConfiguration *config = [MLModelConfiguration new];
    config.computeUnits = units;
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

    if (progress) progress([NSString stringWithFormat:@"Loading CoreML model into memory..."]);
    double loadStart = nowSeconds();
    MLModel *model = [MLModel modelWithContentsOfURL:compiledModelURL configuration:config error:&error];
    res.loadTimeMs = (nowSeconds() - loadStart) * 1000.0;

    if (!model) {
        res.success = NO;
        res.statusMessage = [NSString stringWithFormat:@"Failed to load model: %@", error.localizedDescription];
        return res;
    }

    NSString *inName = @"x";
    NSString *outName = [NSString stringWithFormat:@"conv_%lu", (unsigned long)(L - 1)];
    if (model.modelDescription.inputDescriptionsByName.count > 0) {
        inName = model.modelDescription.inputDescriptionsByName.allKeys.firstObject;
    }
    if (model.modelDescription.outputDescriptionsByName.count > 0) {
        outName = model.modelDescription.outputDescriptionsByName.allKeys.firstObject;
    }

    // Allocate & initialize non-canceling input tensor
    MLMultiArray *inArray = [[MLMultiArray alloc] initWithShape:@[@(B), @(C), @(H), @(W)] dataType:MLMultiArrayDataTypeFloat16 error:&error];
    if (!inArray) {
        res.success = NO;
        res.statusMessage = [NSString stringWithFormat:@"Failed to allocate input MLMultiArray: %@", error.localizedDescription];
        return res;
    }
    fillDenseFloat16(inArray.dataPointer, B * C * H * W);

    id<MLFeatureProvider> inputFeatures = [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{inName: inArray} error:&error];

    // Warmup
    if (progress) progress([NSString stringWithFormat:@"Warming up (%lu iterations)...", (unsigned long)warmup]);
    for (NSUInteger w = 0; w < warmup; w++) {
        @autoreleasepool {
            [model predictionFromFeatures:inputFeatures error:nil];
        }
    }

    // Timed Predictions
    if (progress) progress([NSString stringWithFormat:@"Running %lu timed prediction iterations...", (unsigned long)iterations]);
    NSMutableArray<NSNumber *> *latencies = [NSMutableArray arrayWithCapacity:iterations];
    double totalSec = 0.0;
    id<MLFeatureProvider> lastResult = nil;

    for (NSUInteger iter = 0; iter < iterations; iter++) {
        @autoreleasepool {
            double t0 = nowSeconds();
            id<MLFeatureProvider> pred = [model predictionFromFeatures:inputFeatures error:&error];
            double dt = (nowSeconds() - t0) * 1000.0;
            if (pred) {
                [latencies addObject:@(dt)];
                totalSec += (dt / 1000.0);
                lastResult = pred;
            } else {
                res.success = NO;
                res.statusMessage = [NSString stringWithFormat:@"Prediction failed on iter %lu: %@", (unsigned long)iter, error.localizedDescription];
                return res;
            }
        }
    }

    double avgSec = totalSec / (double)iterations;
    res.avgLatencyMs = avgSec * 1000.0;
    res.tops = (totalOps / 1e12) / avgSec;
    res.fps = 1.0 / avgSec;

    double minMs = DBL_MAX, maxMs = 0.0;
    for (NSNumber *n in latencies) {
        double v = n.doubleValue;
        if (v < minMs) minMs = v;
        if (v > maxMs) maxMs = v;
    }
    res.minLatencyMs = minMs;
    res.maxLatencyMs = maxMs;

    // Output Verification
    MLMultiArray *outArray = [lastResult featureValueForName:outName].multiArrayValue;
    if (outArray) {
        res.totalElementCount = (NSInteger)(B * C * H * W);
        NSInteger zeros = 0;
        uint16_t *ptr = (uint16_t *)outArray.dataPointer;
        for (NSInteger i = 0; i < res.totalElementCount; i++) {
            if (ptr[i] == 0x0000 || ptr[i] == 0x8000) {
                zeros++;
            }
        }
        res.zeroElementCount = zeros;
        res.outputFiniteAndNonZero = (zeros < res.totalElementCount);
    }

    // Optional PMU Telemetry via _ANEClient
    if (usePMU && units == MLComputeUnitsCPUAndNeuralEngine) {
        if (progress) progress(@"Evaluating hardware PMU performance counters via _ANEClient...");
        Class aneClientClass = NSClassFromString(@"_ANEClient");
        Class aneModelClass = NSClassFromString(@"_ANEModel");
        Class aneRequestClass = NSClassFromString(@"_ANERequest");
        Class aneIOSurfClass = NSClassFromString(@"_ANEIOSurfaceObject");
        Class anePerfSurfClass = NSClassFromString(@"_ANEPerformanceStatsIOSurface");

        if (aneClientClass && aneModelClass && aneRequestClass) {
            _ANEClient *client = [_ANEClient sharedConnection];
            NSURL *milURL = [compiledModelURL URLByAppendingPathComponent:@"model.mil"];
            NSDictionary *opts = @{ kANEFModelTypeKey: kANEFModelMILValue };
            _ANEModel *aneModel = [_ANEModel modelAtURL:milURL key:nil];

            if ([client compileModel:aneModel options:opts qos:0 error:nil] &&
                [client loadModel:aneModel options:opts qos:0 error:nil]) {

                size_t inBytes = B * C * H * W * sizeof(uint16_t);
                size_t outBytes = B * C * H * W * sizeof(uint16_t);

                NSDictionary *surfProps = @{
                    (id)kIOSurfaceWidth: @(inBytes),
                    (id)kIOSurfaceHeight: @1,
                    (id)kIOSurfaceBytesPerElement: @1,
                    (id)kIOSurfaceBytesPerRow: @(inBytes),
                    (id)kIOSurfaceAllocSize: @(inBytes),
                };
                IOSurfaceRef inSurf = IOSurfaceCreate((CFDictionaryRef)surfProps);
                IOSurfaceRef outSurf = IOSurfaceCreate((CFDictionaryRef)surfProps);

                IOSurfaceLock(inSurf, 0, NULL);
                fillDenseFloat16(IOSurfaceGetBaseAddress(inSurf), B * C * H * W);
                IOSurfaceUnlock(inSurf, 0, NULL);

                _ANEIOSurfaceObject *inObj = [_ANEIOSurfaceObject objectWithIOSurface:inSurf];
                _ANEIOSurfaceObject *outObj = [_ANEIOSurfaceObject objectWithIOSurface:outSurf];

                IOSurfaceRef statsSurf = IOSurfaceCreate((CFDictionaryRef)@{
                    (id)kIOSurfaceWidth: @(1024),
                    (id)kIOSurfaceHeight: @1,
                    (id)kIOSurfaceBytesPerElement: @1,
                    (id)kIOSurfaceBytesPerRow: @(1024),
                    (id)kIOSurfaceAllocSize: @(1024),
                });
                _ANEIOSurfaceObject *statsObj = [_ANEIOSurfaceObject objectWithIOSurface:statsSurf];
                _ANEPerformanceStatsIOSurface *perfObj = [_ANEPerformanceStatsIOSurface objectWithIOSurface:statsObj statType:0];

                _ANERequest *req = [_ANERequest requestWithInputs:@[inObj]
                                                     inputIndices:@[@0]
                                                          outputs:@[outObj]
                                                    outputIndices:@[@0]
                                                        perfStats:@[perfObj]
                                                   procedureIndex:@0];

                NSDictionary *evalOpts = @{ kANEFPerformanceStatsMaskKey: @((1 << 0) | (1 << 3) | (1 << 5) | (1 << 7)) };
                if ([client evaluateWithModel:aneModel options:evalOpts request:req qos:0 error:nil]) {
                    NSData *d = req.perfStats.perfCounterData;
                    if (d && d.length >= 29 * sizeof(uint64_t)) {
                        const uint64_t *regs = (const uint64_t *)d.bytes;
                        res.nominalCycles = regs[10];
                        res.computeCycles = regs[13];
                        res.outputStallCycles = regs[15];
                        res.dmaBytes = regs[17];
                        if (res.nominalCycles > 0) {
                            res.aluSaturation = ((double)res.computeCycles / (double)res.nominalCycles) * 100.0;
                            res.effectiveClockGhz = ((double)res.nominalCycles / (avgSec * 1e9));
                        }
                    }
                }
                [client unloadModel:aneModel options:opts qos:0 error:nil];
                CFRelease(inSurf);
                CFRelease(outSurf);
                CFRelease(statsSurf);
            }
        }
    }

    res.success = YES;
    res.statusMessage = [NSString stringWithFormat:@"Success: %.2f TOPS, Latency: %.2f ms", res.tops, res.avgLatencyMs];
    return res;
}

@end
