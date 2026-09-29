//
//  MILCapacityEngineBridge.h
//  ANECapacityWithMILApp
//
//  Objective-C bridge to CoreML and _ANEClient for on-device MIL capacity measurement.
//

#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>

NS_ASSUME_NONNULL_BEGIN

@interface MILBenchmarkExecutionResult : NSObject

@property (nonatomic, assign) BOOL success;
@property (nonatomic, copy) NSString *statusMessage;

// Timing & Throughput
@property (nonatomic, assign) double compileTimeMs;
@property (nonatomic, assign) double loadTimeMs;
@property (nonatomic, assign) double avgLatencyMs;
@property (nonatomic, assign) double minLatencyMs;
@property (nonatomic, assign) double maxLatencyMs;
@property (nonatomic, assign) double tops;
@property (nonatomic, assign) double fps;
@property (nonatomic, assign) double totalGOPs;

// Verification
@property (nonatomic, assign) BOOL outputFiniteAndNonZero;
@property (nonatomic, assign) NSInteger zeroElementCount;
@property (nonatomic, assign) NSInteger totalElementCount;

// Hardware PMU Counters (via _ANEClient)
@property (nonatomic, assign) uint64_t computeCycles;
@property (nonatomic, assign) uint64_t nominalCycles;
@property (nonatomic, assign) uint64_t outputStallCycles;
@property (nonatomic, assign) uint64_t inputStallCycles;
@property (nonatomic, assign) uint64_t dmaBytes;
@property (nonatomic, assign) double aluSaturation;
@property (nonatomic, assign) double effectiveClockGhz;
@property (nonatomic, copy, nullable) NSString *devicePlacement;

@end

@interface MILCapacityEngineBridge : NSObject

/// Evaluates a compiled CoreML model (.mlmodelc) at the given URL with specified compute units and iterations.
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
                                    progressHandler:(nullable void (^)(NSString *log))progress;

/// Resolves or compiles a temporary CoreML model (.mlmodelc) from specification bytes and evaluates it.
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
                                            progressHandler:(nullable void (^)(NSString *log))progress;

/// Dynamically builds, compiles, and evaluates a CoreML MIL convolution model with zero bundle dependencies.
+ (MILBenchmarkExecutionResult *)evaluateDynamicModelWithBatch:(NSUInteger)B
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
                                               progressHandler:(nullable void (^)(NSString *log))progress;

@end

NS_ASSUME_NONNULL_END
