//
// ANEClientBridge.h
// ANEClientVerifier
//
// Objective-C interface to AppleNeuralEngine (_ANEClient) for on-device iOS verification.
//

#import <Foundation/Foundation.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

NS_ASSUME_NONNULL_BEGIN

#pragma mark - Model Introspection Metadata

@interface ANEModelInfo : NSObject

@property (nonatomic, assign) BOOL success;
@property (nonatomic, copy) NSString *statusMessage;
@property (nonatomic, copy, nullable) NSString *modelFormat;
@property (nonatomic, copy, nullable) NSString *cacheURLIdentifier;
@property (nonatomic, assign) uint64_t programHandle;
@property (nonatomic, assign) uint64_t maxDramUsageBytes;
@property (nonatomic, copy) NSArray<NSDictionary<NSString *, id> *> *liveInputs;
@property (nonatomic, copy) NSArray<NSDictionary<NSString *, id> *> *liveOutputs;
@property (nonatomic, copy) NSDictionary<NSString *, id> *rawAttributes;

@end

#pragma mark - Live Silicon Execution & PMU Telemetry

@interface ANEVerificationResult : NSObject

@property (nonatomic, assign) BOOL success;
@property (nonatomic, copy) NSString *statusMessage;
@property (nonatomic, copy, nullable) NSString *modelFormat;

// Timing & Latency
@property (nonatomic, assign) uint64_t hwExecutionTimeNs;
@property (nonatomic, assign) double hwExecutionTimeMs;
@property (nonatomic, assign) double meanLatencyMs;
@property (nonatomic, assign) double warmupMs;
@property (nonatomic, assign) double throughputFps;
@property (nonatomic, copy) NSArray<NSNumber *> *latencies;

// Buffer Byte Sizes
@property (nonatomic, assign) NSUInteger rawStatsBytes;
@property (nonatomic, assign) NSUInteger perfCounterBytes;

// Computational Throughput & Silicon Capacity (TOPS & MACs)
@property (nonatomic, assign) double totalMacs;
@property (nonatomic, assign) double totalGops;
@property (nonatomic, assign) double topsRealized;
@property (nonatomic, assign) double macsPerCoreCycle;
@property (nonatomic, assign) double chipMacsPerCycle;
@property (nonatomic, assign) double aluSaturationFp16;
@property (nonatomic, assign) double aluSaturationInt8;
@property (nonatomic, assign) double effectiveClockGhz;
@property (nonatomic, assign) double effectiveCoreClockGhz;
@property (nonatomic, assign) double burstMacsPerCycle;

// 24 PMU Hardware Counters (Raw & Deltas)
@property (nonatomic, copy) NSDictionary<NSString *, NSNumber *> *performanceCounters; // Final raw counter values
@property (nonatomic, copy) NSDictionary<NSString *, NSNumber *> *performanceCounterInitial; // Baseline values captured post-warmup
@property (nonatomic, copy) NSDictionary<NSString *, NSNumber *> *performanceCounterDeltas; // Total delta across benchmark iterations
@property (nonatomic, copy) NSDictionary<NSString *, NSNumber *> *performanceCounterDeltasPerIter; // Average delta per iteration

// Categorized dictionaries for UI display
@property (nonatomic, copy) NSDictionary<NSString *, NSDictionary<NSString *, NSNumber *> *> *performanceCountersCategorized; // Final raw categorized
@property (nonatomic, copy) NSDictionary<NSString *, NSDictionary<NSString *, NSNumber *> *> *performanceCounterDeltasCategorized; // Delta per iter categorized
@property (nonatomic, copy) NSDictionary<NSString *, NSDictionary<NSString *, NSNumber *> *> *performanceCounterTotalDeltasCategorized; // Total delta categorized

// Descriptor Events & Metadata
@property (nonatomic, assign) uint32_t numTDs;
@property (nonatomic, assign) uint32_t totalEventsRecorded;
@property (nonatomic, assign) uint32_t totalEventsReceived;
@property (nonatomic, copy) NSArray<NSDictionary<NSString *, id> *> *decodedEvents;
@property (nonatomic, copy) NSDictionary<NSString *, id> *metadata;

@end

#pragma mark - Bridge Controller Interface

@interface ANEClientBridge : NSObject

/// Inspect model structure and memory footprint via _ANEClient compileModel
+ (ANEModelInfo *)introspectModelAtURL:(NSURL *)url
                                   key:(NSString *)key
                                  arch:(NSString *)arch
                                 error:(NSError * _Nullable * _Nullable)error;

/// Execute model on silicon with real-time PMU telemetry via _ANEClient (auto-detects workload MACs)
+ (ANEVerificationResult *)verifyModelAtURL:(NSURL *)url
                                        key:(NSString *)key
                                       arch:(NSString *)arch
                                   perfMask:(uint32_t)perfMask
                                 iterations:(NSUInteger)iterations;

/// Execute model on silicon with real-time PMU telemetry and explicit workload MACs count
+ (ANEVerificationResult *)verifyModelAtURL:(NSURL *)url
                                        key:(NSString *)key
                                       arch:(NSString *)arch
                                   perfMask:(uint32_t)perfMask
                                 iterations:(NSUInteger)iterations
                                  totalMacs:(double)totalMacs;

/// Unload currently loaded silicon model if active
+ (void)unloadActiveModel;

/// Convert ANEF performance stats mask (0..15) to driver mask using driverMaskForANEFMask:
+ (uint32_t)driverMaskForANEFMask:(uint32_t)anefMask;

/// String representation for 24 PMU hardware counter indexes
+ (NSString *)nameForPerfCounter:(int32_t)counterIndex;

/// String representation for descriptor event type opcodes
+ (NSString *)nameForEventType:(uint16_t)eventType;

/// Check device architecture and internal build status
+ (NSDictionary<NSString *, id> *)deviceSiliconInfo;

/// Safe Objective-C exception handling wrapper for MPSGraph compilation and execution
+ (BOOL)catchException:(void (NS_NOESCAPE ^)(void))block error:(__autoreleasing NSError * _Nullable * _Nullable)error NS_SWIFT_NAME(catchException(_:));

/// Sample current hardware PMU counter state from physical silicon
+ (NSDictionary<NSString *, NSNumber *> *)sampleCurrentPMUCounters;

/// Snapshot of existing ANE temporary directories prior to MPSGraph compilation
+ (NSSet<NSString *> *)existingANETempDirectories;

/// Locate newly created or most recent ANE compiled microcode bundle directory
+ (NSString * _Nullable)findNewANETempDirectorySince:(NSSet<NSString *> * _Nullable)beforeDirs;

/// Profile an ANE microcode bundle directory with specific tensor dimensions directly via _ANEClient with mask 15
+ (ANEVerificationResult *)profileANECIRBundleAtURL:(NSURL *)bundleURL
                                              batch:(NSUInteger)B
                                             height:(NSUInteger)H
                                              width:(NSUInteger)W
                                         inChannels:(NSUInteger)Ci
                                        outChannels:(NSUInteger)Co
                                           dataType:(MPSDataType)dataType
                                         iterations:(NSUInteger)iterations
                                          totalMacs:(double)totalMacs;

/// Locate recent ANE compiled microcode bundle directory in temporary storage
+ (NSString * _Nullable)findANETempDirectorySince:(NSDate * _Nullable)sinceDate;

/// Profile an ANE microcode bundle directory directly via _ANEClient with mask 15
+ (ANEVerificationResult *)profileANEBundleAtURL:(NSURL *)url
                                      iterations:(NSUInteger)iterations
                                       totalMacs:(double)totalMacs;

@end

NS_ASSUME_NONNULL_END
