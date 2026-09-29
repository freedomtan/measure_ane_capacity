//
// ANEClientBridge.mm
// ANEClientVerifier
//
// Implementation of AppleNeuralEngine (_ANEClient) iOS verification bridge.
//

#import "ANEClientBridge.h"
#import <IOSurface/IOSurfaceRef.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <mach/mach_time.h>

#pragma mark - Model Info Implementation

@implementation ANEModelInfo
- (instancetype)init {
    self = [super init];
    if (self) {
        _success = NO;
        _statusMessage = @"Uninitialized";
        _liveInputs = @[];
        _liveOutputs = @[];
        _rawAttributes = @{};
    }
    return self;
}
@end

#pragma mark - Verification Result Implementation

@implementation ANEVerificationResult
- (instancetype)init {
    self = [super init];
    if (self) {
        _success = NO;
        _statusMessage = @"Uninitialized";
        _latencies = @[];
        _totalMacs = 0.0;
        _totalGops = 0.0;
        _topsRealized = 0.0;
        _macsPerCoreCycle = 0.0;
        _chipMacsPerCycle = 0.0;
        _aluSaturationFp16 = 0.0;
        _aluSaturationInt8 = 0.0;
        _effectiveClockGhz = 0.0;
        _effectiveCoreClockGhz = 0.0;
        _burstMacsPerCycle = 0.0;
        _performanceCounters = @{};
        _performanceCounterInitial = @{};
        _performanceCounterDeltas = @{};
        _performanceCounterDeltasPerIter = @{};
        _performanceCountersCategorized = @{};
        _performanceCounterDeltasCategorized = @{};
        _performanceCounterTotalDeltasCategorized = @{};
        _decodedEvents = @[];
        _metadata = @{};
    }
    return self;
}
@end

#pragma mark - Model Descriptor

typedef NS_ENUM(NSInteger, ANEModelFormat) {
    ANEModelFormatUnknown,
    ANEModelFormatMIL,
    ANEModelFormatANECIR
};

@interface ANEModelDescriptor : NSObject
@property (nonatomic, assign) ANEModelFormat format;
@property (nonatomic, copy) NSString *formatString;
@property (nonatomic, copy) NSString *regionKey;
@property (nonatomic, copy) NSString *netFile;
@property (nonatomic, copy) NSString *compilerOptionsFile;
@property (nonatomic, strong) NSURL *bundleURL;
@end

@implementation ANEModelDescriptor
@end

#pragma mark - Private AppleNeuralEngine Declarations

__attribute__((objc_runtime_visible))
@interface _ANEDeviceInfo : NSObject
+ (NSString *)aneArchitectureType;
+ (BOOL)isInternalBuild;
+ (long long)numANEs;
+ (long long)numANECores;
@end

__attribute__((objc_runtime_visible))
@interface _ANEIOSurfaceObject : NSObject
+ (instancetype)objectWithIOSurface:(IOSurfaceRef)surface;
@end

__attribute__((objc_runtime_visible))
@interface _ANEPerformanceStatsIOSurface : NSObject
+ (instancetype)objectWithIOSurface:(_ANEIOSurfaceObject *)ioSurface statType:(int)statType;
- (instancetype)initWithIOSurface:(_ANEIOSurfaceObject *)ioSurface statType:(NSInteger)statType;
@end

__attribute__((objc_runtime_visible))
@interface _ANEPerformanceStats : NSObject
@property (nonatomic, readonly) NSData *perfCounterData;
@property (nonatomic, readonly) NSData *pStatsRawData;
@property (nonatomic, readonly) unsigned long long hwExecutionTime;
@property (nonatomic, readonly) NSDictionary *performanceCounters;
+ (NSString *)stringForPerfCounter:(int)index;
+ (NSString *)stringForEventType:(unsigned short)eventType;
+ (unsigned int)driverMaskForANEFMask:(unsigned int)mask;
+ (NSDictionary *)decodePerformanceStats:(id)stats withOptions:(NSDictionary *)options;
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
+ (instancetype)modelAtURL:(NSURL *)url key:(NSString *)key;
+ (instancetype)modelAtURL:(NSURL *)url key:(NSString *)key mpsConstants:(NSString *)constants;
@property (nonatomic, readonly) NSDictionary *modelAttributes;
@property (nonatomic, readonly) NSString *cacheURLIdentifier;
@end

__attribute__((objc_runtime_visible))
@interface _ANEClient : NSObject
+ (instancetype)sharedConnection;
- (BOOL)compileModel:(_ANEModel *)model options:(NSDictionary *)options qos:(unsigned int)qos error:(NSError **)error;
- (BOOL)loadModel:(_ANEModel *)model options:(NSDictionary *)options qos:(unsigned int)qos error:(NSError **)error;
- (BOOL)unloadModel:(_ANEModel *)model options:(NSDictionary *)options qos:(unsigned int)qos error:(NSError **)error;
- (BOOL)evaluateWithModel:(_ANEModel *)model options:(NSDictionary *)options request:(_ANERequest *)request qos:(unsigned int)qos error:(NSError **)error;
- (BOOL)evaluateRealTimeWithModel:(_ANEModel *)model options:(NSDictionary *)options request:(_ANERequest *)request error:(NSError **)error;
@end

#pragma mark - Private Bridge Core

@implementation ANEClientBridge

+ (void)ensureFrameworksLoaded {
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
        dlopen("/System/Library/PrivateFrameworks/ANECompiler.framework/ANECompiler", RTLD_NOW);
        dlopen("/System/Library/PrivateFrameworks/ANEServices.framework/ANEServices", RTLD_NOW);
    });
}

+ (NSDictionary<NSString *, id> *)deviceSiliconInfo {
    [self ensureFrameworksLoaded];
    
    NSString *arch = @"Unknown";
    BOOL isInternal = NO;
    long long numAnes = 1;
    long long numCores = 16;
    
    if ([_ANEDeviceInfo respondsToSelector:@selector(aneArchitectureType)]) {
        arch = [_ANEDeviceInfo aneArchitectureType];
    }
    if ([_ANEDeviceInfo respondsToSelector:@selector(isInternalBuild)]) {
        isInternal = [_ANEDeviceInfo isInternalBuild];
    }
    if ([_ANEDeviceInfo respondsToSelector:@selector(numANEs)]) {
        numAnes = [_ANEDeviceInfo numANEs];
    }
    if ([_ANEDeviceInfo respondsToSelector:@selector(numANECores)]) {
        numCores = [_ANEDeviceInfo numANECores];
    }
    
    return @{
        @"architecture": arch ?: @"Unknown",
        @"isInternalBuild": @(isInternal),
        @"numANEs": @(numAnes),
        @"numANECores": @(numCores)
    };
}

+ (uint32_t)driverMaskForANEFMask:(uint32_t)anefMask {
    [self ensureFrameworksLoaded];
    if ([_ANEPerformanceStats respondsToSelector:@selector(driverMaskForANEFMask:)]) {
        return [_ANEPerformanceStats driverMaskForANEFMask:anefMask];
    }
    // Direct silicon bitwise translation fallback
    if (anefMask > 0x0f) return 0;
    uint32_t dm = (anefMask & 1)
                | (((anefMask >> 3) & 1) << 1)
                | (((anefMask >> 1) & 3) << 2);
    return dm;
}

+ (NSString *)nameForPerfCounter:(int32_t)index {
    [self ensureFrameworksLoaded];
    if ([_ANEPerformanceStats respondsToSelector:@selector(stringForPerfCounter:)]) {
        NSString *name = [_ANEPerformanceStats stringForPerfCounter:index];
        if (name && [name length] > 0) {
            if ([name hasSuffix:@":"]) {
                name = [name substringToIndex:name.length - 1];
            }
            return name;
        }
    }
    
    static NSString * const kNames[] = {
        @"kANE_AF_TO_L2_DATA", @"kANE_AF_TO_KM_DATA", @"kANE_L2_TO_AF_DATA", @"kANE_L2_TO_NE_DATA",
        @"kANE_NE_TO_L2_DATA", @"kANE_INT8_CYCLES", @"kANE_FP16_CYCLES", @"kANE_L2_READ_STALL_CYCLES",
        @"kANE_L2_WRITE_STALL_CYCLES", @"kANE_KM_STALL_CYCLES", @"kANE_NE_NOMINAL_CYCLES",
        @"kANE_NE_THROTTLE_CYCLES", @"kANE_L2_THROTTLE_CYCLES", @"kANE_NE_COMPUTE_CYCLES",
        @"kANE_NE_INPUT_STALL_CYCLES", @"kANE_NE_OUTPUT_STALL_CYCLES", @"kANE_NE_KERNEL_STALL_CYCLES",
        @"kANE_DMA_READWRITE_BYTES", @"kANE_DMA_READ_BYTES", @"kANE_DPE_ENERGY",
        @"kANE_L2_NOMINAL_CYCLES", @"kANE_L2PE_COMPUTE_CYCLES", @"kANE_L2PE_INPUT_STALL_CYCLES",
        @"kANE_L2PE_OUTPUT_STALL_CYCLES"
    };
    if (index >= 0 && index < 24) return kNames[index];
    return [NSString stringWithFormat:@"kANE_UNKNOWN_%d", index];
}

+ (NSString *)nameForEventType:(uint16_t)eventType {
    [self ensureFrameworksLoaded];
    if ([_ANEPerformanceStats respondsToSelector:@selector(stringForEventType:)]) {
        NSString *name = [_ANEPerformanceStats stringForEventType:eventType];
        if (name) return name;
    }
    
    switch (eventType) {
        case 0: return @"Prefetch";
        case 1: return @"Commit";
        case 2: return @"FetchDone";
        case 3: return @"StartExecution";
        case 4: return @"OutputToL2";
        case 5: return @"FinishExecution";
        case 6: return @"TaskSwitch";
        case 7: return @"ChangeDestination";
        case 8: return @"ChangeSource";
        case 9: return @"NoHighestPriority";
        case 10: return @"MacNan (Exception: NaN in Matrix unit)";
        case 11: return @"MacInf (Exception: Inf in Matrix unit)";
        case 12: return @"PpInf (Exception: Inf in Post-processor)";
        case 13: return @"BaseAddressError";
        case 14: return @"NeNonLinearLut";
        case 15: return @"PeSrc1Inf (Exception: Inf in PE Source 1)";
        case 16: return @"PeSrc2Inf (Exception: Inf in PE Source 2)";
        case 17: return @"PeSrc1Nan (Exception: NaN in PE Source 1)";
        case 18: return @"PeSrc2Nan (Exception: NaN in PE Source 2)";
        case 19: return @"PeOutputNan (Exception: NaN in PE Output)";
        case 20: return @"PeGlobalReductInf";
        case 21: return @"PeGlobalReductNan";
        case 22: return @"NeSrcInf";
        case 23: return @"NeSrcNan";
        case 24: return @"PrefetchFinish";
        case 25: return @"PrefetchTerminate";
        case 26: return @"FinishExecutionNEDMA";
        case 27: return @"DMAAXIError";
        case 28: return @"Delay";
        case 29: return @"PerfTracingCounter";
        case 30: return @"ContextSwitchOut";
        case 31: return @"WinoInpTransInf";
        case 32: return @"WinoKernTransInf";
        default: return [NSString stringWithFormat:@"Unknown_0x%02x", eventType];
    }
}

static IOSurfaceRef createIOSurface(size_t bytes) {
    size_t allocSize = (bytes + 0xFFF) & ~0xFFF;
    if (allocSize < 0x4000) allocSize = 0x4000;
    NSDictionary *props = @{
        (id)kIOSurfaceWidth: @(allocSize),
        (id)kIOSurfaceHeight: @(1),
        (id)kIOSurfaceBytesPerElement: @(1),
        (id)kIOSurfaceBytesPerRow: @(allocSize),
        (id)kIOSurfaceAllocSize: @(allocSize)
    };
    return IOSurfaceCreate((CFDictionaryRef)props);
}

static void fillIOSurfaceNonZero(IOSurfaceRef surf, size_t bytes, MPSDataType dataType) {
    if (!surf || bytes == 0) return;
    IOSurfaceLock(surf, 0, nil);
    void *ptr = IOSurfaceGetBaseAddress(surf);
    if (ptr) {
        uint64_t state = 0x5EED5EED5EED5EEDULL;
        if ((uint32_t)dataType == 0x10430008) { // MPSDataTypeFloat8e4m3
            // Random-sign, physical magnitude 0.5 (E4M3 0x30 = +0.5, 0xB0 = -0.5).
            // Matches decoupled FP8 fill in ANECapacityEngine.swift to prevent H18 zero-skip cliff.
            uint8_t *u8 = (uint8_t *)ptr;
            for (size_t i = 0; i < bytes; i++) {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                u8[i] = (state & 1) ? 0xB0 : 0x30;
            }
        } else if (dataType == MPSDataTypeFloat16) {
            // Random-sign, magnitude 1/32 (FP16 0x2800 = +0.03125, 0xA800 = -0.03125).
            uint16_t *f16 = (uint16_t *)ptr;
            size_t count = bytes / sizeof(uint16_t);
            for (size_t i = 0; i < count; i++) {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                f16[i] = (state & 1) ? 0xA800 : 0x2800;
            }
        } else {
            // INT8: Deterministic pseudo-random non-canceling signs {-1, 1}
            int8_t *i8 = (int8_t *)ptr;
            for (size_t i = 0; i < bytes; i++) {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                i8[i] = (state & 1) ? -1 : 1;
            }
        }
    }
    IOSurfaceUnlock(surf, 0, nil);
}

static NSDictionary<NSString *, NSDictionary<NSString *, NSNumber *> *> *categorizeCountersDict(NSDictionary<NSString *, NSNumber *> *src) {
    NSMutableDictionary *catCompute = [NSMutableDictionary dictionary];
    NSMutableDictionary *catMemory = [NSMutableDictionary dictionary];
    NSMutableDictionary *catStalls = [NSMutableDictionary dictionary];
    NSMutableDictionary *catThrottle = [NSMutableDictionary dictionary];
    NSMutableDictionary *catEnergy = [NSMutableDictionary dictionary];
    NSMutableDictionary *catL2PE = [NSMutableDictionary dictionary];
    
    for (NSString *key in src) {
        NSNumber *val = src[key];
        if ([key containsString:@"L2PE"]) {
            catL2PE[key] = val;
        } else if ([key containsString:@"INT8"] || [key containsString:@"FP16"] ||
                   [key containsString:@"NE_COMPUTE"] || [key containsString:@"NE_NOMINAL"] ||
                   [key containsString:@"L2_NOMINAL"]) {
            catCompute[key] = val;
        } else if ([key containsString:@"DATA"] || [key containsString:@"DMA"]) {
            catMemory[key] = val;
        } else if ([key containsString:@"STALL"]) {
            catStalls[key] = val;
        } else if ([key containsString:@"THROTTLE"]) {
            catThrottle[key] = val;
        } else if ([key containsString:@"ENERGY"] || [key containsString:@"DPE"]) {
            catEnergy[key] = val;
        }
    }
    
    return @{
        @"Compute & Arithmetic Cycles": catCompute,
        @"SRAM / DRAM Memory Bandwidth": catMemory,
        @"Pipeline Bottlenecks & Stalls": catStalls,
        @"Thermal & Clock Throttling": catThrottle,
        @"Digital Power & Energy": catEnergy,
        @"L2 Processing Element (L2PE)": catL2PE
    };
}

+ (ANEModelDescriptor *)detectDescriptorForURL:(NSURL *)url key:(NSString *)key arch:(NSString *)arch {
    ANEModelDescriptor *desc = [[ANEModelDescriptor alloc] init];
    desc.format = ANEModelFormatUnknown;
    desc.bundleURL = url;
    
    NSFileManager *fm = [NSFileManager defaultManager];
    NSString *path = url.path;
    BOOL isDir = NO;
    if (![fm fileExistsAtPath:path isDirectory:&isDir]) {
        desc.format = ANEModelFormatMIL;
        desc.formatString = @"MIL";
        desc.netFile = key ?: @"model.mil";
        desc.regionKey = key ?: @"model.mil";
        return desc;
    }
    
    NSString *bundleDir = isDir ? path : [path stringByDeletingLastPathComponent];
    desc.bundleURL = [NSURL fileURLWithPath:bundleDir];
    NSArray *files = [fm contentsOfDirectoryAtPath:bundleDir error:nil];
    
    // 1. Check for ANECIR (*.bc.mlir)
    for (NSString *f in files) {
        if ([f hasSuffix:@".bc.mlir"]) {
            desc.format = ANEModelFormatANECIR;
            desc.formatString = @"ANECIR";
            desc.netFile = f;
            desc.regionKey = [f substringToIndex:(f.length - @".bc.mlir".length)];
            break;
        }
    }
    
    if (desc.format == ANEModelFormatANECIR) {
        NSString *cand = [NSString stringWithFormat:@"compiler_options_%@.plist", desc.regionKey];
        if ([files containsObject:cand]) {
            desc.compilerOptionsFile = cand;
        } else if ([files containsObject:@"compiler_options.plist"]) {
            desc.compilerOptionsFile = @"compiler_options.plist";
        } else {
            for (NSString *f in files) {
                if ([f hasPrefix:@"compiler_options"] && [f hasSuffix:@".plist"]) {
                    desc.compilerOptionsFile = f;
                    break;
                }
            }
        }
        return desc;
    }
    
    // 2. Check for MIL (model.mil)
    if ([files containsObject:@"model.mil"] || [path.lastPathComponent isEqualToString:@"model.mil"] || [path.pathExtension isEqualToString:@"mil"]) {
        desc.format = ANEModelFormatMIL;
        desc.formatString = @"MIL";
        desc.netFile = @"model.mil";
        desc.regionKey = key ?: @"model.mil";
        return desc;
    }
    
    desc.format = ANEModelFormatMIL;
    desc.formatString = @"MIL";
    desc.netFile = key ?: @"model.mil";
    desc.regionKey = key ?: @"model.mil";
    return desc;
}

+ (ANEModelInfo *)introspectModelAtURL:(NSURL *)url
                                   key:(NSString *)key
                                  arch:(NSString *)arch
                                 error:(NSError * _Nullable * _Nullable)error {
    [self ensureFrameworksLoaded];
    ANEModelInfo *info = [[ANEModelInfo alloc] init];
    
    _ANEClient *client = [_ANEClient sharedConnection];
    if (!client) {
        info.statusMessage = @"Failed to acquire _ANEClient sharedConnection.";
        return info;
    }
    
    NSString *effectiveArch = arch ?: @"h16g";
    ANEModelDescriptor *desc = [self detectDescriptorForURL:url key:key arch:effectiveArch];
    info.modelFormat = desc.formatString;
    
    _ANEModel *model = nil;
    if (desc.format == ANEModelFormatANECIR) {
        if ([_ANEModel respondsToSelector:@selector(modelAtURL:key:mpsConstants:)]) {
            model = [_ANEModel modelAtURL:desc.bundleURL key:desc.regionKey mpsConstants:@"constants"];
        } else {
            model = [_ANEModel modelAtURL:desc.bundleURL key:desc.regionKey];
        }
    } else {
        model = [_ANEModel modelAtURL:desc.bundleURL key:desc.regionKey];
    }
    
    if (!model) {
        info.statusMessage = [NSString stringWithFormat:@"Failed to create _ANEModel at URL: %@", desc.bundleURL.path];
        return info;
    }
    
    NSMutableDictionary *compileOpts = [NSMutableDictionary dictionary];
    if (desc.format == ANEModelFormatANECIR) {
        compileOpts[@"kANEFModelType"] = @"kANEFModelANECIR";
        compileOpts[@"kANEFCompilerOptionsFilenameKey"] = desc.compilerOptionsFile ?: [NSString stringWithFormat:@"compiler_options_%@.plist", desc.regionKey];
        compileOpts[@"kANEFNetPlistFilenameKey"] = desc.netFile;
        compileOpts[@"kANEFTargetArchitectureKey"] = effectiveArch;
    } else {
        compileOpts[@"kANEFModelType"] = @"kANEFModelMIL";
        compileOpts[@"kANEFNetPlistFilenameKey"] = desc.netFile;
        compileOpts[@"kANEFTargetArchitectureKey"] = effectiveArch;
    }
    compileOpts[@"GenerateAnalyticsBuffer"] = @YES;
    compileOpts[@"GenerateStaticPerfAnalytics"] = @YES;
    compileOpts[@"DumpStatusDictionaryToFile"] = @YES;
    
    NSError *compileErr = nil;
    BOOL ok = [client compileModel:model options:compileOpts qos:25 error:&compileErr];
    
    if (!ok) {
        info.statusMessage = [NSString stringWithFormat:@"_ANEClient compileModel failed: %@", compileErr.localizedDescription ?: @"Unknown error"];
        if (error) *error = compileErr;
        return info;
    }
    
    info.success = YES;
    info.statusMessage = [NSString stringWithFormat:@"%@ model compiled and introspected successfully.", desc.formatString];
    
    if ([model respondsToSelector:@selector(cacheURLIdentifier)]) {
        info.cacheURLIdentifier = model.cacheURLIdentifier;
    }
    
    NSDictionary *attrs = model.modelAttributes;
    if (attrs && [attrs isKindOfClass:[NSDictionary class]]) {
        info.rawAttributes = attrs;
        
        // Extract ModelMaxDramUsage
        if (attrs[@"ModelMaxDramUsage"]) {
            info.maxDramUsageBytes = [attrs[@"ModelMaxDramUsage"] unsignedLongLongValue];
        }
        
        // Extract LiveInputList and LiveOutputList
        NSArray *netList = attrs[@"NetworkStatusList"];
        if (netList && netList.count > 0) {
            NSDictionary *mainNet = netList[0];
            if (mainNet[@"LiveInputList"]) {
                info.liveInputs = mainNet[@"LiveInputList"];
            }
            if (mainNet[@"LiveOutputList"]) {
                info.liveOutputs = mainNet[@"LiveOutputList"];
            }
        }
    }
    
    return info;
}

static _ANEModel *gActiveLoadedModel = nil;
static NSString *gActiveModelKey = nil;
static NSURL *gActiveModelURL = nil;
static NSString *gActiveArch = nil;
static uint32_t gActivePerfMask = 0xFFFFFFFF;
static NSDictionary *gActiveLoadOpts = nil;

+ (void)unloadActiveModel {
    if (gActiveLoadedModel) {
        _ANEClient *client = [_ANEClient sharedConnection];
        if (client) {
            [client unloadModel:gActiveLoadedModel options:gActiveLoadOpts ?: @{} qos:25 error:nil];
        }
        gActiveLoadedModel = nil;
        gActiveModelKey = nil;
        gActiveModelURL = nil;
        gActiveArch = nil;
        gActivePerfMask = 0xFFFFFFFF;
        gActiveLoadOpts = nil;
    }
}

+ (ANEVerificationResult *)verifyModelAtURL:(NSURL *)url
                                        key:(NSString *)key
                                       arch:(NSString *)arch
                                   perfMask:(uint32_t)perfMask
                                 iterations:(NSUInteger)iterations {
    return [self verifyModelAtURL:url key:key arch:arch perfMask:perfMask iterations:iterations totalMacs:0.0];
}

+ (ANEVerificationResult *)verifyModelAtURL:(NSURL *)url
                                        key:(NSString *)key
                                       arch:(NSString *)arch
                                   perfMask:(uint32_t)perfMask
                                 iterations:(NSUInteger)iterations
                                  totalMacs:(double)explicitMacs {
    [self ensureFrameworksLoaded];
    ANEVerificationResult *res = [[ANEVerificationResult alloc] init];
    
    // Auto-detect theoretical workload MACs if not explicitly passed
    double workloadMacs = explicitMacs;
    if (workloadMacs <= 0.0) {
        NSString *modelIdent = [key lowercaseString] ?: @"";
        NSString *urlIdent = [[url lastPathComponent] lowercaseString] ?: @"";
        if ([modelIdent containsString:@"conv_fp16"] || [urlIdent containsString:@"conv_fp16"] ||
            [modelIdent containsString:@"conv_int8"] || [urlIdent containsString:@"conv_int8"]) {
            workloadMacs = 193273528320.0; // 193.27 GMACs (20 layers of 256x256x128 3x3 conv)
        } else if ([modelIdent containsString:@"test_conv3x3"] || [urlIdent containsString:@"test_conv3x3"]) {
            workloadMacs = 2359296.0; // 2.36 MMACs (1 layer of 32x32x16 3x3 conv)
        }
    }
    res.totalMacs = workloadMacs;
    res.totalGops = (workloadMacs * 2.0) / 1e9;
    
    MPSDataType modelDataType = MPSDataTypeFloat16;
    NSString *fullIdent = [NSString stringWithFormat:@"%@ %@", key ?: @"", [url lastPathComponent] ?: @""].lowercaseString;
    if ([fullIdent containsString:@"int8"]) {
        modelDataType = MPSDataTypeInt8;
    } else if ([fullIdent containsString:@"fp8"]) {
        modelDataType = (MPSDataType)0x10430008;
    }
    
    _ANEClient *client = [_ANEClient sharedConnection];
    if (!client) {
        res.statusMessage = @"Failed to acquire _ANEClient connection.";
        return res;
    }
    
    NSString *effectiveArch = arch ?: @"h16g";
    ANEModelDescriptor *desc = [self detectDescriptorForURL:url key:key arch:effectiveArch];
    res.modelFormat = desc.formatString;
    
    _ANEModel *model = nil;
    BOOL needLoad = YES;
    
    // Check if we can reuse the already loaded model on ANE silicon
    if (gActiveLoadedModel && [gActiveModelURL isEqual:desc.bundleURL] && [gActiveModelKey isEqualToString:desc.regionKey] && [gActiveArch isEqualToString:effectiveArch] && gActivePerfMask == perfMask) {
        model = gActiveLoadedModel;
        needLoad = NO;
    } else {
        // Model or mask changed -> unload previously loaded model first
        [self unloadActiveModel];
    }
    
    NSMutableDictionary *compileOpts = [NSMutableDictionary dictionary];
    if (desc.format == ANEModelFormatANECIR) {
        compileOpts[@"kANEFModelType"] = @"kANEFModelANECIR";
        compileOpts[@"kANEFCompilerOptionsFilenameKey"] = desc.compilerOptionsFile ?: [NSString stringWithFormat:@"compiler_options_%@.plist", desc.regionKey];
        compileOpts[@"kANEFNetPlistFilenameKey"] = desc.netFile;
        compileOpts[@"kANEFTargetArchitectureKey"] = effectiveArch;
    } else {
        compileOpts[@"kANEFModelType"] = @"kANEFModelMIL";
        compileOpts[@"kANEFNetPlistFilenameKey"] = desc.netFile;
        compileOpts[@"kANEFTargetArchitectureKey"] = effectiveArch;
    }
    compileOpts[@"GenerateAnalyticsBuffer"] = @YES;
    compileOpts[@"GenerateStaticPerfAnalytics"] = @YES;
    compileOpts[@"DumpStatusDictionaryToFile"] = @YES;
    
    NSMutableDictionary *loadOpts = [compileOpts mutableCopy];
    loadOpts[@"kANEFPerformanceStatsMask"] = @(perfMask);
    loadOpts[@"kANEFRetainModelsWithoutSourceURL"] = @1;
    
    if (needLoad) {
        if (desc.format == ANEModelFormatANECIR) {
            if ([_ANEModel respondsToSelector:@selector(modelAtURL:key:mpsConstants:)]) {
                model = [_ANEModel modelAtURL:desc.bundleURL key:desc.regionKey mpsConstants:@"constants"];
            } else {
                model = [_ANEModel modelAtURL:desc.bundleURL key:desc.regionKey];
            }
        } else {
            model = [_ANEModel modelAtURL:desc.bundleURL key:desc.regionKey];
        }
        
        if (!model) {
            res.statusMessage = @"Failed to initialize _ANEModel.";
            return res;
        }
        
        // Step 1: Compile Model (MIL models only; ANECIR is already precompiled)
        if (desc.format != ANEModelFormatANECIR) {
            NSError *compileErr = nil;
            BOOL compOk = [client compileModel:model options:compileOpts qos:25 error:&compileErr];
            if (!compOk) {
                res.statusMessage = [NSString stringWithFormat:@"Compilation error: %@", compileErr.localizedDescription ?: @"Unknown"];
                return res;
            }
        }
        
        // Step 2: Load Model with Performance Stats Mask
        NSError *loadErr = nil;
        BOOL loadOk = [client loadModel:model options:loadOpts qos:25 error:&loadErr];
        if (!loadOk) {
            res.statusMessage = [NSString stringWithFormat:@"Load error: %@", loadErr.localizedDescription ?: @"Unknown"];
            return res;
        }
        
        gActiveLoadedModel = model;
        gActiveModelURL = desc.bundleURL;
        gActiveModelKey = desc.regionKey;
        gActiveArch = effectiveArch;
        gActivePerfMask = perfMask;
        gActiveLoadOpts = loadOpts;
    }
    
    // Step 3: Inspect NetworkStatusList for tensor requirements
    NSDictionary *attrs = model.modelAttributes;
    NSArray *netList = (attrs && [attrs isKindOfClass:[NSDictionary class]]) ? attrs[@"NetworkStatusList"] : nil;
    
    NSArray *inputs = @[];
    NSArray *outputs = @[];
    if (netList && netList.count > 0) {
        NSDictionary *mainNet = netList[0];
        inputs = mainNet[@"LiveInputList"] ?: @[];
        outputs = mainNet[@"LiveOutputList"] ?: @[];
    }
    
    // Step 4: Allocate Input & Output IOSurfaces
    NSMutableArray<_ANEIOSurfaceObject *> *inObjects = [NSMutableArray array];
    NSMutableArray<NSNumber *> *inIndices = [NSMutableArray array];
    NSMutableArray *inSurfs = [NSMutableArray array];
    
    uint32_t inIdx = 0;
    for (NSDictionary *t in inputs) {
        uint64_t bStride = [t[@"BatchStride"] unsignedLongLongValue] ?: 4096;
        uint64_t batches = [t[@"Batches"] unsignedLongLongValue] ?: 1;
        size_t bytes = (size_t)(bStride * batches);
        IOSurfaceRef surf = createIOSurface(bytes);
        fillIOSurfaceNonZero(surf, bytes, modelDataType);
        [inSurfs addObject:(__bridge id)surf];
        _ANEIOSurfaceObject *obj = [_ANEIOSurfaceObject objectWithIOSurface:surf];
        [inObjects addObject:obj];
        [inIndices addObject:@(inIdx++)];
        CFRelease(surf);
    }
    
    // Fallback if model has no declared live inputs
    if (inObjects.count == 0) {
        IOSurfaceRef surf = createIOSurface(0x4000);
        fillIOSurfaceNonZero(surf, 0x4000, modelDataType);
        [inSurfs addObject:(__bridge id)surf];
        _ANEIOSurfaceObject *obj = [_ANEIOSurfaceObject objectWithIOSurface:surf];
        [inObjects addObject:obj];
        [inIndices addObject:@(0)];
        CFRelease(surf);
    }
    
    NSMutableArray<_ANEIOSurfaceObject *> *outObjects = [NSMutableArray array];
    NSMutableArray<NSNumber *> *outIndices = [NSMutableArray array];
    NSMutableArray *outSurfs = [NSMutableArray array];
    
    uint32_t outIdx = 0;
    for (NSDictionary *t in outputs) {
        uint64_t bStride = [t[@"BatchStride"] unsignedLongLongValue] ?: 4096;
        uint64_t batches = [t[@"Batches"] unsignedLongLongValue] ?: 1;
        size_t bytes = (size_t)(bStride * batches);
        IOSurfaceRef surf = createIOSurface(bytes);
        [outSurfs addObject:(__bridge id)surf];
        _ANEIOSurfaceObject *obj = [_ANEIOSurfaceObject objectWithIOSurface:surf];
        [outObjects addObject:obj];
        [outIndices addObject:@(outIdx++)];
        CFRelease(surf);
    }
    
    if (outObjects.count == 0) {
        IOSurfaceRef surf = createIOSurface(0x4000);
        [outSurfs addObject:(__bridge id)surf];
        _ANEIOSurfaceObject *obj = [_ANEIOSurfaceObject objectWithIOSurface:surf];
        [outObjects addObject:obj];
        [outIndices addObject:@(0)];
        CFRelease(surf);
    }
    
    // Step 5: Allocate Stats IOSurface (statType = 2 for PMU telemetry)
    IOSurfaceRef statsSurf = createIOSurface(0x4000);
    _ANEIOSurfaceObject *statsIoObj = [_ANEIOSurfaceObject objectWithIOSurface:statsSurf];
    _ANEPerformanceStatsIOSurface *statsSurfObj = nil;
    if ([_ANEPerformanceStatsIOSurface respondsToSelector:@selector(objectWithIOSurface:statType:)]) {
        statsSurfObj = [_ANEPerformanceStatsIOSurface objectWithIOSurface:statsIoObj statType:2];
    } else {
        statsSurfObj = [[_ANEPerformanceStatsIOSurface alloc] initWithIOSurface:statsIoObj statType:2];
    }
    CFRelease(statsSurf);
    
    // Step 6: Create _ANERequest
    _ANERequest *request = [_ANERequest requestWithInputs:inObjects
                                             inputIndices:inIndices
                                                  outputs:outObjects
                                            outputIndices:outIndices
                                                perfStats:@[statsSurfObj]
                                           procedureIndex:@(0)];
    
    if (!request) {
        res.statusMessage = @"Failed to construct _ANERequest with perfStats.";
        [self unloadActiveModel];
        return res;
    }
    
    NSDictionary *evalOpts = @{
        @"kANEFPerformanceStatsMask": @(perfMask),
        @"enableProfiling": @YES
    };
    
    // Step 7: Warm-up Iteration
    NSError *evalErr = nil;
    uint64_t tW0 = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
    BOOL warmOk = [client evaluateWithModel:model options:evalOpts request:request qos:25 error:&evalErr];
    uint64_t dtWarm = clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tW0;
    res.warmupMs = (double)dtWarm / 1000000.0;
    
    if (!warmOk) {
        res.statusMessage = [NSString stringWithFormat:@"Evaluation error on silicon: %@", evalErr.localizedDescription ?: @"Unknown error"];
        [self unloadActiveModel];
        return res;
    }
    
    // Capture baseline initial PMU counters immediately after warm-up
    NSMutableDictionary<NSString *, NSNumber *> *initialCounters = [NSMutableDictionary dictionary];
    _ANEPerformanceStats *warmPerfStats = request.perfStats;
    if (warmPerfStats) {
        if ([warmPerfStats respondsToSelector:@selector(performanceCounters)]) {
            NSDictionary *c = warmPerfStats.performanceCounters;
            if (c) {
                for (id rawK in c) {
                    NSString *k = [rawK description];
                    if ([k hasSuffix:@":"]) {
                        k = [k substringToIndex:k.length - 1];
                    }
                    initialCounters[k] = c[rawK];
                }
            }
        }
        NSData *wData = warmPerfStats.perfCounterData;
        if (wData && wData.length >= sizeof(uint64_t)) {
            const uint64_t *wRegs = (const uint64_t *)wData.bytes;
            size_t wCount = wData.length / sizeof(uint64_t);
            for (size_t i = 0; i < wCount && i < 29; i++) {
                NSString *regName = [self nameForPerfCounter:(int32_t)i];
                if (!initialCounters[regName]) {
                    initialCounters[regName] = @(wRegs[i]);
                }
            }
        }
    }
    
    // Step 8: Execution Iterations
    NSUInteger numIters = (iterations > 0) ? iterations : 1;
    NSMutableArray<NSNumber *> *latencies = [NSMutableArray arrayWithCapacity:numIters];
    uint64_t totalNs = 0;
    
    for (NSUInteger i = 0; i < numIters; i++) {
        uint64_t t0 = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
        BOOL ok = [client evaluateWithModel:model options:evalOpts request:request qos:25 error:&evalErr];
        uint64_t dt = clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - t0;
        totalNs += dt;
        [latencies addObject:@((double)dt / 1000000.0)];
        if (!ok) break;
    }
    
    res.latencies = latencies;
    res.meanLatencyMs = (double)(totalNs / numIters) / 1000000.0;
    if (res.meanLatencyMs > 0.0) {
        res.throughputFps = 1000.0 / res.meanLatencyMs;
    }
    
    // Step 9: Extract & Decode _ANEPerformanceStats
    _ANEPerformanceStats *perfStats = request.perfStats;
    if (perfStats) {
        if ([perfStats respondsToSelector:@selector(hwExecutionTime)]) {
            res.hwExecutionTimeNs = perfStats.hwExecutionTime;
            res.hwExecutionTimeMs = (double)res.hwExecutionTimeNs / 1000000.0;
        }
        
        NSData *perfCounterData = perfStats.perfCounterData;
        if (perfCounterData) {
            res.perfCounterBytes = perfCounterData.length;
        }
        
        NSData *pStatsRawData = perfStats.pStatsRawData;
        if (pStatsRawData) {
            res.rawStatsBytes = pStatsRawData.length;
        }
        
        // Extract final raw counters
        NSMutableDictionary<NSString *, NSNumber *> *finalCounters = [NSMutableDictionary dictionary];
        if ([perfStats respondsToSelector:@selector(performanceCounters)]) {
            NSDictionary *c = perfStats.performanceCounters;
            if (c) {
                for (id rawK in c) {
                    NSString *k = [rawK description];
                    if ([k hasSuffix:@":"]) {
                        k = [k substringToIndex:k.length - 1];
                    }
                    finalCounters[k] = c[rawK];
                }
            }
        }
        if (perfCounterData && perfCounterData.length >= sizeof(uint64_t)) {
            const uint64_t *fRegs = (const uint64_t *)perfCounterData.bytes;
            size_t fCount = perfCounterData.length / sizeof(uint64_t);
            for (size_t i = 0; i < fCount && i < 29; i++) {
                NSString *regName = [self nameForPerfCounter:(int32_t)i];
                if (!finalCounters[regName]) {
                    finalCounters[regName] = @(fRegs[i]);
                }
            }
        }
        
        res.performanceCounters = finalCounters;
        res.performanceCounterInitial = initialCounters;
        
        // Calculate Deltas: Total Delta across all iterations and Delta Per Iteration
        NSMutableDictionary<NSString *, NSNumber *> *deltasTotal = [NSMutableDictionary dictionary];
        NSMutableDictionary<NSString *, NSNumber *> *deltasPerIter = [NSMutableDictionary dictionary];
        
        for (NSString *key in finalCounters) {
            uint64_t vFinal = [finalCounters[key] unsignedLongLongValue];
            uint64_t vInitial = initialCounters[key] ? [initialCounters[key] unsignedLongLongValue] : 0;
            uint64_t dTotal = 0;
            if (vFinal >= vInitial) {
                dTotal = vFinal - vInitial;
            } else {
                dTotal = vFinal;
            }
            uint64_t dPerIter = (numIters > 0) ? (dTotal / numIters) : dTotal;
            deltasTotal[key] = @(dTotal);
            deltasPerIter[key] = @(dPerIter);
        }
        
        res.performanceCounterDeltas = deltasTotal;
        res.performanceCounterDeltasPerIter = deltasPerIter;
        
        // Organize into the 6 intuitive hardware categories
        res.performanceCountersCategorized = categorizeCountersDict(finalCounters);
        res.performanceCounterDeltasCategorized = categorizeCountersDict(deltasPerIter);
        res.performanceCounterTotalDeltasCategorized = categorizeCountersDict(deltasTotal);
        
        // Microarchitectural Silicon Throughput & Capacity Calculations
        uint64_t neNomPerIter = 0;
        uint64_t neCompPerIter = 0;
        
        for (NSString *key in deltasPerIter) {
            if ([key containsString:@"NE_NOMINAL_CYCLES"]) {
                neNomPerIter = [deltasPerIter[key] unsignedLongLongValue];
            } else if ([key containsString:@"NE_COMPUTE_CYCLES"]) {
                neCompPerIter = [deltasPerIter[key] unsignedLongLongValue];
            }
        }
        
        uint64_t timeNs = res.hwExecutionTimeNs;
        if (timeNs == 0 && res.meanLatencyMs > 0) {
            timeNs = (uint64_t)(res.meanLatencyMs * 1e6);
        }
        
        if (workloadMacs > 0.0) {
            if (timeNs > 0) {
                // Realized TOPS = (Total Operations) / (time (s) * 1e12) = (2 * Total MACs) / (timeNs * 1000)
                res.topsRealized = ((workloadMacs * 2.0) / ((double)timeNs * 1000.0));
            }
            
            if (neNomPerIter > 0) {
                // Throughput / Core Cycle (Target bounded by 256 for FP16, 512 for INT8)
                res.macsPerCoreCycle = workloadMacs / (double)neNomPerIter;
                // Total Chip Throughput across 16 cores (Target bounded by 4,096 for FP16, 8,192 for INT8)
                res.chipMacsPerCycle = res.macsPerCoreCycle * 16.0;
                // Sustained ALU Saturation vs. Theoretical Architectural Peaks
                res.aluSaturationFp16 = (res.macsPerCoreCycle / 256.0) * 100.0;
                res.aluSaturationInt8 = (res.macsPerCoreCycle / 512.0) * 100.0;
            }
            
            if (neCompPerIter > 0) {
                // Unstalled active execution burst intensity
                res.burstMacsPerCycle = workloadMacs / (double)neCompPerIter;
            }
        }
        
        if (timeNs > 0 && neNomPerIter > 0) {
            // Dynamic Silicon DVFS Clock:
            // neNomPerIter records the aggregate reference clock cycles summed across 16 cores.
            // neNomPerIter / timeNs gives aggregate GHz; dividing by 16 gives per-core GHz.
            res.effectiveClockGhz = (double)neNomPerIter / (double)timeNs;
            res.effectiveCoreClockGhz = res.effectiveClockGhz / 16.0;
        }
        
        // Decode raw descriptor telemetry
        if ([_ANEPerformanceStats respondsToSelector:@selector(decodePerformanceStats:withOptions:)]) {
            NSDictionary *decoded = [_ANEPerformanceStats decodePerformanceStats:perfStats withOptions:@{ @"kANEFPerformanceStatsMask": @(perfMask) }];
            
            if (decoded && [decoded isKindOfClass:[NSDictionary class]]) {
                NSDictionary *rawStats = decoded[@"rawStats"];
                if (rawStats && [rawStats isKindOfClass:[NSDictionary class]]) {
                    if (rawStats[@"metadata"]) {
                        res.metadata = rawStats[@"metadata"];
                    }
                    NSArray *descs = rawStats[@"descriptors"];
                    if (descs && descs.count > 0) {
                        NSDictionary *d0 = descs[0];
                        res.numTDs = [d0[@"numTDs"] unsignedIntValue];
                        res.totalEventsRecorded = [d0[@"totalEventsRecorded"] unsignedIntValue];
                        res.totalEventsReceived = [d0[@"totalEventsReceived"] unsignedIntValue];
                        if (d0[@"events"]) {
                            res.decodedEvents = d0[@"events"];
                        }
                    }
                }
            }
        }
    }
    
    // Fallback TOPS calculation if perfStats was not enabled or returned nil
    if (workloadMacs > 0.0 && res.topsRealized == 0.0) {
        uint64_t timeNs = res.hwExecutionTimeNs;
        if (timeNs == 0 && res.meanLatencyMs > 0) {
            timeNs = (uint64_t)(res.meanLatencyMs * 1e6);
        }
        if (timeNs > 0) {
            res.topsRealized = ((workloadMacs * 2.0) / ((double)timeNs * 1000.0));
        }
    }
    
    // Model remains loaded in gActiveLoadedModel for immediate subsequent verification runs
    // without triggering IOKit exclusive access conflict (0xe00002f0).
    res.success = YES;
    if (res.topsRealized > 0.0) {
        res.statusMessage = [NSString stringWithFormat:@"Verified on silicon (%@): mean %.3f ms (%.1f FPS) | Realized: %.2f TOPS.", desc.formatString, res.meanLatencyMs, res.throughputFps, res.topsRealized];
    } else {
        res.statusMessage = [NSString stringWithFormat:@"Verified on silicon (%@): mean %.3f ms (%.1f FPS).", desc.formatString, res.meanLatencyMs, res.throughputFps];
    }
    return res;
}

+ (BOOL)catchException:(void (NS_NOESCAPE ^)(void))block error:(__autoreleasing NSError * _Nullable * _Nullable)error {
    @try {
        block();
        return YES;
    } @catch (NSException *exception) {
        if (error) {
            NSString *reason = exception.reason ?: exception.name;
            *error = [NSError errorWithDomain:@"com.apple.ane.exception"
                                         code:-1
                                     userInfo:@{
                NSLocalizedDescriptionKey: [NSString stringWithFormat:@"MPSGraph Exception: %@", reason]
            }];
        }
        return NO;
    }
}

+ (NSDictionary<NSString *, NSNumber *> *)sampleCurrentPMUCounters {
    NSURL *probeURL = nil;
    NSString *probeKey = @"main_ANE_region_0_0";
    if (gActiveLoadedModel && gActiveModelURL) {
        probeURL = gActiveModelURL;
        probeKey = gActiveModelKey ?: @"main_ANE_region_0_0";
    } else {
        probeURL = [[NSBundle mainBundle] URLForResource:@"test_conv3x3" withExtension:nil];
        if (!probeURL) {
            probeURL = [[NSBundle mainBundle] URLForResource:@"test_conv3x3" withExtension:nil subdirectory:@"Resources"];
        }
    }
    if (!probeURL) return @{};
    
    ANEVerificationResult *res = [self verifyModelAtURL:probeURL key:probeKey arch:@"h16g" perfMask:15 iterations:1 totalMacs:0];
    return res.performanceCounters ?: @{};
}

static NSArray<NSString *> *getANETempBaseDirectories(void) {
    NSMutableArray<NSString *> *bases = [NSMutableArray array];
    NSString *tmp = NSTemporaryDirectory();
    if (tmp) {
        [bases addObject:[tmp stringByAppendingPathComponent:@"com.apple.MetalPerformanceShadersGraph"]];
        [bases addObject:tmp];
    }
    [bases addObject:@"/tmp/com.apple.MetalPerformanceShadersGraph"];
    [bases addObject:@"/tmp"];
    return bases;
}

+ (NSSet<NSString *> *)existingANETempDirectories {
    NSFileManager *fm = [NSFileManager defaultManager];
    NSMutableSet<NSString *> *set = [NSMutableSet set];
    for (NSString *base in getANETempBaseDirectories()) {
        BOOL isDir = NO;
        if ([fm fileExistsAtPath:base isDirectory:&isDir] && isDir) {
            NSArray *subdirs = [fm contentsOfDirectoryAtPath:base error:nil];
            for (NSString *sub in subdirs) {
                [set addObject:[base stringByAppendingPathComponent:sub]];
            }
        }
    }
    return set;
}

+ (NSString * _Nullable)findNewANETempDirectorySince:(NSSet<NSString *> * _Nullable)beforeDirs {
    NSFileManager *fm = [NSFileManager defaultManager];

    // Only ever return a directory that did not exist in beforeDirs. There
    // used to be a second fallback pass here that, when no *new* bundle was
    // found, returned the most-recently-modified bundle from ANY prior run --
    // with no check against beforeDirs or "now" at all. That meant a config
    // whose ANE compile silently fails and falls back to GPU (e.g. every FP8
    // benchmark: MPSGraph's dequantize/quantize passes reject FP8 MLIR on the
    // ANE compiler) would silently be attributed the PMU telemetry of
    // whatever real ANE bundle a previous, unrelated benchmark happened to
    // leave behind in /tmp -- reporting a plausible-looking but fabricated
    // TOPS/ALU-saturation number instead of "no ANE bundle for this run."
    // Returning nil here instead surfaces the honest
    // "[PMU Note] No temporary ANE bundle emitted..." log path in
    // ANECapacityEngine.swift.
    for (NSString *base in getANETempBaseDirectories()) {
        BOOL isDir = NO;
        if (![fm fileExistsAtPath:base isDirectory:&isDir] || !isDir) continue;

        NSArray *subdirs = [fm contentsOfDirectoryAtPath:base error:nil];
        for (NSString *sub in subdirs) {
            NSString *full = [base stringByAppendingPathComponent:sub];
            if (!beforeDirs || ![beforeDirs containsObject:full]) {
                BOOL subIsDir = NO;
                if ([fm fileExistsAtPath:full isDirectory:&subIsDir] && subIsDir) {
                    NSArray *files = [fm contentsOfDirectoryAtPath:full error:nil];
                    for (NSString *f in files) {
                        if ([f hasSuffix:@".bc.mlir"] || [f hasSuffix:@".hwx"] || [f hasSuffix:@".mil"]) {
                            return full;
                        }
                    }
                }
            }
        }
    }

    return nil;
}

+ (NSString * _Nullable)findANETempDirectorySince:(NSDate * _Nullable)sinceDate {
    return [self findNewANETempDirectorySince:nil];
}

+ (ANEVerificationResult *)profileANECIRBundleAtURL:(NSURL *)bundleURL
                                              batch:(NSUInteger)B
                                             height:(NSUInteger)H
                                              width:(NSUInteger)W
                                         inChannels:(NSUInteger)Ci
                                        outChannels:(NSUInteger)Co
                                           dataType:(MPSDataType)dataType
                                         iterations:(NSUInteger)iterations
                                          totalMacs:(double)totalMacs {
    [self ensureFrameworksLoaded];
    ANEVerificationResult *res = [[ANEVerificationResult alloc] init];
    res.totalMacs = totalMacs;
    res.totalGops = (totalMacs * 2.0) / 1e9;
    
    NSFileManager *fm = [NSFileManager defaultManager];
    NSString *bundlePath = bundleURL.path;
    NSArray *bundleFiles = [fm contentsOfDirectoryAtPath:bundlePath error:nil];
    if (!bundleFiles || bundleFiles.count == 0) {
        res.statusMessage = [NSString stringWithFormat:@"Bundle directory empty or not accessible: %@", bundlePath];
        return res;
    }
    
    NSString *bcMlir = nil;
    NSString *regionKey = nil;
    for (NSString *f in bundleFiles) {
        if ([f hasSuffix:@".bc.mlir"]) {
            bcMlir = f;
            regionKey = [f substringToIndex:(f.length - @".bc.mlir".length)];
            break;
        }
    }
    if (!regionKey) {
        for (NSString *f in bundleFiles) {
            if ([f hasSuffix:@".mil"] || [f hasSuffix:@".hwx"]) {
                bcMlir = f;
                regionKey = [[f stringByDeletingPathExtension] stringByDeletingPathExtension];
                break;
            }
        }
    }
    if (!regionKey) {
        res.statusMessage = [NSString stringWithFormat:@"Could not locate .bc.mlir ANECIR in bundle: %@", bundlePath];
        return res;
    }
    
    res.modelFormat = @"ANECIR";
    
    NSString *compilerOptionsFile = [NSString stringWithFormat:@"compiler_options_%@.plist", regionKey];
    if (![bundleFiles containsObject:compilerOptionsFile]) {
        compilerOptionsFile = @"compiler_options.plist";
        if (![bundleFiles containsObject:compilerOptionsFile]) {
            for (NSString *f in bundleFiles) {
                if ([f hasPrefix:@"compiler_options"] && [f hasSuffix:@".plist"]) {
                    compilerOptionsFile = f;
                    break;
                }
            }
        }
    }
    
    _ANEClient *client = [_ANEClient sharedConnection];
    if (!client) {
        res.statusMessage = @"Failed to acquire _ANEClient connection.";
        return res;
    }
    
    [self unloadActiveModel];
    
    NSString *effectiveArch = nil;
    if ([_ANEDeviceInfo respondsToSelector:@selector(aneArchitectureType)]) {
        effectiveArch = [_ANEDeviceInfo aneArchitectureType];
    }
    if (!effectiveArch || effectiveArch.length == 0) {
        effectiveArch = @"h16g";
    }
    
    _ANEModel *model = nil;
    if ([_ANEModel respondsToSelector:@selector(modelAtURL:key:mpsConstants:)]) {
        model = [_ANEModel modelAtURL:bundleURL key:regionKey mpsConstants:@"constants"];
    } else {
        model = [_ANEModel modelAtURL:bundleURL key:regionKey];
    }
    
    if (!model) {
        res.statusMessage = @"Failed to initialize _ANEModel.";
        return res;
    }
    
    NSDictionary *loadOpts = @{
        @"kANEFModelType": @"kANEFModelANECIR",
        @"kANEFCompilerOptionsFilenameKey": compilerOptionsFile,
        @"kANEFNetPlistFilenameKey": bcMlir,
        @"kANEFTargetArchitectureKey": effectiveArch,
        @"kANEFPerformanceStatsMask": @(15),
        @"kANEFRetainModelsWithoutSourceURL": @1
    };
    
    NSError *loadErr = nil;
    BOOL loadOk = [client loadModel:model options:loadOpts qos:25 error:&loadErr];
    if (!loadOk) {
        res.statusMessage = [NSString stringWithFormat:@"_ANEClient loadModel failed: %@", loadErr.localizedDescription ?: @"Unknown error"];
        return res;
    }
    
    // Inspect NetworkStatusList for exact tensor buffer requirements
    NSDictionary *attrs = model.modelAttributes;
    NSArray *netList = (attrs && [attrs isKindOfClass:[NSDictionary class]]) ? attrs[@"NetworkStatusList"] : nil;
    
    NSArray *inputs = @[];
    NSArray *outputs = @[];
    if (netList && netList.count > 0) {
        NSDictionary *mainNet = netList[0];
        inputs = mainNet[@"LiveInputList"] ?: @[];
        outputs = mainNet[@"LiveOutputList"] ?: @[];
    }
    
    NSMutableArray<_ANEIOSurfaceObject *> *inObjects = [NSMutableArray array];
    NSMutableArray<NSNumber *> *inIndices = [NSMutableArray array];
    NSMutableArray *inSurfs = [NSMutableArray array];
    
    uint32_t inIdx = 0;
    for (NSDictionary *t in inputs) {
        uint64_t bStride = [t[@"BatchStride"] unsignedLongLongValue] ?: 4096;
        uint64_t batches = [t[@"Batches"] unsignedLongLongValue] ?: 1;
        size_t bytes = (size_t)(bStride * batches);
        IOSurfaceRef surf = createIOSurface(bytes);
        fillIOSurfaceNonZero(surf, bytes, dataType);
        [inSurfs addObject:(__bridge id)surf];
        _ANEIOSurfaceObject *obj = [_ANEIOSurfaceObject objectWithIOSurface:surf];
        [inObjects addObject:obj];
        [inIndices addObject:@(inIdx++)];
        CFRelease(surf);
    }
    
    // Fallback if model has no declared live inputs
    if (inObjects.count == 0) {
        size_t inElementSize = (dataType == MPSDataTypeFloat16) ? 2 : 1;
        size_t inBytes = B * H * W * Ci * inElementSize;
        if (inBytes < 0x4000) inBytes = 0x4000;
        IOSurfaceRef surf = createIOSurface(inBytes);
        fillIOSurfaceNonZero(surf, inBytes, dataType);
        [inSurfs addObject:(__bridge id)surf];
        _ANEIOSurfaceObject *obj = [_ANEIOSurfaceObject objectWithIOSurface:surf];
        [inObjects addObject:obj];
        [inIndices addObject:@(0)];
        CFRelease(surf);
    }
    
    NSMutableArray<_ANEIOSurfaceObject *> *outObjects = [NSMutableArray array];
    NSMutableArray<NSNumber *> *outIndices = [NSMutableArray array];
    NSMutableArray *outSurfs = [NSMutableArray array];
    
    uint32_t outIdx = 0;
    for (NSDictionary *t in outputs) {
        uint64_t bStride = [t[@"BatchStride"] unsignedLongLongValue] ?: 4096;
        uint64_t batches = [t[@"Batches"] unsignedLongLongValue] ?: 1;
        size_t bytes = (size_t)(bStride * batches);
        IOSurfaceRef surf = createIOSurface(bytes);
        [outSurfs addObject:(__bridge id)surf];
        _ANEIOSurfaceObject *obj = [_ANEIOSurfaceObject objectWithIOSurface:surf];
        [outObjects addObject:obj];
        [outIndices addObject:@(outIdx++)];
        CFRelease(surf);
    }
    
    if (outObjects.count == 0) {
        size_t outElementSize = (dataType == MPSDataTypeFloat16) ? 2 : 1;
        size_t outBytes = B * H * W * Co * outElementSize;
        if (outBytes < 0x4000) outBytes = 0x4000;
        IOSurfaceRef surf = createIOSurface(outBytes);
        [outSurfs addObject:(__bridge id)surf];
        _ANEIOSurfaceObject *obj = [_ANEIOSurfaceObject objectWithIOSurface:surf];
        [outObjects addObject:obj];
        [outIndices addObject:@(0)];
        CFRelease(surf);
    }
    
    IOSurfaceRef pmuSurf = createIOSurface(0x4000);
    _ANEIOSurfaceObject *pmuObj = [_ANEIOSurfaceObject objectWithIOSurface:pmuSurf];
    _ANEPerformanceStatsIOSurface *pmuStatsSurf = nil;
    if ([_ANEPerformanceStatsIOSurface respondsToSelector:@selector(objectWithIOSurface:statType:)]) {
        pmuStatsSurf = [_ANEPerformanceStatsIOSurface objectWithIOSurface:pmuObj statType:2];
    } else {
        pmuStatsSurf = [[_ANEPerformanceStatsIOSurface alloc] initWithIOSurface:pmuObj statType:2];
    }
    CFRelease(pmuSurf);
    
    _ANERequest *req = [_ANERequest requestWithInputs:inObjects
                                         inputIndices:inIndices
                                              outputs:outObjects
                                        outputIndices:outIndices
                                            perfStats:@[pmuStatsSurf]
                                       procedureIndex:@0];
    
    if (!req) {
        res.statusMessage = @"Failed to construct _ANERequest with perfStats.";
        [client unloadModel:model options:loadOpts qos:25 error:nil];
        return res;
    }
    
    NSDictionary *evalOpts = @{
        @"kANEFPerformanceStatsMask": @(15),
        @"enableProfiling": @YES
    };
    
    // Warm-up iteration & baseline register latch
    NSError *evalErr = nil;
    uint64_t tW0 = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
    BOOL warmOk = [client evaluateWithModel:model options:evalOpts request:req qos:25 error:&evalErr];
    res.warmupMs = (double)(clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tW0) / 1e6;
    
    if (!warmOk) {
        res.statusMessage = [NSString stringWithFormat:@"_ANEClient evaluate failed on warmup: %@", evalErr.localizedDescription ?: @"Unknown error"];
        [client unloadModel:model options:loadOpts qos:25 error:nil];
        return res;
    }
    
    uint64_t initRegs[29] = {0};
    _ANEPerformanceStats *warmPerfStats = req.perfStats;
    NSData *d0 = warmPerfStats.perfCounterData;
    if (d0 && d0.length >= sizeof(uint64_t)) {
        size_t copyBytes = MIN(d0.length, sizeof(initRegs));
        memcpy(initRegs, d0.bytes, copyBytes);
    }
    
    NSUInteger numIters = (iterations > 0) ? iterations : 1;
    NSMutableArray<NSNumber *> *latencies = [NSMutableArray arrayWithCapacity:numIters];
    uint64_t totalNs = 0;
    
    for (NSUInteger i = 0; i < numIters; i++) {
        uint64_t t0 = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
        BOOL ok = [client evaluateWithModel:model options:evalOpts request:req qos:25 error:&evalErr];
        uint64_t dt = clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - t0;
        totalNs += dt;
        [latencies addObject:@((double)dt / 1e6)];
        if (!ok) break;
    }
    
    res.latencies = latencies;
    res.meanLatencyMs = (double)(totalNs / numIters) / 1e6;
    if (res.meanLatencyMs > 0.0) {
        res.throughputFps = 1000.0 / res.meanLatencyMs;
    }
    
    // Final registers & stats
    _ANEPerformanceStats *finalPerfStats = req.perfStats;
    if (finalPerfStats && [finalPerfStats respondsToSelector:@selector(hwExecutionTime)]) {
        res.hwExecutionTimeNs = finalPerfStats.hwExecutionTime;
        res.hwExecutionTimeMs = (double)res.hwExecutionTimeNs / 1e6;
    }
    
    uint64_t finalRegs[29] = {0};
    NSData *d1 = finalPerfStats.perfCounterData;
    if (d1) {
        res.perfCounterBytes = d1.length;
    }
    if (d1 && d1.length >= sizeof(uint64_t)) {
        size_t copyBytes = MIN(d1.length, sizeof(finalRegs));
        memcpy(finalRegs, d1.bytes, copyBytes);
    }
    
    NSMutableDictionary<NSString *, NSNumber *> *finalCounters = [NSMutableDictionary dictionary];
    NSMutableDictionary<NSString *, NSNumber *> *initCounters = [NSMutableDictionary dictionary];
    NSMutableDictionary<NSString *, NSNumber *> *deltasTotal = [NSMutableDictionary dictionary];
    NSMutableDictionary<NSString *, NSNumber *> *deltasPerIter = [NSMutableDictionary dictionary];
    
    if ([finalPerfStats respondsToSelector:@selector(performanceCounters)]) {
        NSDictionary *c = finalPerfStats.performanceCounters;
        if (c) {
            for (id rawK in c) {
                NSString *k = [rawK description];
                if ([k hasSuffix:@":"]) {
                    k = [k substringToIndex:k.length - 1];
                }
                finalCounters[k] = c[rawK];
            }
        }
    }
    if ([warmPerfStats respondsToSelector:@selector(performanceCounters)]) {
        NSDictionary *c = warmPerfStats.performanceCounters;
        if (c) {
            for (id rawK in c) {
                NSString *k = [rawK description];
                if ([k hasSuffix:@":"]) {
                    k = [k substringToIndex:k.length - 1];
                }
                initCounters[k] = c[rawK];
            }
        }
    }
    
    size_t numRegs = (d1 && d1.length >= sizeof(uint64_t)) ? MIN(d1.length / sizeof(uint64_t), (size_t)29) : 24;
    for (size_t i = 0; i < numRegs; i++) {
        NSString *name = [self nameForPerfCounter:(int32_t)i];
        if (!finalCounters[name]) {
            finalCounters[name] = @(finalRegs[i]);
        }
        if (!initCounters[name]) {
            initCounters[name] = @(initRegs[i]);
        }
    }
    
    for (NSString *name in finalCounters) {
        uint64_t fVal = [finalCounters[name] unsignedLongLongValue];
        uint64_t iVal = initCounters[name] ? [initCounters[name] unsignedLongLongValue] : 0;
        uint64_t dTotal = (fVal >= iVal) ? (fVal - iVal) : fVal;
        deltasTotal[name] = @(dTotal);
        deltasPerIter[name] = @(numIters > 0 ? (dTotal / numIters) : dTotal);
    }
    
    res.performanceCounters = finalCounters;
    res.performanceCounterInitial = initCounters;
    res.performanceCounterDeltas = deltasTotal;
    res.performanceCounterDeltasPerIter = deltasPerIter;
    
    // Categorize
    res.performanceCountersCategorized = categorizeCountersDict(finalCounters);
    res.performanceCounterDeltasCategorized = categorizeCountersDict(deltasPerIter);
    res.performanceCounterTotalDeltasCategorized = categorizeCountersDict(deltasTotal);
    
    // Throughput and silicon capacity
    uint64_t nominal = [deltasPerIter[@"kANE_NE_NOMINAL_CYCLES"] unsignedLongLongValue];
    if (nominal > 0 && totalMacs > 0) {
        res.macsPerCoreCycle = totalMacs / (double)nominal;
        res.chipMacsPerCycle = res.macsPerCoreCycle * 16.0;
        res.aluSaturationFp16 = (res.macsPerCoreCycle / 256.0) * 100.0;
        res.aluSaturationInt8 = (res.macsPerCoreCycle / 512.0) * 100.0;
    }
    if (res.meanLatencyMs > 0 && nominal > 0) {
        res.effectiveCoreClockGhz = (double)nominal / (res.meanLatencyMs * 1e6 * 16.0);
        res.effectiveClockGhz = res.effectiveCoreClockGhz;
    }
    if (res.meanLatencyMs > 0 && totalMacs > 0) {
        res.topsRealized = (totalMacs * 2.0 / 1e12) / (res.meanLatencyMs / 1000.0);
    }
    
    res.success = YES;
    res.statusMessage = @"Success";
    
    // Unload model and release resources
    [client unloadModel:model options:loadOpts qos:25 error:nil];
    
    return res;
}

+ (ANEVerificationResult *)profileANEBundleAtURL:(NSURL *)url
                                      iterations:(NSUInteger)iterations
                                       totalMacs:(double)totalMacs {
    return [self profileANECIRBundleAtURL:url
                                    batch:1
                                   height:256
                                    width:256
                               inChannels:128
                              outChannels:128
                                 dataType:MPSDataTypeFloat16
                               iterations:iterations
                                totalMacs:totalMacs];
}

@end
