//
// measure_ane_pmu.m
// High-Throughput MPSGraph Convolution Capacity Benchmark with Physical Silicon PMU Telemetry
//
// Based on and adapted from the Apple Neural Engine hardware profiler research in:
//   https://github.com/freedomtan/ane_pmu_profiler/
//
// Features:
//   1. MPSGraph Conv2D benchmark for FP16, INT8, and QDQ (Quantize-Dequantize-Quantize)
//   2. Automatic MPSGraphPackage serialization (--save-package) with self-contained ANE bundles
//   3. Hardware PMU Telemetry across 29 registers (Compute Cycles, L2PE Cycles, Memory Stalls, DMA, Energy)
//   4. GPU baseline comparison + ANE hardware cycle efficiency & TOPS calculation
//

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#import <IOSurface/IOSurface.h>
#import <IOKit/IOKitLib.h>
#import <Security/Security.h>
#import <sys/sysctl.h>
#import <time.h>
#import <objc/runtime.h>

extern SecTaskRef SecTaskCreateFromSelf(CFAllocatorRef allocator);
extern CFTypeRef SecTaskCopyValueForEntitlement(SecTaskRef task, CFStringRef entitlement, CFErrorRef *error);

#define kANEFModelTypeKey                     @"kANEFModelType"
#define kANEFModelANECIRValue                 @"kANEFModelANECIR"
#define kANEFPerformanceStatsMaskKey          @"kANEFPerformanceStatsMask"
#define kANEFNetPlistFilenameKey              @"kANEFNetPlistFilenameKey"
#define kANEFCompilerOptionsFilenameKey       @"kANEFCompilerOptionsFilenameKey"
#define kANEFRetainModelsWithoutSourceURLKey  @"kANEFRetainModelsWithoutSourceURLKey"
#define kANEFTargetArchitectureKey            @"kANEFTargetArchitectureKey"

@interface MPSGraphCompilationDescriptor (Private)
@property (nonatomic, assign) unsigned long long preferredDevice;
@end

@interface MPSGraphExecutable (Package)
- (void)serializeToMPSGraphPackageAtURL:(NSURL *)url descriptor:(id)desc;
@end

// Private AppleNeuralEngine declarations
@interface _ANEIOSurfaceObject : NSObject
+ (instancetype)objectWithIOSurface:(IOSurfaceRef)surface;
@end

@interface _ANEPerformanceStatsIOSurface : NSObject
+ (instancetype)objectWithIOSurface:(_ANEIOSurfaceObject *)ioSurface statType:(int)statType;
@end

@interface _ANEPerformanceStats : NSObject
@property (nonatomic, readonly) NSData *perfCounterData;
@property (nonatomic, readonly) unsigned long long hwExecutionTime;
- (NSString *)stringForPerfCounter:(int)index;
@end

@interface _ANERequest : NSObject
+ (instancetype)requestWithInputs:(NSArray *)inputs
                     inputIndices:(NSArray *)inputIndices
                          outputs:(NSArray *)outputs
                    outputIndices:(NSArray *)outputIndices
                        perfStats:(NSArray *)perfStats
                   procedureIndex:(NSNumber *)procedureIndex;
@property (nonatomic, readonly) _ANEPerformanceStats *perfStats;
@end

@interface _ANEModel : NSObject
+ (instancetype)modelAtURL:(NSURL *)url key:(NSString *)key mpsConstants:(NSString *)constants;
@property (nonatomic, readonly) NSDictionary *modelAttributes;
@end

@interface _ANEClient : NSObject
+ (instancetype)sharedConnection;
- (BOOL)loadModel:(_ANEModel *)model options:(NSDictionary *)options qos:(unsigned int)qos error:(NSError **)error;
- (BOOL)unloadModel:(_ANEModel *)model options:(NSDictionary *)options qos:(unsigned int)qos error:(NSError **)error;
- (BOOL)evaluateWithModel:(_ANEModel *)model options:(NSDictionary *)options request:(_ANERequest *)request qos:(unsigned int)qos error:(NSError **)error;
@end

// Benchmark Configuration
typedef struct {
    NSUInteger B;          // Batch
    NSUInteger H;          // Height
    NSUInteger W;          // Width
    NSUInteger Ci;         // In channels
    NSUInteger Co;         // Out channels
    NSUInteger K;          // Kernel size
    NSUInteger L;          // Number of chained conv layers
    int iterations;        // Number of test iterations
    BOOL runGPU;           // Run GPU comparison
    BOOL runANE;           // Run ANE via MPSGraph
    BOOL profilePMU;       // Run Physical Silicon PMU profile
    BOOL verbosePMU;       // Print all 29 raw registers
    BOOL savePackage;      // Serialize .mpsgraphpackage
    NSString *packageDir;  // Target package output dir
    NSString *variant;     // "all", "fp16", "int8", "qdq"
} BenchConfig;

// Results container
typedef struct {
    double gpuTimeMs;
    double gpuTops;
    double aneMpsTimeMs;
    double aneMpsTops;
    double anePmuTimeMs;
    double anePmuTops;
    uint64_t computeCycles;
    uint64_t l2peCycles;
    uint64_t inputStallCycles;
    uint64_t outputStallCycles;
    uint64_t dmaRwBytes;
    uint64_t dpeEnergy;
    double clockGhz;
} BenchResult;

// --- Helper: Query System ANE Architecture ---
static NSString *querySystemANEArchitecture(void) {
    CFMutableDictionaryRef matching = IOServiceMatching("H11ANEIn");
    io_service_t service = IOServiceGetMatchingService(kIOMainPortDefault, matching);
    if (!service) {
        matching = IOServiceMatching("AppleH16ANEInterface");
        service = IOServiceGetMatchingService(kIOMainPortDefault, matching);
    }
    NSString *archStr = @"h16g";
    if (service) {
        CFMutableDictionaryRef props = NULL;
        if (IORegistryEntryCreateCFProperties(service, &props, kCFAllocatorDefault, 0) == KERN_SUCCESS && props) {
            NSDictionary *dict = (__bridge NSDictionary *)props;
            NSDictionary *devProps = dict[@"DeviceProperties"];
            if (devProps && devProps[@"ANEDevicePropertyTypeANEArchitectureTypeStr"]) {
                archStr = [devProps[@"ANEDevicePropertyTypeANEArchitectureTypeStr"] copy];
            }
            CFRelease(props);
        }
        IOObjectRelease(service);
    }
    return archStr;
}

// --- Helper: Check Silicon PMU Entitlement & Driver Gate ---
static BOOL checkPMUGate(void) {
    BOOL hasEntitlement = NO;
    SecTaskRef task = SecTaskCreateFromSelf(NULL);
    if (task) {
        CFErrorRef err = NULL;
        CFTypeRef val = SecTaskCopyValueForEntitlement(task, CFSTR("com.apple.ane.hardware-counters"), &err);
        if (val) {
            if (CFGetTypeID(val) == CFBooleanGetTypeID()) {
                hasEntitlement = CFBooleanGetValue((CFBooleanRef)val);
            }
            CFRelease(val);
        }
        CFRelease(task);
    }

    char bootArgs[1024] = {0};
    size_t size = sizeof(bootArgs);
    BOOL hasAneDebug = NO;
    if (sysctlbyname("kern.bootargs", bootArgs, &size, NULL, 0) == 0) {
        if (strstr(bootArgs, "anedebug") != NULL) {
            hasAneDebug = YES;
        }
    }
    return hasEntitlement || hasAneDebug;
}

// --- Helper: Create IOSurface ---
static IOSurfaceRef createIOSurface(size_t allocSize) {
    NSDictionary *props = @{
        (id)kIOSurfaceWidth: @(allocSize),
        (id)kIOSurfaceHeight: @1,
        (id)kIOSurfaceBytesPerElement: @1,
        (id)kIOSurfaceBytesPerRow: @(allocSize),
        (id)kIOSurfaceAllocSize: @(allocSize)
    };
    return IOSurfaceCreate((CFDictionaryRef)props);
}

// --- Helper: Find Latest ANE Temp Directory ---
static NSSet<NSString *> *getExistingANETempDirs(void) {
    NSFileManager *fm = [NSFileManager defaultManager];
    NSString *tmpBase = [NSTemporaryDirectory() stringByAppendingPathComponent:@"com.apple.MetalPerformanceShadersGraph"];
    NSArray *dirs = [fm contentsOfDirectoryAtPath:tmpBase error:nil];
    return dirs ? [NSSet setWithArray:dirs] : [NSSet set];
}

static NSString *findANETempDirectory(NSSet<NSString *> *beforeDirs) {
    NSFileManager *fm = [NSFileManager defaultManager];
    NSString *tmpBase = [NSTemporaryDirectory() stringByAppendingPathComponent:@"com.apple.MetalPerformanceShadersGraph"];
    NSArray *dirs = [fm contentsOfDirectoryAtPath:tmpBase error:nil];
    
    // 1. Look for directories created during this compilation pass
    for (NSString *d in dirs) {
        if (![beforeDirs containsObject:d]) {
            NSString *full = [tmpBase stringByAppendingPathComponent:d];
            NSArray *subfiles = [fm contentsOfDirectoryAtPath:full error:nil];
            for (NSString *sf in subfiles) {
                if ([sf hasSuffix:@".bc.mlir"]) {
                    return full;
                }
            }
        }
    }

    // 2. Fallback: look for latest modified directory containing .bc.mlir
    NSString *bestDir = nil;
    NSDate *bestDate = [NSDate distantPast];
    for (NSString *d in dirs) {
        NSString *full = [tmpBase stringByAppendingPathComponent:d];
        NSDictionary *attrs = [fm attributesOfItemAtPath:full error:nil];
        NSDate *mod = attrs[NSFileModificationDate];
        if (mod && [mod compare:bestDate] == NSOrderedDescending) {
            NSArray *subfiles = [fm contentsOfDirectoryAtPath:full error:nil];
            for (NSString *sf in subfiles) {
                if ([sf hasSuffix:@".bc.mlir"]) {
                    bestDate = mod;
                    bestDir = full;
                    break;
                }
            }
        }
    }
    return bestDir;
}

// --- Helper: Copy ANE Bundle to Destination ---
static NSString *copyANEBundle(NSString *srcDir, NSString *destDir) {
    if (!srcDir) return nil;
    NSFileManager *fm = [NSFileManager defaultManager];
    NSString *aneBundleDir = [destDir stringByAppendingPathComponent:@"ane_bundle"];
    [fm removeItemAtPath:aneBundleDir error:nil];
    [fm createDirectoryAtPath:aneBundleDir withIntermediateDirectories:YES attributes:nil error:nil];

    NSArray *files = [fm contentsOfDirectoryAtPath:srcDir error:nil];
    for (NSString *f in files) {
        NSString *src = [srcDir stringByAppendingPathComponent:f];
        NSString *dst = [aneBundleDir stringByAppendingPathComponent:f];
        [fm copyItemAtPath:src toPath:dst error:nil];
    }
    return aneBundleDir;
}

// --- Benchmark Runner for a Single Variant ---
static BenchResult benchmarkVariant(const BenchConfig *cfg, id<MTLDevice> dev,
                                    NSString *variantName, MPSDataType dataType, BOOL isQDQ) {
    BenchResult res = {0};
    NSFileManager *fm = [NSFileManager defaultManager];

    NSUInteger B = cfg->B, H = cfg->H, W = cfg->W, Ci = cfg->Ci, Co = cfg->Co, K = cfg->K, L = cfg->L;
    double totalOps = 2.0 * B * H * W * Ci * Co * K * K * L;

    NSArray *inShape = @[@(B), @(Ci), @(H), @(W)];
    NSArray *wShape = @[@(Co), @(Ci), @(K), @(K)];

    printf("\n========================================================================================================\n");
    printf("🚀 BENCHMARK: %s | Dimensions: [%lu, %lu, %lu, %lu] -> Conv %lux%lu (L=%lu) -> [%lu, %lu, %lu, %lu]\n",
           variantName.UTF8String, B, Ci, H, W, K, K, L, B, Co, H, W);
    printf("   Workload: %.2f GOPs (%.4f TOPs) per inference pass\n", totalOps / 1e9, totalOps / 1e12);
    printf("========================================================================================================\n");

    // --- 1. Construct MPSGraph ---
    MPSGraph *graph = [MPSGraph new];
    MPSGraphTensor *input = [graph placeholderWithShape:inShape dataType:dataType name:@"input_0"];
    MPSGraphTensor *cur = input;

    // Weights allocation
    NSUInteger weightElementSize = (isQDQ || dataType == MPSDataTypeFloat16) ? 2 : 1;
    MPSDataType weightType = (isQDQ) ? MPSDataTypeFloat16 : dataType;
    NSMutableData *wData = [NSMutableData dataWithLength:Co * Ci * K * K * weightElementSize];
    MPSGraphTensor *w = [graph constantWithData:wData shape:wShape dataType:weightType];

    MPSGraphConvolution2DOpDescriptor *convDesc = [MPSGraphConvolution2DOpDescriptor
        descriptorWithStrideInX:1
                      strideInY:1
                dilationRateInX:1
                dilationRateInY:1
                         groups:1
                   paddingStyle:MPSGraphPaddingStyleTF_SAME
                     dataLayout:MPSGraphTensorNamedDataLayoutNCHW
                  weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];

    if (isQDQ) {
        MPSGraphTensor *scale = [graph constantWithScalar:1.0 dataType:MPSDataTypeFloat32];
        MPSGraphTensor *zp = [graph constantWithScalar:0.0 dataType:MPSDataTypeInt32];
        for (int i = 0; i < L; i++) {
            cur = [graph dequantizeTensor:cur scaleTensor:scale zeroPointTensor:zp dataType:MPSDataTypeFloat16 axis:1 name:nil];
            cur = [graph convolution2DWithSourceTensor:cur weightsTensor:w descriptor:convDesc name:nil];
            cur = [graph quantizeTensor:cur scaleTensor:scale zeroPointTensor:zp dataType:MPSDataTypeInt8 axis:1 name:nil];
        }
    } else {
        for (int i = 0; i < L; i++) {
            cur = [graph convolution2DWithSourceTensor:cur weightsTensor:w descriptor:convDesc name:nil];
            if (dataType == MPSDataTypeInt8) {
                MPSGraphTensor *fp = [graph castTensor:cur toType:MPSDataTypeFloat16 name:@"dequant"];
                cur = [graph castTensor:fp toType:MPSDataTypeInt8 name:@"requant"];
            }
        }
    }

    NSDictionary *feeds = @{ input: [[MPSGraphShapedType alloc] initWithShape:inShape dataType:dataType] };

    // --- 2. GPU Benchmark (Optional) ---
    if (cfg->runGPU && (dataType == MPSDataTypeFloat16 || isQDQ)) {
        printf("🖥️  [GPU Benchmark] Compiling and executing on Metal GPU...\n");
        MPSGraphDevice *gpuDev = [MPSGraphDevice deviceWithMTLDevice:dev];
        MPSGraphCompilationDescriptor *gpuCd = [MPSGraphCompilationDescriptor new];
        gpuCd.optimizationLevel = MPSGraphOptimizationLevel0;

        MPSGraphExecutable *gpuExe = [graph compileWithDevice:gpuDev feeds:feeds targetTensors:@[cur] targetOperations:nil compilationDescriptor:gpuCd];
        if (gpuExe) {
            id<MTLBuffer> iBuf = [dev newBufferWithLength:B * H * W * Ci * ((dataType == MPSDataTypeFloat16)?2:1) options:0];
            MPSGraphTensorData *iData = [[MPSGraphTensorData alloc] initWithMTLBuffer:iBuf shape:inShape dataType:dataType];
            id<MTLCommandQueue> q = [dev newCommandQueue];
            MPSGraphExecutableExecutionDescriptor *ed = [MPSGraphExecutableExecutionDescriptor new];
            ed.waitUntilCompleted = YES;

            // Warm-up
            [gpuExe runWithMTLCommandQueue:q inputsArray:@[iData] resultsArray:nil executionDescriptor:ed];

            struct timespec t0, t1;
            clock_gettime(CLOCK_MONOTONIC, &t0);
            for (int i = 0; i < cfg->iterations; i++) {
                [gpuExe runWithMTLCommandQueue:q inputsArray:@[iData] resultsArray:nil executionDescriptor:ed];
            }
            clock_gettime(CLOCK_MONOTONIC, &t1);

            double duration = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
            res.gpuTimeMs = (duration / cfg->iterations) * 1000.0;
            res.gpuTops = (totalOps / 1e12) / (res.gpuTimeMs / 1000.0);
            printf("   GPU Result: %6.2f ms | %6.2f TOPS\n", res.gpuTimeMs, res.gpuTops);
        }
    }

    // --- 3. ANE Compilation via MPSGraph ---
    printf("⚡️ [ANE Compilation] Specializing graph for Apple Neural Engine...\n");
    NSSet<NSString *> *beforeDirs = getExistingANETempDirs();
    MPSGraphDevice *aneDev = [MPSGraphDevice deviceWithMTLDevice:dev];
    MPSGraphCompilationDescriptor *aneCd = [MPSGraphCompilationDescriptor new];
    aneCd.optimizationLevel = MPSGraphOptimizationLevel1;
    aneCd.preferredDevice = 2; // MPSGraphDeviceTypeANE

    uint64_t tComp0 = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
    MPSGraphExecutable *aneExe = [graph compileWithDevice:aneDev feeds:feeds targetTensors:@[cur] targetOperations:nil compilationDescriptor:aneCd];
    uint64_t dtComp = clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tComp0;

    if (!aneExe) {
        printf("❌ Failed to compile MPSGraph for ANE!\n");
        return res;
    }
    printf("   Compilation completed in %.2f ms\n", (double)dtComp / 1e6);

    // Warm-up dispatch via Metal Queue (triggers ANE region emission to temp storage)
    id<MTLBuffer> mtlInBuf = [dev newBufferWithLength:B * H * W * Ci * ((dataType == MPSDataTypeFloat16)?2:1) options:0];
    MPSGraphTensorData *mtlInData = [[MPSGraphTensorData alloc] initWithMTLBuffer:mtlInBuf shape:inShape dataType:dataType];
    id<MTLCommandQueue> aneQueue = [dev newCommandQueue];
    MPSGraphExecutableExecutionDescriptor *aneEd = [MPSGraphExecutableExecutionDescriptor new];
    aneEd.waitUntilCompleted = YES;

    [aneExe runWithMTLCommandQueue:aneQueue inputsArray:@[mtlInData] resultsArray:nil executionDescriptor:aneEd];

    // Find the ANE temp bundle created during compilation & dispatch
    NSString *latestTmpDir = findANETempDirectory(beforeDirs);

    // --- 4. Serialize .mpsgraphpackage (If Requested) ---
    NSString *packagePath = nil;
    NSString *activeBundlePath = latestTmpDir;

    if (cfg->savePackage) {
        [fm createDirectoryAtPath:cfg->packageDir withIntermediateDirectories:YES attributes:nil error:nil];
        packagePath = [cfg->packageDir stringByAppendingPathComponent:[NSString stringWithFormat:@"%@.mpsgraphpackage", variantName]];
        [fm removeItemAtPath:packagePath error:nil];

        id sDesc = [NSClassFromString(@"MPSGraphExecutableSerializationDescriptor") new];
        [aneExe serializeToMPSGraphPackageAtURL:[NSURL fileURLWithPath:packagePath] descriptor:sDesc];

        // Copy ANE bundle inside the package
        if (latestTmpDir) {
            NSString *copied = copyANEBundle(latestTmpDir, packagePath);
            if (copied) {
                activeBundlePath = copied;
            }
        }
        printf("📦 [Package Export] Saved self-contained package:\n   Path: %s\n", packagePath.UTF8String);
        if (activeBundlePath) {
            printf("   Bundled ANE Microcode: %s\n", activeBundlePath.UTF8String);
        }
    }

    // --- 5. ANE MPSGraph Wall-Clock Benchmark ---
    if (cfg->runANE) {
        printf("🏃 [MPSGraph ANE Runner] Executing %d iterations...\n", cfg->iterations);
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        for (int i = 0; i < cfg->iterations; i++) {
            [aneExe runWithMTLCommandQueue:aneQueue inputsArray:@[mtlInData] resultsArray:nil executionDescriptor:aneEd];
        }
        clock_gettime(CLOCK_MONOTONIC, &t1);

        double duration = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
        res.aneMpsTimeMs = (duration / cfg->iterations) * 1000.0;
        res.aneMpsTops = (totalOps / 1e12) / (res.aneMpsTimeMs / 1000.0);
        printf("   MPSGraph ANE Result: %6.2f ms | %6.2f TOPS (Framerate: %6.1f FPS)\n",
               res.aneMpsTimeMs, res.aneMpsTops, 1000.0 / res.aneMpsTimeMs);
    }

    // --- 6. Physical Silicon PMU Hardware Profiling ---
    if (cfg->profilePMU && activeBundlePath) {
        printf("🔬 [Silicon PMU Profiler] Initializing hardware counters on physical ANE...\n");

        NSArray *bundleFiles = [fm contentsOfDirectoryAtPath:activeBundlePath error:nil];
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
            printf("⚠️  Could not locate .bc.mlir ANECIR in bundle. Skipping PMU.\n");
            return res;
        }

        NSString *targetArch = querySystemANEArchitecture();
        _ANEClient *client = [_ANEClient sharedConnection];
        _ANEModel *model = [_ANEModel modelAtURL:[NSURL fileURLWithPath:activeBundlePath] key:regionKey mpsConstants:@"constants"];

        NSDictionary *loadOpts = @{
            kANEFModelTypeKey: kANEFModelANECIRValue,
            kANEFCompilerOptionsFilenameKey: [NSString stringWithFormat:@"compiler_options_%@.plist", regionKey],
            kANEFNetPlistFilenameKey: bcMlir,
            kANEFTargetArchitectureKey: targetArch,
            kANEFPerformanceStatsMaskKey: @(15),
            kANEFRetainModelsWithoutSourceURLKey: @1
        };

        NSError *loadErr = nil;
        BOOL loadOk = [client loadModel:model options:loadOpts qos:25 error:&loadErr];
        if (!loadOk) {
            printf("❌ Failed to load model directly into _ANEClient: %s\n", loadErr.localizedDescription.UTF8String ?: "Unknown");
            return res;
        }

        // Determine tensor element sizes
        size_t inElementSize = (dataType == MPSDataTypeFloat16) ? 2 : 1;
        size_t outElementSize = (dataType == MPSDataTypeFloat16) ? 2 : 1;
        if (isQDQ) {
            inElementSize = 1;  // Input is INT8
            outElementSize = 1; // Output is INT8
        }

        size_t inBytes = B * H * W * Ci * inElementSize;
        size_t outBytes = B * H * W * Co * outElementSize;

        IOSurfaceRef inSurf = createIOSurface(inBytes);
        IOSurfaceRef outSurf = createIOSurface(outBytes);
        IOSurfaceRef pmuSurf = createIOSurface(4096);

        _ANEIOSurfaceObject *inObj = [_ANEIOSurfaceObject objectWithIOSurface:inSurf];
        _ANEIOSurfaceObject *outObj = [_ANEIOSurfaceObject objectWithIOSurface:outSurf];
        _ANEIOSurfaceObject *pmuObj = [_ANEIOSurfaceObject objectWithIOSurface:pmuSurf];
        _ANEPerformanceStatsIOSurface *pmuStatsSurf = [_ANEPerformanceStatsIOSurface objectWithIOSurface:pmuObj statType:2];

        _ANERequest *req = [_ANERequest requestWithInputs:@[inObj] inputIndices:@[@0]
                                                  outputs:@[outObj] outputIndices:@[@0]
                                                perfStats:@[pmuStatsSurf] procedureIndex:@0];

        NSDictionary *evalOpts = @{
            kANEFPerformanceStatsMaskKey: @(15),
            @"enableProfiling": @YES
        };

        // Warm-up iteration & baseline latch
        NSError *evalErr = nil;
        [client evaluateWithModel:model options:evalOpts request:req qos:25 error:&evalErr];
        uint64_t initRegs[29] = {0};
        NSData *d0 = req.perfStats.perfCounterData;
        if (d0 && d0.length >= sizeof(initRegs)) {
            memcpy(initRegs, d0.bytes, sizeof(initRegs));
        }

        // PMU Benchmark Loop
        uint64_t totalNs = 0;
        for (int i = 0; i < cfg->iterations; i++) {
            uint64_t t0 = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
            [client evaluateWithModel:model options:evalOpts request:req qos:25 error:&evalErr];
            totalNs += clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - t0;
        }

        uint64_t finalRegs[29] = {0};
        NSData *d1 = req.perfStats.perfCounterData;
        if (d1 && d1.length >= sizeof(finalRegs)) {
            memcpy(finalRegs, d1.bytes, sizeof(finalRegs));
        }

        if (cfg->verbosePMU) {
            printf("\n🔍 DETAILED SILICON PMU REGISTERS (Per-Iteration Deltas):\n");
            _ANEPerformanceStats *pStats = req.perfStats;
            for (int i = 0; i < 29; i++) {
                NSString *name = [pStats stringForPerfCounter:i];
                uint64_t d = (finalRegs[i] >= initRegs[i]) ? (finalRegs[i] - initRegs[i]) / cfg->iterations : 0;
                printf("   [%02d] %-30s : %'16llu | init=%'16llu, final=%'16llu\n",
                       i, name.UTF8String ?: "kANE_UNKNOWN", d, initRegs[i], finalRegs[i]);
            }
            printf("\n");
        }

        res.anePmuTimeMs = (double)totalNs / (cfg->iterations * 1e6);
        res.anePmuTops = (totalOps / 1e12) / (res.anePmuTimeMs / 1000.0);

        res.computeCycles = (finalRegs[13] - initRegs[13]) / cfg->iterations;
        res.l2peCycles = (finalRegs[21] - initRegs[21]) / cfg->iterations;
        res.inputStallCycles = (finalRegs[14] - initRegs[14]) / cfg->iterations;
        res.outputStallCycles = (finalRegs[15] - initRegs[15]) / cfg->iterations;
        res.dmaRwBytes = (finalRegs[17] - initRegs[17]) / cfg->iterations;
        res.dpeEnergy = (finalRegs[19] - initRegs[19]) / cfg->iterations;

        uint64_t nominalCycles = (finalRegs[10] - initRegs[10]) / cfg->iterations;
        if (res.anePmuTimeMs > 0) {
            res.clockGhz = (double)nominalCycles / (res.anePmuTimeMs * 1e6 * 16.0); // 16 cores
        }

        printf("   ANE Silicon PMU Result:\n");
        printf("   • Latency (Driver Execution) : %6.3f ms (Framerate: %6.1f FPS)\n", res.anePmuTimeMs, 1000.0 / res.anePmuTimeMs);
        printf("   • Physical Realized TOPS     : %6.2f TOPS\n", res.anePmuTops);
        printf("   • Active NE Compute Cycles   : %'llu cycles/iter (kANE_NE_COMPUTE_CYCLES)\n", res.computeCycles);
        printf("   • Planar (L2PE) Cycles       : %'llu cycles/iter (kANE_L2PE_COMPUTE_CYCLES)\n", res.l2peCycles);
        printf("   • Input Operand Stalls       : %'llu cycles/iter (kANE_NE_INPUT_STALL_CYCLES)\n", res.inputStallCycles);
        printf("   • Output Writeback Stalls    : %'llu cycles/iter (kANE_NE_OUTPUT_STALL_CYCLES)\n", res.outputStallCycles);
        printf("   • Unified Memory DMA Traffic : %.2f MB/iter (kANE_DMA_READWRITE_BYTES)\n", (double)res.dmaRwBytes / (1024.0*1024.0));
        printf("   • DPE Energy Units           : %'llu units/iter (kANE_DPE_ENERGY)\n", res.dpeEnergy);

        double totalMacs = totalOps / 2.0;
        printf("   • Total MAC Operations       : %'.0f MACs\n", totalMacs);
        if (nominalCycles > 0) {
            double macsPerNominalCycle = totalMacs / (double)nominalCycles;
            printf("   • Throughput / Silicon Cycle : %.1f MACs / cycle (%.1f MACs / cycle / core across 16 cores)\n",
                   macsPerNominalCycle, macsPerNominalCycle / 16.0);
        }

        [client unloadModel:model options:@{} qos:25 error:nil];
        CFRelease(inSurf);
        CFRelease(outSurf);
        CFRelease(pmuSurf);
    }

    return res;
}

static void printUsage(const char *prog) {
    printf("Usage: %s [options]\n\n", prog);
    printf("Options:\n");
    printf("  --variant <type>      Benchmark variant: all, fp16, int8, qdq (default: all)\n");
    printf("  --batch <B>           Batch size (default: 1)\n");
    printf("  --size <H>            Spatial height & width (default: 256)\n");
    printf("  --channels <C>        Input & output channel depth (default: 128)\n");
    printf("  --layers <L>          Number of chained convolution layers (default: 20)\n");
    printf("  --iterations <N>      Number of benchmark iterations (default: 20)\n");
    printf("  --save-package <dir>  Serialize .mpsgraphpackage models to directory (default: ./packages)\n");
    printf("  --no-gpu              Skip GPU comparison benchmark\n");
    printf("  --no-pmu              Skip physical silicon PMU hardware profiling\n");
    printf("  --verbose-pmu         Print all 29 hardware PMU registers & deltas\n");
    printf("  -h, --help            Show this help message\n\n");
}

int main(int argc, char *argv[]) {
    @autoreleasepool {
        setlocale(LC_NUMERIC, "");
        NSBundle *aneBundle = [NSBundle bundleWithPath:@"/System/Library/PrivateFrameworks/AppleNeuralEngine.framework"];
        [aneBundle load];

        BenchConfig cfg;
        cfg.B = 1;
        cfg.H = 256;
        cfg.W = 256;
        cfg.Ci = 128;
        cfg.Co = 128;
        cfg.K = 3;
        cfg.L = 20;
        cfg.iterations = 20;
        cfg.runGPU = YES;
        cfg.runANE = YES;
        cfg.profilePMU = YES;
        cfg.verbosePMU = NO;
        cfg.savePackage = YES;
        cfg.packageDir = @"./packages";
        cfg.variant = @"all";

        for (int i = 1; i < argc; i++) {
            NSString *arg = [NSString stringWithUTF8String:argv[i]];
            if ([arg isEqualToString:@"--variant"] && i + 1 < argc) {
                cfg.variant = [NSString stringWithUTF8String:argv[++i]].lowercaseString;
            } else if ([arg isEqualToString:@"--batch"] && i + 1 < argc) {
                cfg.B = atoi(argv[++i]);
            } else if ([arg isEqualToString:@"--size"] && i + 1 < argc) {
                cfg.H = atoi(argv[++i]);
                cfg.W = cfg.H;
            } else if ([arg isEqualToString:@"--channels"] && i + 1 < argc) {
                cfg.Ci = atoi(argv[++i]);
                cfg.Co = cfg.Ci;
            } else if ([arg isEqualToString:@"--layers"] && i + 1 < argc) {
                cfg.L = atoi(argv[++i]);
            } else if ([arg isEqualToString:@"--iterations"] && i + 1 < argc) {
                cfg.iterations = atoi(argv[++i]);
            } else if ([arg isEqualToString:@"--save-package"]) {
                cfg.savePackage = YES;
                if (i + 1 < argc && argv[i+1][0] != '-') {
                    cfg.packageDir = [NSString stringWithUTF8String:argv[++i]];
                }
            } else if ([arg isEqualToString:@"--no-gpu"]) {
                cfg.runGPU = NO;
            } else if ([arg isEqualToString:@"--no-pmu"]) {
                cfg.profilePMU = NO;
            } else if ([arg isEqualToString:@"--verbose-pmu"]) {
                cfg.verbosePMU = YES;
            } else if ([arg isEqualToString:@"-h"] || [arg isEqualToString:@"--help"]) {
                printUsage(argv[0]);
                return 0;
            }
        }

        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) {
            fprintf(stderr, "❌ No Metal-capable GPU found!\n");
            return 1;
        }

        BOOL pmuGateUnlocked = checkPMUGate();
        NSString *archStr = querySystemANEArchitecture();

        printf("========================================================================================================\n");
        printf("🧠 APPLE NEURAL ENGINE (ANE) HIGH-THROUGHPUT CONVOLUTION BENCHMARK & PMU PROFILER\n");
        printf("========================================================================================================\n");
        printf("  • Host Device              : %s\n", dev.name.UTF8String);
        printf("  • Silicon Architecture     : Apple %s (16 Physical ANE Cores)\n", archStr.UTF8String);
        printf("  • Silicon PMU Access Gate  : %s\n", pmuGateUnlocked ? "UNLOCKED (Hardware PMU active)" : "GATED (Run with anedebug=1 or entitlement)");
        printf("  • Convolution Dimensions   : [%lu, %lu, %lu, %lu] | Kernel: %lux%lu | Layers: %lu\n", cfg.B, cfg.Ci, cfg.H, cfg.W, cfg.K, cfg.K, cfg.L);
        printf("  • Benchmark Iterations     : %d\n", cfg.iterations);
        if (cfg.savePackage) {
            printf("  • MPSGraphPackage Export   : ENABLED -> %s\n", cfg.packageDir.UTF8String);
        }
        printf("========================================================================================================\n");

        NSMutableArray *results = [NSMutableArray array];

        if ([cfg.variant isEqualToString:@"all"] || [cfg.variant isEqualToString:@"fp16"]) {
            BenchResult rFp16 = benchmarkVariant(&cfg, dev, @"conv_fp16", MPSDataTypeFloat16, NO);
            [results addObject:@{ @"name": @"ANE FP16", @"res": [NSValue valueWithBytes:&rFp16 objCType:@encode(BenchResult)] }];
        }
        if ([cfg.variant isEqualToString:@"all"] || [cfg.variant isEqualToString:@"int8"]) {
            BenchResult rInt8 = benchmarkVariant(&cfg, dev, @"conv_int8", MPSDataTypeInt8, NO);
            [results addObject:@{ @"name": @"ANE INT8", @"res": [NSValue valueWithBytes:&rInt8 objCType:@encode(BenchResult)] }];
        }
        if ([cfg.variant isEqualToString:@"all"] || [cfg.variant isEqualToString:@"qdq"]) {
            BenchResult rQdq = benchmarkVariant(&cfg, dev, @"conv_qdq", MPSDataTypeInt8, YES);
            [results addObject:@{ @"name": @"ANE QDQ", @"res": [NSValue valueWithBytes:&rQdq objCType:@encode(BenchResult)] }];
        }

        // Print Final Executive Comparison Matrix
        printf("\n\n========================================================================================================\n");
        printf("📊 FINAL BENCHMARK SUMMARY & PMU TELEMETRY COMPARISON MATRIX\n");
        printf("========================================================================================================\n");
        printf("%-12s | %-12s | %-11s | %-13s | %-14s | %-13s | %-10s\n",
               "Variant", "Latency", "Throughput", "Compute Cycles", "Output Stalls", "Planar Cycles", "DMA I/O");
        printf("-------------+--------------+-------------+---------------+----------------+---------------+-----------\n");

        for (NSDictionary *dict in results) {
            NSString *name = dict[@"name"];
            BenchResult r;
            [dict[@"res"] getValue:&r];
            printf("%-12s | %6.2f ms    | %6.2f TOPS | %'13llu | %'14llu | %'13llu | %6.2f MB\n",
                   name.UTF8String,
                   (r.anePmuTimeMs > 0) ? r.anePmuTimeMs : r.aneMpsTimeMs,
                   (r.anePmuTops > 0) ? r.anePmuTops : r.aneMpsTops,
                   r.computeCycles,
                   r.outputStallCycles,
                   r.l2peCycles,
                   (double)r.dmaRwBytes / (1024.0 * 1024.0));
        }
        printf("========================================================================================================\n\n");
    }
    return 0;
}
