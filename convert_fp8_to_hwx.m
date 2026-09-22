/**
 * convert_fp8_to_hwx.m -- Convert MPSGraph FP8 QDQ graph to ANE .hwx for target architecture
 *
 * PIPELINE:
 *   1. Create an MPSGraphDeviceDescriptor configured for the specified target architecture (e.g. h19, h18).
 *   2. Construct the FP8 QDQ conv MPSGraph (matching measure_conv_fp8.m).
 *   3. Configure MPSGraphCompilationDescriptor with preferredDevice = ANE and
 *      enableCompileResourcesForPackage = YES.
 *   4. Compile the graph with compileWithDevice: using the target device.
 *   5. Serialize the executable to an .mpsgraphpackage directory.
 *      MPSGraph invokes the ANE compiler for the target architecture and generates binary_0.hwx.
 *   6. Extract the compiled .hwx binary into the output directory and report Mach-O metadata.
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#import <mach-o/loader.h>
#import <sys/stat.h>

// ---------------------------------------------------------------------------
// Private MPSGraph Interfaces
// ---------------------------------------------------------------------------

@interface MPSGraphDeviceDescriptor : NSObject
- (instancetype)initWithSerializedProperties:(NSArray *)props;
- (NSString *)architecture;
@end

@interface MPSGraphDevice (TargetArch)
- (instancetype)initWithDeviceDescriptor:(MPSGraphDeviceDescriptor *)desc;
@end

@interface MPSGraphCompilationDescriptor (Private)
@property (nonatomic, assign) unsigned long long preferredDevice;
- (void)setEnableCompileResourcesForPackage:(BOOL)val;
@end

@interface MPSGraphExecutableSerializationDescriptor (Private)
@property (readwrite, assign) BOOL serializeOriginalModule;
@end

@interface MPSGraphExecutable (Package)
- (void)serializeToMPSGraphPackageAtURL:(NSURL *)url descriptor:(id)desc;
@end

// ---------------------------------------------------------------------------
// Architecture Mapping & Names
// ---------------------------------------------------------------------------

struct ArchDesc {
    const char *name;
    const char *chip;
    uint32_t subtype;
};

static const struct ArchDesc kKnownArchs[] = {
    {"h13",  "A14/M1",        7},
    {"h14",  "A15/M2",       11},
    {"h15",  "A16/M3",        8},
    {"h16",  "A17 Pro/M4",   17},
    {"h16g", "M4",           17},
    {"h16s", "M4 Pro",       17},
    {"h17",  "A18/M5",       19},
    {"h17p", "A18 Pro",      19},
    {"h18",  "A19 / H18",    10},
    {"h19",  "A20 Pro / H19", 11},
    {NULL,   NULL,            0}
};

static const char *getChipForArch(NSString *arch) {
    for (int i = 0; kKnownArchs[i].name != NULL; i++) {
        if ([arch caseInsensitiveCompare:[NSString stringWithUTF8String:kKnownArchs[i].name]] == NSOrderedSame) {
            return kKnownArchs[i].chip;
        }
    }
    return "Custom/Generic";
}

// ---------------------------------------------------------------------------
// FP8 Random Data Fill (from measure_conv_fp8.m)
// ---------------------------------------------------------------------------

static void fillFP8Random(void *buffer, size_t byteCount, uint64_t seed, int shift) {
    if (!buffer || byteCount == 0) return;
    int storedExp = shift + 7;
    if (storedExp < 1) storedExp = 1;
    if (storedExp > 14) storedExp = 14;
    uint8_t posByte = (uint8_t)(storedExp << 3);
    uint8_t negByte = (uint8_t)(0x80 | posByte);
    uint8_t *p = (uint8_t *)buffer;
    uint64_t state = seed;
    for (size_t i = 0; i < byteCount; i++) {
        state ^= state << 13; state ^= state >> 7; state ^= state << 17;
        p[i] = (state & 1) ? negByte : posByte;
    }
}

// ---------------------------------------------------------------------------
// Mach-O HWX Header Inspection
// ---------------------------------------------------------------------------

static void inspectHWX(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return;

    struct mach_header_64 header;
    if (fread(&header, sizeof(header), 1, f) != 1) {
        fclose(f);
        return;
    }

    if (header.magic == MH_MAGIC_64 || header.magic == 0xbeefface) {
        printf("  [HWX Validation]\n");
        printf("    Magic               : 0x%08x\n", header.magic);
        printf("    CPU Type            : 0x%x\n", header.cputype);
        printf("    CPU Subtype         : 0x%x\n", header.cpusubtype);
        printf("    Load Commands       : %u (%u bytes)\n", header.ncmds, header.sizeofcmds);

        // Scan load commands for segments
        for (uint32_t i = 0; i < header.ncmds; i++) {
            struct segment_command_64 seg;
            if (fread(&seg, sizeof(uint32_t) * 2, 1, f) != 1) break;
            fseek(f, -((long)(sizeof(uint32_t) * 2)), SEEK_CUR);

            if (seg.cmd == LC_SEGMENT_64) {
                if (fread(&seg, sizeof(seg), 1, f) != 1) break;
                printf("    Segment %-12s: vm=0x%llx (0x%llx bytes), fileoff=0x%llx (0x%llx bytes)\n",
                       seg.segname, seg.vmaddr, seg.vmsize, seg.fileoff, seg.filesize);
                fseek(f, seg.cmdsize - sizeof(seg), SEEK_CUR);
            } else {
                uint32_t cmd[2];
                if (fread(cmd, sizeof(cmd), 1, f) != 1) break;
                fseek(f, cmd[1] - sizeof(cmd), SEEK_CUR);
            }
        }
    }
    fclose(f);
}

// ---------------------------------------------------------------------------
// Main Conversion Driver
// ---------------------------------------------------------------------------

int main(int argc, char *argv[]) {
    @autoreleasepool {
        NSString *arch = @"h19";
        NSString *outputDir = @"hwx_output";
        NSUInteger layers = 20;
        int logicalShift = -5;
        int physicalShift = -1;
        BOOL weightOnly = NO;
        NSUInteger batchSize = 1;
        NSUInteger spatial = 256;
        NSUInteger inChannels = 128;
        NSUInteger outChannels = 128;
        NSUInteger kernelSize = 3;

        for (int i = 1; i < argc; i++) {
            NSString *arg = [NSString stringWithUTF8String:argv[i]];
            if ([arg isEqualToString:@"--arch"] && i + 1 < argc) {
                arch = [NSString stringWithUTF8String:argv[++i]];
            } else if ([arg isEqualToString:@"--output"] && i + 1 < argc) {
                outputDir = [NSString stringWithUTF8String:argv[++i]];
            } else if ([arg isEqualToString:@"--layers"] && i + 1 < argc) {
                layers = (NSUInteger)atoi(argv[++i]);
            } else if ([arg isEqualToString:@"--logical-shift"] && i + 1 < argc) {
                logicalShift = atoi(argv[++i]);
            } else if ([arg isEqualToString:@"--physical-shift"] && i + 1 < argc) {
                physicalShift = atoi(argv[++i]);
            } else if ([arg isEqualToString:@"--mode"] && i + 1 < argc) {
                NSString *m = [NSString stringWithUTF8String:argv[++i]];
                if ([m isEqualToString:@"weight-only"]) {
                    weightOnly = YES;
                } else if ([m isEqualToString:@"full"]) {
                    weightOnly = NO;
                }
            } else if ([arg isEqualToString:@"--spatial"] && i + 1 < argc) {
                spatial = (NSUInteger)atoi(argv[++i]);
            } else if ([arg isEqualToString:@"--channels"] && i + 1 < argc) {
                inChannels = outChannels = (NSUInteger)atoi(argv[++i]);
            } else if ([arg isEqualToString:@"--help"] || [arg isEqualToString:@"-h"]) {
                printf(
                    "Usage: %s [OPTIONS]\n\n"
                    "Convert MPSGraph FP8 QDQ Conv to Apple Neural Engine .hwx offline.\n\n"
                    "Options:\n"
                    "  --arch <arch>         Target ANE architecture (default: h19)\n"
                    "                        Supported: h19 (A20 Pro), h18 (A19), h17p, h16s, etc.\n"
                    "  --output <dir>        Output directory (default: hwx_output)\n"
                    "  --layers N            Chained conv layers (default: 20)\n"
                    "  --logical-shift E     FP16 math magnitude = 2^E (default: -5)\n"
                    "  --physical-shift E    FP8 byte magnitude  = 2^E (default: -1)\n"
                    "  --mode <mode>         'full' (W8A8 QDQ, default) or 'weight-only'\n"
                    "  --spatial <size>      H/W spatial dimensions (default: 256)\n"
                    "  --channels <C>        Input and output channels (default: 128)\n"
                    "  --help                Show this help message\n",
                    argv[0]);
                return 0;
            }
        }

        printf("================================================================\n");
        printf("       MPSGraph FP8 Conv to ANE HWX Offline Converter           \n");
        printf("================================================================\n");
        printf("Target Architecture : %s (%s)\n", arch.UTF8String, getChipForArch(arch));
        printf("Configuration       : Layers=%lu, Channels=%lu, Spatial=%lux%lu, K=%lux%lu\n",
               (unsigned long)layers, (unsigned long)inChannels, (unsigned long)spatial,
               (unsigned long)spatial, (unsigned long)kernelSize, (unsigned long)kernelSize);
        printf("Quantization        : Mode=%s, Logical Shift=2^%d, Physical Shift=2^%d\n",
               weightOnly ? "Weight-Only QDQ" : "W8A8 Full QDQ", logicalShift, physicalShift);
        printf("Output Directory    : %s\n\n", outputDir.UTF8String);

        // 1. Create target MPSGraphDevice for specified architecture
        printf("🔨 [Step 1/4] Configuring MPSGraph target device for %s...\n", arch.UTF8String);
        MPSGraphDeviceDescriptor *devDesc = [[MPSGraphDeviceDescriptor alloc]
            initWithSerializedProperties:@[@(0), @(6), arch]];
        MPSGraphDevice *targetDev = [[MPSGraphDevice alloc] initWithDeviceDescriptor:devDesc];
        if (!targetDev) {
            fprintf(stderr, "Error: Failed to create MPSGraphDevice for architecture %s\n", arch.UTF8String);
            return 1;
        }

        // 2. Build FP8 Conv Graph
        printf("📐 [Step 2/4] Constructing FP8 QDQ MPSGraph...\n");
        MPSDataType fp8Type = MPSDataTypeFloat8e4m3;
        MPSDataType actType = weightOnly ? MPSDataTypeFloat16 : fp8Type;

        NSArray *inShape = @[@(batchSize), @(inChannels), @(spatial), @(spatial)];
        NSArray *wShape  = @[@(outChannels), @(inChannels), @(kernelSize), @(kernelSize)];
        double scale = exp2((double)(logicalShift - physicalShift));

        MPSGraph *graph = [MPSGraph new];
        MPSGraphTensor *input = [graph placeholderWithShape:inShape dataType:actType name:@"input"];
        MPSGraphTensor *cur = input;

        NSMutableData *wData = [NSMutableData dataWithLength:outChannels * inChannels * kernelSize * kernelSize];
        fillFP8Random(wData.mutableBytes, wData.length, 0x5EED5EED5EED5EEDULL, physicalShift);
        MPSGraphTensor *wFP8 = [graph constantWithData:wData shape:wShape dataType:fp8Type];
        MPSGraphTensor *w = [graph dequantizeTensor:wFP8 scale:scale zeroPoint:0.0
                                           dataType:MPSDataTypeFloat16 name:@"w_dequant"];

        MPSGraphConvolution2DOpDescriptor *convDesc = [MPSGraphConvolution2DOpDescriptor
            descriptorWithStrideInX:1 strideInY:1 dilationRateInX:1 dilationRateInY:1
                             groups:1 paddingStyle:MPSGraphPaddingStyleTF_SAME
                         dataLayout:MPSGraphTensorNamedDataLayoutNCHW
                      weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];

        for (NSUInteger i = 0; i < layers; i++) {
            if (weightOnly) {
                cur = [graph convolution2DWithSourceTensor:cur weightsTensor:w descriptor:convDesc name:nil];
            } else {
                MPSGraphTensor *inFP16 = [graph dequantizeTensor:cur scale:scale zeroPoint:0.0
                                                        dataType:MPSDataTypeFloat16 name:nil];
                MPSGraphTensor *outFP16 = [graph convolution2DWithSourceTensor:inFP16
                                                                weightsTensor:w descriptor:convDesc name:nil];
                cur = [graph quantizeTensor:outFP16 scale:scale zeroPoint:0.0
                                   dataType:fp8Type name:nil];
            }
        }

        // 3. Compile Graph with ANE resources enabled
        printf("⚙️  [Step 3/4] Compiling graph for ANE via MPSGraph compiler...\n");
        MPSGraphCompilationDescriptor *cd = [MPSGraphCompilationDescriptor new];
        cd.preferredDevice = 2; // MPSGraphDeviceTypeANE
        cd.optimizationLevel = MPSGraphOptimizationLevel1;
        if ([cd respondsToSelector:@selector(setEnableCompileResourcesForPackage:)]) {
            [cd setEnableCompileResourcesForPackage:YES];
        }

        NSDictionary *feeds = @{input: [[MPSGraphShapedType alloc] initWithShape:inShape dataType:actType]};
        MPSGraphExecutable *exe = [graph compileWithDevice:targetDev
                                                     feeds:feeds
                                             targetTensors:@[cur]
                                          targetOperations:nil
                                     compilationDescriptor:cd];
        if (!exe) {
            fprintf(stderr, "Error: MPSGraph compilation failed for target %s\n", arch.UTF8String);
            return 1;
        }

        // 4. Serialize to .mpsgraphpackage directory
        printf("📦 [Step 4/4] Serializing to directory and extracting .hwx...\n");
        NSFileManager *fm = [NSFileManager defaultManager];
        [fm createDirectoryAtPath:outputDir withIntermediateDirectories:YES attributes:nil error:nil];

        NSString *archDir = [outputDir stringByAppendingPathComponent:arch];
        [fm createDirectoryAtPath:archDir withIntermediateDirectories:YES attributes:nil error:nil];

        NSString *pkgPath = [archDir stringByAppendingPathComponent:@"model.mpsgraphpackage"];
        [fm removeItemAtPath:pkgPath error:nil];

        MPSGraphExecutableSerializationDescriptor *sDesc = [MPSGraphExecutableSerializationDescriptor new];
        if ([sDesc respondsToSelector:@selector(setSerializeOriginalModule:)]) {
            sDesc.serializeOriginalModule = YES;
        }

        [exe serializeToMPSGraphPackageAtURL:[NSURL fileURLWithPath:pkgPath] descriptor:sDesc];

        // Find binary_0.hwx in the package
        NSString *bundledHwx = [pkgPath stringByAppendingPathComponent:@"binary_0.hwx"];
        NSString *finalHwx = [archDir stringByAppendingPathComponent:@"model.hwx"];

        if ([fm fileExistsAtPath:bundledHwx]) {
            [fm removeItemAtPath:finalHwx error:nil];
            [fm copyItemAtPath:bundledHwx toPath:finalHwx error:nil];

            unsigned long long hwxSize = [[fm attributesOfItemAtPath:finalHwx error:nil] fileSize];
            printf("\n✨ SUCCESS!\n");
            printf("  • Compiled ANE Binary : %s (%llu bytes)\n", finalHwx.UTF8String, hwxSize);
            printf("  • Full Package        : %s\n\n", pkgPath.UTF8String);

            inspectHWX(finalHwx.UTF8String);
            return 0;
        } else {
            fprintf(stderr, "Error: binary_0.hwx not found in package at %s\n", pkgPath.UTF8String);
            NSArray *files = [fm contentsOfDirectoryAtPath:pkgPath error:nil];
            fprintf(stderr, "Files in package: %s\n", files.description.UTF8String);
            return 1;
        }
    }
}
