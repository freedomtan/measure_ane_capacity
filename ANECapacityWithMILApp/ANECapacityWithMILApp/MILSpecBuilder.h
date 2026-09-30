//
//  MILSpecBuilder.h
//  ANECapacityWithMILApp
//
//  Pure Objective-C dynamic builder for CoreML MIL (Model Intermediate Language) model specifications.
//  Encodes the protobuf wire format directly without any protobuf library, Python, or coremltools.
//

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

typedef NS_ENUM(NSInteger, MILWeightMode) {
    MILWeightModeDense = 0,
    MILWeightModeRepeat = 1,
};

typedef NS_ENUM(NSInteger, MILPrecision) {
    MILPrecisionFP16 = 0,
    MILPrecisionINT8 = 1,
    MILPrecisionFP8 = 2,
};

typedef struct {
    NSUInteger batch;
    NSUInteger channelsIn;
    NSUInteger channelsOut;
    NSUInteger height;
    NSUInteger width;
    NSUInteger kernel;
    NSUInteger layers;
    MILWeightMode weightMode;
    MILPrecision precision;
} MILConvChainConfig;

FOUNDATION_EXPORT NSString *const MILSpecBuilderErrorDomain;

FOUNDATION_EXPORT NSString *MILConvChainInputName(void);
FOUNDATION_EXPORT NSString *MILConvChainOutputName(MILConvChainConfig config);
FOUNDATION_EXPORT NSString *MILPrecisionName(MILPrecision precision);

/// Emits the raw bytes of a CoreML Model specification (.mlmodel) dynamically.
FOUNDATION_EXPORT NSData *_Nullable MILBuildConvChainSpec(MILConvChainConfig config, NSError **error);

/// Synthesizes a native FP8 model package (.mlmodelc directory) directly on device
/// with MILBlob DataType 16, constexpr_blockwise_shift_scale, and activation QDQ.
FOUNDATION_EXPORT BOOL MILBuildNativeFP8ModelPackage(MILConvChainConfig config, NSURL *outputDirectoryURL, NSError **error);

NS_ASSUME_NONNULL_END
