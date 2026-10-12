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

/// Synthesizes a native model package (.mlmodelc directory) directly on device
/// with MILBlob weight storage, optional activation QDQ, and zero-overhead fused hardware bias.
FOUNDATION_EXPORT BOOL MILBuildNativeModelPackage(MILConvChainConfig config, NSURL *outputDirectoryURL, NSError **error);

/// Synthesizes a native FP8 model package (.mlmodelc directory) directly on device
/// with MILBlob DataType 16, constexpr_blockwise_shift_scale, and activation QDQ.
FOUNDATION_EXPORT BOOL MILBuildNativeFP8ModelPackage(MILConvChainConfig config, NSURL *outputDirectoryURL, NSError **error);

/// A chain of `layers` [batch,1,M,K] x [batch,1,K,N] -> [batch,1,M,N] matmuls
/// sharing one dense-mode weight constant, mirroring the C++
/// MILSpecBuilder.mm's MILMatmulChainConfig (see the main repo's
/// docs/CPU_Matrix_Convolution_Capability.md). Chaining requires K == N,
/// same reasoning as the conv chain's channelsIn == channelsOut.
/// `precision` mirrors the conv chain's QDQ pattern (see MILConvChainConfig
/// above): INT8 quantizes/dequantizes activations around an FP16 matmul;
/// FP8 dequantizes FP8-E4M3 weights but does not quantize activations.
typedef struct {
    NSUInteger batch;
    NSUInteger M;
    NSUInteger K;
    NSUInteger N;
    NSUInteger layers;
    MILWeightMode weightMode;
    MILPrecision precision;
} MILMatmulChainConfig;

FOUNDATION_EXPORT NSString *MILMatmulChainInputName(void);
FOUNDATION_EXPORT NSString *MILMatmulChainOutputName(MILMatmulChainConfig config);

/// Emits the raw bytes of a CoreML Model specification (.mlmodel) dynamically.
/// Returns nil and fills `error` if any dimension is zero, or if K != N.
FOUNDATION_EXPORT NSData *_Nullable MILBuildMatmulChainSpec(MILMatmulChainConfig config, NSError **error);

/// A single `[M, K] x [K, N] -> [M, N]` matmul whose weight is a genuine
/// function **input**, not a `const`/`constexpr_*` -- mirrors the root
/// repo's MILMatmulWeightInputConfig (see
/// docs/CPU_Matrix_Convolution_Capability.md Finding 12). If `weightIsInt8`
/// is true, the weight is declared as an INT32 model input (CoreML's
/// top-level ArrayFeatureType has no INT8 option) and `cast` to INT8 as the
/// function's first op, then `dequantize`d (scale 1/32) before the matmul;
/// otherwise the weight input is FP16 and feeds the matmul directly.
typedef struct {
    NSUInteger M;
    NSUInteger K;
    NSUInteger N;
    BOOL weightIsInt8;
} MILMatmulWeightInputConfig;

FOUNDATION_EXPORT NSString *MILMatmulWeightInputXName(void);
FOUNDATION_EXPORT NSString *MILMatmulWeightInputWeightName(void);
FOUNDATION_EXPORT NSString *MILMatmulWeightInputOutputName(void);

FOUNDATION_EXPORT NSData *_Nullable MILBuildMatmulWeightInputSpec(MILMatmulWeightInputConfig config, NSError **error);

NS_ASSUME_NONNULL_END
