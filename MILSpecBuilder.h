// Native construction of CoreML MIL (Model Intermediate Language) model specs.
//
// Emits the bytes of a CoreML `Model` protobuf -- an MLProgram holding a chain
// of convolutions -- with no coremltools, no Python, and no protobuf library.
// The result is fed straight to +[MLModelAsset modelAssetWithSpecificationData:]
// (macOS 13 / iOS 16), so nothing ever touches the filesystem and there is no
// coremlcompiler step. That is what makes this path portable to iOS, where
// coremltools cannot run at all.
//
// Weight constants are stored inline in the spec as MIL TensorValue.RepeatedBytes.
// The alternative -- coremltools' "blob v2" weights/weight.bin referenced as
// "@model_path/weights/weight.bin" -- cannot work for an in-memory asset,
// because there is no model path to resolve it against.
//
// Field numbers below follow coremltools' mlmodel/format/{Model,MIL,
// FeatureTypes}.proto.

#import <Foundation/Foundation.h>

/// How weight constants are filled.
typedef NS_ENUM(NSInteger, MILWeightMode) {
  /// Random-sign, constant-magnitude 1/32. Non-zero and, crucially,
  /// non-cancelling under the convolution reduction.
  MILWeightModeDense = 0,
  /// The tiled +/-0.0625, +/-0.03125 pattern used by fillNonZeroData() in the
  /// MPSGraph benchmarks. Bit-comparable with those binaries, but degenerate:
  /// the pattern cancels exactly under the Ci*K*K reduction, so layer 1 emits
  /// all zeros and every later layer consumes a zero tensor -- exactly the
  /// condition H17+ hardware zero-skipping detects.
  MILWeightModeRepeat = 1,
};

/// The workload: `layers` chained KxK convolutions over [batch, channelsIn,
/// height, width], NCHW layout, SAME padding, fp16 throughout. Mirrors the
/// MPSGraph graph in measure_conv_universal.m.
typedef struct {
  NSUInteger batch;
  NSUInteger channelsIn;
  NSUInteger channelsOut;
  NSUInteger height;
  NSUInteger width;
  NSUInteger kernel;
  NSUInteger layers;
  MILWeightMode weightMode;
} MILConvChainConfig;

FOUNDATION_EXPORT NSString *const MILSpecBuilderErrorDomain;

/// Name of the single model input ("x").
FOUNDATION_EXPORT NSString *MILConvChainInputName(void);

/// Name of the single model output (the last conv's result).
FOUNDATION_EXPORT NSString *MILConvChainOutputName(MILConvChainConfig config);

FOUNDATION_EXPORT NSString *MILWeightModeName(MILWeightMode mode);

/// Serialize a complete CoreML `Model` protobuf for `config`.
/// Returns nil and fills `error` if any dimension is zero.
FOUNDATION_EXPORT NSData *MILBuildConvChainSpec(MILConvChainConfig config,
                                                NSError **error);

/// MIL-text-style dump of the program that MILBuildConvChainSpec emits, for
/// eyeballing the graph without a protobuf decoder.
FOUNDATION_EXPORT NSString *MILConvChainText(MILConvChainConfig config);
