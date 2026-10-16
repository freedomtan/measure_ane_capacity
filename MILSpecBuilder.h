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

/// Precision mode for the benchmark.
typedef NS_ENUM(NSInteger, MILPrecision) {
  MILPrecisionFP16 = 0,
  /// Quantized INT8 / W8A8 QDQ: activation quantize/dequantize, INT8 weights
  /// dequantized to FP16, and FP16 conv executed on the ANE matrix engine.
  MILPrecisionINT8 = 1,
  /// Quantized FP8 / W8A8 QDQ: activation quantize/dequantize to FP8 E4M3,
  /// FP8 weights dequantized to FP16, targeting native FP8 ANE engines on H18/H19.
  MILPrecisionFP8 = 2,
  /// Matmul only: weight-only INT8 (W8A16, no activation QDQ at all).
  /// Activations stay FP16 straight through; only the weight constant is
  /// INT8 (constexpr_blockwise_shift_scale), same as MILPrecisionINT8's
  /// weight representation. Tests whether BNNS's CPU GEMM dispatcher picks
  /// its KERNEL_MATMUL_WITH_AFFINE_QUANTIZE_WEIGHT kernel (see
  /// docs/CPU_Matrix_Convolution_Capability.md Finding 6) once the
  /// surrounding activation quantize/dequantize ops are no longer in the way.
  MILPrecisionINT8WeightOnly = 3,
};

/// The workload: `layers` chained KxK convolutions over [batch, channelsIn,
/// height, width], NCHW layout, SAME padding.
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

/// Name of the single model input ("x").
FOUNDATION_EXPORT NSString *MILConvChainInputName(void);

/// Name of the single model output (the last conv's result).
FOUNDATION_EXPORT NSString *MILConvChainOutputName(MILConvChainConfig config);

FOUNDATION_EXPORT NSString *MILWeightModeName(MILWeightMode mode);
FOUNDATION_EXPORT NSString *MILPrecisionName(MILPrecision precision);

/// Serialize a complete CoreML `Model` protobuf for `config`.
/// Returns nil and fills `error` if any dimension is zero.
FOUNDATION_EXPORT NSData *MILBuildConvChainSpec(MILConvChainConfig config,
                                                NSError **error);

/// MIL-text-style dump of the program that MILBuildConvChainSpec emits, for
/// eyeballing the graph without a protobuf decoder.
FOUNDATION_EXPORT NSString *MILConvChainText(MILConvChainConfig config);

/// The workload: `layers` chained GEMMs, each `[batch, 1, M, K] x [batch, 1,
/// K, N] -> [batch, 1, M, N]`, matching measure_matmul_universal.m's tensor
/// shapes exactly. Chaining requires K == N, same as the conv chain requires
/// channelsIn == channelsOut: each layer's output becomes the next layer's
/// input, so the shape must be self-consistent across layers.
///
/// `precision` mirrors the conv chain's QDQ pattern: INT8 quantizes/
/// dequantizes activations around an FP16 matmul (weights stored as
/// constexpr_blockwise_shift_scale INT8); FP8 dequantizes FP8-E4M3 weights
/// (constexpr_lut_to_dense) but, like the conv chain, does not quantize
/// activations.
typedef struct {
  NSUInteger batch;
  NSUInteger M;
  NSUInteger K;
  NSUInteger N;
  NSUInteger layers;
  MILWeightMode weightMode;
  MILPrecision precision;
} MILMatmulChainConfig;

/// Name of the single model input ("x").
FOUNDATION_EXPORT NSString *MILMatmulChainInputName(void);

/// Name of the single model output (the last matmul's result).
FOUNDATION_EXPORT NSString *MILMatmulChainOutputName(MILMatmulChainConfig config);

/// Serialize a complete CoreML `Model` protobuf for `config`.
/// Returns nil and fills `error` if any dimension is zero, or if K != N.
FOUNDATION_EXPORT NSData *MILBuildMatmulChainSpec(MILMatmulChainConfig config,
                                                  NSError **error);

/// MIL-text-style dump of the program that MILBuildMatmulChainSpec emits, for
/// eyeballing the graph without a protobuf decoder.
FOUNDATION_EXPORT NSString *MILMatmulChainText(MILMatmulChainConfig config);

/// A single `[M, K] x [K, N] -> [M, N]` matmul whose *weight* is a genuine
/// function **input**, not a `const`/`constexpr_*` -- unlike
/// MILMatmulChainConfig's INT8/FP8 modes, there is nothing here for the MIL
/// compiler to constant-fold (see docs/CPU_Matrix_Convolution_Capability.md
/// Finding 10). If `weightIsInt8` is true, the weight input is INT8 and is
/// `dequantize`d (scale 1/32) before the matmul; otherwise the weight input
/// is FP16 and feeds the matmul directly. Built for use with
/// `BNNSGraphCompileFromFile` (Finding 11's natural follow-up): compile the
/// resulting spec's `.mlmodelc` and execute it directly through BNNSGraph,
/// supplying the weight tensor at call time so no compiler or loader ever
/// sees it as fixed data.
typedef struct {
  NSUInteger M;
  NSUInteger K;
  NSUInteger N;
  BOOL weightIsInt8;
  /// If true, appends `relu` immediately after `matmul` as the function's
  /// real output, instead of returning the matmul's result directly.
  /// Probes whether BNNS's compiler fuses the pair into a single
  /// "elementwise_activation_matmul" op -- a different "kernel_routine
  /// type" than bare matmul (see
  /// docs/AMX_GEMM_Microkernel_Disassembly.md's "Clues toward actually
  /// reaching the SME2 path" section) that may dispatch to the real SME2
  /// kernel instead of the AMX one every bare matmul in this repo reaches.
  BOOL fuseRelu;
} MILMatmulWeightInputConfig;

FOUNDATION_EXPORT NSString *MILMatmulWeightInputXName(void);
FOUNDATION_EXPORT NSString *MILMatmulWeightInputWeightName(void);
FOUNDATION_EXPORT NSString *MILMatmulWeightInputOutputName(void);

FOUNDATION_EXPORT NSData *_Nullable MILBuildMatmulWeightInputSpec(
    MILMatmulWeightInputConfig config, NSError **error);

/// `scaled_dot_product_attention(query, key, value)` -- coremltools 9.0's
/// single native MIL op (iOS18 opset), computing
/// `softmax(Q @ K^T / sqrt(E)) @ V` as one op, not three chained ones.
/// Candidate for reaching BNNS's `KERNEL_MATMUL_SOFTMAX_MATMUL_SME` tier
/// (see docs/AMX_GEMM_Microkernel_Disassembly.md's "Clues toward actually
/// reaching the SME2 path" -- the fused `matmul`+`relu` experiment there
/// didn't trigger it because CoreML never actually fused those two ops;
/// this op is a genuine single fused primitive from the start).
/// query: [B, L, E], key/value: [B, S, E] (value's last dim EV == E here,
/// kept square for simplicity). All three are genuine function inputs, not
/// consts, same non-foldable-weight design as MILMatmulWeightInputConfig.
typedef struct {
  NSUInteger B;
  NSUInteger L;
  NSUInteger S;
  NSUInteger E;
} MILAttentionConfig;

FOUNDATION_EXPORT NSString *MILAttentionQueryName(void);
FOUNDATION_EXPORT NSString *MILAttentionKeyName(void);
FOUNDATION_EXPORT NSString *MILAttentionValueName(void);
FOUNDATION_EXPORT NSString *MILAttentionOutputName(void);

FOUNDATION_EXPORT NSData *_Nullable MILBuildAttentionSpec(MILAttentionConfig config, NSError **error);
