// Native MIL program construction using coremltools' own schema.
//
// coremltools has no C++ builder -- mb.program/mb.conv live only in
// coremltools/converters/mil/mil/builder.py. What it does ship natively is the
// protobuf schema (mlmodel/format/*.proto, vendored under third_party/) plus the
// generated C++ classes it checks in at mlmodel/build/format/*.pb.{h,cc}. Those
// generated classes ARE the native builder API: the same CoreML::Specification
// types coremltools' own C++ validators manipulate. We regenerate them with the
// local protoc rather than using the checked-in copies, which are pinned to
// protobuf 3.x.
//
// So the Python pipeline
//     mb.program -> ct.convert -> mlpackage -> coremlcompiler
// becomes
//     MILSpec::Program -> SerializeToString -> MLModelAsset
// with no file system involved at any point, which is what makes this portable
// to iOS where coremltools cannot run.

#import "MILSpecBuilder.h"

#import <string>
#import <vector>

#import "Model.pb.h"

namespace ms = CoreML::Specification;
namespace mil = CoreML::Specification::MILSpec;

NSString *const MILSpecBuilderErrorDomain = @"MILSpecBuilder";

// CoreML specification version. 9 == iOS 18 / macOS 15, whose MIL opset is named
// "CoreML8". The two must agree: the opset string keys both Function.opset and
// Function.block_specializations.
static const int32_t kSpecificationVersion = 9;
static const char *const kOpset = "CoreML8";

// fp16 bit patterns; 1/32 and 1/16 are both exact in fp16.
enum : uint16_t {
  kFP16PlusOneThirtySecond = 0x2800,
  kFP16MinusOneThirtySecond = 0xA800,
  kFP16PlusOneSixteenth = 0x2C00,
  kFP16MinusOneSixteenth = 0xAC00,
};

// Fixed seed so a given configuration always yields identical weights, and
// distinct from the input filler's seed so weights and activations do not
// correlate.
static const uint64_t kWeightSeed = 0x5EED5EED5EED5EEDULL;

#pragma mark - Type and value helpers

static void SetTensorType(mil::ValueType *type, mil::DataType dataType,
                          const std::vector<uint64_t> &shape) {
  mil::TensorType *tensor = type->mutable_tensortype();
  tensor->set_datatype(dataType);
  tensor->set_rank((int64_t)shape.size());
  for (uint64_t dim : shape) {
    tensor->add_dimensions()->mutable_constant()->set_size(dim);
  }
}

static mil::ValueType TensorValueType(mil::DataType dataType,
                                      const std::vector<uint64_t> &shape) {
  mil::ValueType type;
  SetTensorType(&type, dataType, shape);
  return type;
}

// A rank-0 string, the type of every op's "name" attribute and of pad_type.
static mil::ValueType StringScalarType() {
  mil::ValueType type;
  type.mutable_tensortype()->set_datatype(mil::STRING);
  return type;
}

static mil::Value StringValue(const std::string &text) {
  mil::Value value;
  *value.mutable_type() = StringScalarType();
  value.mutable_immediatevalue()->mutable_tensor()->mutable_strings()->add_values(
      text);
  return value;
}

static mil::Value Int32Value(const std::vector<int32_t> &values, bool scalar) {
  mil::Value value;
  *value.mutable_type() = TensorValueType(
      mil::INT32, scalar ? std::vector<uint64_t>{}
                         : std::vector<uint64_t>{(uint64_t)values.size()});
  mil::TensorValue_RepeatedInts *ints =
      value.mutable_immediatevalue()->mutable_tensor()->mutable_ints();
  for (int32_t v : values) ints->add_values(v);
  return value;
}

#pragma mark - Operations

// A const op carries its payload in attributes["val"], not in inputs. That is
// how coremltools serializes it (confirmed against a converted model) and what
// the CoreML compiler expects.
static void AddConstOp(mil::Block *block, const std::string &name,
                       const mil::ValueType &type, mil::Value value) {
  mil::Operation *op = block->add_operations();
  op->set_type("const");
  mil::NamedValueType *out = op->add_outputs();
  out->set_name(name);
  *out->mutable_type() = type;
  *value.mutable_type() = type;
  (*op->mutable_attributes())["val"] = std::move(value);
  (*op->mutable_attributes())["name"] = StringValue(name);
}

// Every input of a non-const op binds to a previously defined value by name.
static void AddOp(mil::Block *block, const std::string &type,
                  const std::vector<std::pair<std::string, std::string>> &inputs,
                  const std::string &outputName,
                  const mil::ValueType &outputType) {
  mil::Operation *op = block->add_operations();
  op->set_type(type);
  for (const auto &input : inputs) {
    (*op->mutable_inputs())[input.first].add_arguments()->set_name(input.second);
  }
  mil::NamedValueType *out = op->add_outputs();
  out->set_name(outputName);
  *out->mutable_type() = outputType;
  (*op->mutable_attributes())["name"] = StringValue(outputName);
}

#pragma mark - Weights

// Weight constants are stored inline in the spec as MIL
// TensorValue.RepeatedBytes -- the only fp16-capable immediate, since the proto
// has no repeated-half field. The alternative coremltools uses, a "blob v2"
// weights/weight.bin referenced as "@model_path/weights/weight.bin" (see
// MILBlob/Blob/StorageWriter), cannot work for an in-memory asset: there is no
// model path to resolve it against.
static std::string WeightBytes(MILConvChainConfig config) {
  size_t count = (size_t)config.channelsOut * config.channelsIn *
                 config.kernel * config.kernel;
  std::string bytes(count * sizeof(uint16_t), '\0');
  uint16_t *p = (uint16_t *)bytes.data();

  if (config.weightMode == MILWeightModeRepeat) {
    static const uint16_t pattern[4] = {
        kFP16PlusOneSixteenth, kFP16MinusOneSixteenth,
        kFP16PlusOneThirtySecond, kFP16MinusOneThirtySecond};
    for (size_t i = 0; i < count; i++) p[i] = pattern[i % 4];
    return bytes;
  }

  // Random signs over a reduction of Ci*K*K terms give a per-layer gain of
  // sqrt(Ci*K*K)/32, which for Ci=128, K=3 is ~1.06: activations stay inside
  // fp16 range across 20+ layers instead of overflowing or decaying to zero.
  uint64_t state = kWeightSeed;
  for (size_t i = 0; i < count; i++) {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    p[i] = (state & 1) ? kFP16MinusOneThirtySecond : kFP16PlusOneThirtySecond;
  }
  return bytes;
}

#pragma mark - Model description

static void SetArrayFeature(ms::FeatureDescription *desc,
                            const std::string &name,
                            const std::vector<uint64_t> &shape) {
  desc->set_name(name);
  ms::ArrayFeatureType *array =
      desc->mutable_type()->mutable_multiarraytype();
  array->set_datatype(ms::ArrayFeatureType::FLOAT16);
  for (uint64_t dim : shape) array->add_shape((int64_t)dim);
}

// Stamp the workload into the spec so a saved .mlmodel stays self-describing and
// measure_conv_coreml can recover K and L, which are not deducible from the
// input shape. Without them a stale --kernel/--layers silently scales TOPS.
static void SetMetadata(ms::Metadata *metadata, MILConvChainConfig config) {
  NSString *summary = [NSString
      stringWithFormat:@"%lux conv%lux%lu on [%lu, %lu, %lu, %lu], fp16, "
                       @"weights=%@",
                       (unsigned long)config.layers,
                       (unsigned long)config.kernel,
                       (unsigned long)config.kernel, (unsigned long)config.batch,
                       (unsigned long)config.channelsIn,
                       (unsigned long)config.height, (unsigned long)config.width,
                       MILWeightModeName(config.weightMode)];
  metadata->set_shortdescription(summary.UTF8String);

  auto &userDefined = *metadata->mutable_userdefined();
  userDefined["workload.batch"] = std::to_string(config.batch);
  userDefined["workload.channels_in"] = std::to_string(config.channelsIn);
  userDefined["workload.channels_out"] = std::to_string(config.channelsOut);
  userDefined["workload.height"] = std::to_string(config.height);
  userDefined["workload.width"] = std::to_string(config.width);
  userDefined["workload.kernel"] = std::to_string(config.kernel);
  userDefined["workload.layers"] = std::to_string(config.layers);
  userDefined["workload.weights"] =
      MILWeightModeName(config.weightMode).UTF8String;
  userDefined["builder"] = "MILSpecBuilder (native, no coremltools)";
}

#pragma mark - Public

NSString *MILConvChainInputName(void) { return @"x"; }

NSString *MILConvChainOutputName(MILConvChainConfig config) {
  return [NSString stringWithFormat:@"conv_%lu",
                                    (unsigned long)(config.layers - 1)];
}

NSString *MILWeightModeName(MILWeightMode mode) {
  return mode == MILWeightModeRepeat ? @"repeat" : @"dense";
}

static NSError *MILError(NSInteger code, NSString *message) {
  return [NSError errorWithDomain:MILSpecBuilderErrorDomain
                             code:code
                         userInfo:@{NSLocalizedDescriptionKey : message}];
}

static BOOL MILValidate(MILConvChainConfig config, NSError **error) {
  const struct {
    const char *name;
    NSUInteger value;
  } dims[] = {
      {"batch", config.batch},
      {"channels_in", config.channelsIn},
      {"channels_out", config.channelsOut},
      {"height", config.height},
      {"width", config.width},
      {"kernel", config.kernel},
      {"layers", config.layers},
  };
  for (const auto &dim : dims) {
    if (dim.value == 0) {
      if (error) {
        *error = MILError(1, [NSString stringWithFormat:
                                           @"%s must be greater than zero",
                                           dim.name]);
      }
      return NO;
    }
  }
  // A chain of same-padded convs preserves its shape, so layers after the first
  // can only consume the previous output if the channel counts agree.
  if (config.layers > 1 && config.channelsIn != config.channelsOut) {
    if (error) {
      *error = MILError(
          2, @"chained layers require channels_in == channels_out");
    }
    return NO;
  }
  return YES;
}

NSData *MILBuildConvChainSpec(MILConvChainConfig config, NSError **error) {
  if (!MILValidate(config, error)) return nil;

  const std::vector<uint64_t> inputShape = {config.batch, config.channelsIn,
                                            config.height, config.width};
  const std::vector<uint64_t> outputShape = {config.batch, config.channelsOut,
                                             config.height, config.width};
  const std::string outputName = MILConvChainOutputName(config).UTF8String;

  ms::Model model;
  model.set_specificationversion(kSpecificationVersion);

  ms::ModelDescription *description = model.mutable_description();
  SetArrayFeature(description->add_input(),
                  MILConvChainInputName().UTF8String, inputShape);
  SetArrayFeature(description->add_output(), outputName, outputShape);
  SetMetadata(description->mutable_metadata(), config);

  mil::Program *program = model.mutable_mlprogram();
  program->set_version(1);

  mil::Function &function = (*program->mutable_functions())["main"];
  function.set_opset(kOpset);
  mil::NamedValueType *functionInput = function.add_inputs();
  functionInput->set_name(MILConvChainInputName().UTF8String);
  SetTensorType(functionInput->mutable_type(), mil::FLOAT16, inputShape);

  mil::Block &block = (*function.mutable_block_specializations())[kOpset];
  block.add_outputs(outputName);

  // One weight constant shared by every layer, as in the MPSGraph version.
  mil::Value weights;
  weights.mutable_immediatevalue()->mutable_tensor()->mutable_bytes()->set_values(
      WeightBytes(config));
  AddConstOp(&block, "weights",
             TensorValueType(mil::FLOAT16,
                             {config.channelsOut, config.channelsIn,
                              config.kernel, config.kernel}),
             std::move(weights));

  // Constants the conv ops bind to. NCHW plus "same" padding matches the
  // MPSGraph descriptor (NCHW / OIHW / TF_SAME) in measure_conv_universal.m, so
  // both paths compute identical arithmetic.
  AddConstOp(&block, "strides", TensorValueType(mil::INT32, {2}),
             Int32Value({1, 1}, /*scalar=*/false));
  AddConstOp(&block, "dilations", TensorValueType(mil::INT32, {2}),
             Int32Value({1, 1}, /*scalar=*/false));
  AddConstOp(&block, "pad", TensorValueType(mil::INT32, {4}),
             Int32Value({0, 0, 0, 0}, /*scalar=*/false));
  AddConstOp(&block, "groups", TensorValueType(mil::INT32, {}),
             Int32Value({1}, /*scalar=*/true));
  AddConstOp(&block, "pad_type", StringScalarType(), StringValue("same"));

  const mil::ValueType activationType =
      TensorValueType(mil::FLOAT16, outputShape);
  std::string current = MILConvChainInputName().UTF8String;
  for (NSUInteger i = 0; i < config.layers; i++) {
    std::string name = "conv_" + std::to_string(i);
    AddOp(&block, "conv",
          {{"x", current},
           {"weight", "weights"},
           {"strides", "strides"},
           {"pad_type", "pad_type"},
           {"pad", "pad"},
           {"dilations", "dilations"},
           {"groups", "groups"}},
          name, activationType);
    current = std::move(name);
  }

  std::string serialized;
  if (!model.SerializeToString(&serialized)) {
    if (error) *error = MILError(3, @"failed to serialize the model spec");
    return nil;
  }
  return [NSData dataWithBytes:serialized.data() length:serialized.size()];
}

NSString *MILConvChainText(MILConvChainConfig config) {
  NSMutableString *text = [NSMutableString string];
  [text appendFormat:@"function main (opset %s):\n", kOpset];
  [text appendFormat:@"  x: fp16[%lu, %lu, %lu, %lu]\n",
                     (unsigned long)config.batch,
                     (unsigned long)config.channelsIn,
                     (unsigned long)config.height, (unsigned long)config.width];
  [text appendFormat:@"  weights = const(fp16[%lu, %lu, %lu, %lu])  // %@, "
                     @"%lu bytes inline\n",
                     (unsigned long)config.channelsOut,
                     (unsigned long)config.channelsIn,
                     (unsigned long)config.kernel, (unsigned long)config.kernel,
                     MILWeightModeName(config.weightMode),
                     (unsigned long)(config.channelsOut * config.channelsIn *
                                     config.kernel * config.kernel * 2)];
  NSString *current = MILConvChainInputName();
  for (NSUInteger i = 0; i < config.layers; i++) {
    NSString *name = [NSString stringWithFormat:@"conv_%lu", (unsigned long)i];
    [text appendFormat:@"  %@ = conv(x=%@, weight=weights, strides=[1, 1], "
                       @"pad_type=same, pad=[0, 0, 0, 0], dilations=[1, 1], "
                       @"groups=1)\n",
                       name, current];
    current = name;
  }
  [text appendFormat:@"  -> %@\n", current];
  return text;
}
