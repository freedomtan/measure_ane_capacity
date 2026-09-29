//
//  MILSpecBuilder.m
//  ANECapacityWithMILApp
//
//  Self-contained dynamic CoreML MIL model specification encoder in pure Objective-C.
//  Encodes the protobuf wire format directly into NSData with zero external libraries.
//

#import "MILSpecBuilder.h"

NSString *const MILSpecBuilderErrorDomain = @"MILSpecBuilder";

// MIL Data Types
static const uint32_t kMIL_STRING = 2;
static const uint32_t kMIL_FLOAT16 = 10;
static const uint32_t kMIL_INT8 = 21;
static const uint32_t kMIL_INT32 = 23;
static const uint32_t kMIL_FLOAT8E4M3FN = 40;

static const uint16_t kFP16PlusOneThirtySecond  = 0x2800; // 1/32
static const uint16_t kFP16MinusOneThirtySecond = 0xA800; // -1/32
static const uint16_t kFP16PlusOneSixteenth     = 0x2C00; // 1/16
static const uint16_t kFP16MinusOneSixteenth    = 0xAC00; // -1/16

static const uint64_t kWeightSeed = 0x5EED5EED5EED5EEDULL;

#pragma mark - ProtoWriter Helper

@interface ProtoWriter : NSObject
@property (nonatomic, readonly) NSMutableData *data;
- (instancetype)init;
- (void)writeVarint:(uint64_t)val;
- (void)writeTag:(uint32_t)field wireType:(uint8_t)wire;
- (void)writeVarintField:(uint32_t)field val:(uint64_t)val;
- (void)writeBytesField:(uint32_t)field bytes:(NSData *)bytes;
- (void)writeStringField:(uint32_t)field string:(NSString *)str;
- (void)writeMessageField:(uint32_t)field message:(ProtoWriter *)submsg;
@end

@implementation ProtoWriter {
    NSMutableData *_data;
}

- (instancetype)init {
    if (self = [super init]) {
        _data = [NSMutableData data];
    }
    return self;
}

- (NSMutableData *)data { return _data; }

- (void)writeVarint:(uint64_t)val {
    while (val >= 0x80) {
        uint8_t b = (uint8_t)((val & 0x7f) | 0x80);
        [_data appendBytes:&b length:1];
        val >>= 7;
    }
    uint8_t b = (uint8_t)val;
    [_data appendBytes:&b length:1];
}

- (void)writeTag:(uint32_t)field wireType:(uint8_t)wire {
    [self writeVarint:((uint64_t)field << 3) | wire];
}

- (void)writeVarintField:(uint32_t)field val:(uint64_t)val {
    [self writeTag:field wireType:0];
    [self writeVarint:val];
}

- (void)writeBytesField:(uint32_t)field bytes:(NSData *)bytes {
    [self writeTag:field wireType:2];
    [self writeVarint:bytes.length];
    [_data appendData:bytes];
}

- (void)writeStringField:(uint32_t)field string:(NSString *)str {
    NSData *s = [str dataUsingEncoding:NSUTF8StringEncoding];
    [self writeBytesField:field bytes:s];
}

- (void)writeMessageField:(uint32_t)field message:(ProtoWriter *)submsg {
    [self writeBytesField:field bytes:submsg.data];
}

@end

#pragma mark - MIL Schema Encoders

static ProtoWriter *makeTensorType(uint32_t dataType, NSArray<NSNumber *> *shape) {
    ProtoWriter *tt = [[ProtoWriter alloc] init];
    [tt writeVarintField:1 val:dataType];
    [tt writeVarintField:2 val:shape.count];
    for (NSNumber *dim in shape) {
        ProtoWriter *cd = [[ProtoWriter alloc] init];
        [cd writeVarintField:1 val:dim.unsignedLongLongValue];
        ProtoWriter *d = [[ProtoWriter alloc] init];
        [d writeMessageField:1 message:cd];
        [tt writeMessageField:3 message:d];
    }
    ProtoWriter *vt = [[ProtoWriter alloc] init];
    [vt writeMessageField:1 message:tt];
    return vt;
}

static ProtoWriter *makeStringScalarType(void) {
    ProtoWriter *tt = [[ProtoWriter alloc] init];
    [tt writeVarintField:1 val:kMIL_STRING];
    [tt writeVarintField:2 val:0];
    ProtoWriter *vt = [[ProtoWriter alloc] init];
    [vt writeMessageField:1 message:tt];
    return vt;
}

static ProtoWriter *makeStringValue(NSString *text) {
    ProtoWriter *tv = [[ProtoWriter alloc] init];
    ProtoWriter *repStr = [[ProtoWriter alloc] init];
    [repStr writeStringField:1 string:text];
    [tv writeMessageField:4 message:repStr]; // strings = 4

    ProtoWriter *iv = [[ProtoWriter alloc] init];
    [iv writeMessageField:1 message:tv]; // tensor = 1

    ProtoWriter *val = [[ProtoWriter alloc] init];
    [val writeMessageField:2 message:makeStringScalarType()];
    [val writeMessageField:3 message:iv];
    return val;
}

static ProtoWriter *makeInt32Value(NSArray<NSNumber *> *vals, BOOL scalar) {
    ProtoWriter *packedInts = [[ProtoWriter alloc] init];
    for (NSNumber *v in vals) {
        [packedInts writeVarint:v.intValue];
    }
    ProtoWriter *repInts = [[ProtoWriter alloc] init];
    [repInts writeBytesField:1 bytes:packedInts.data];

    ProtoWriter *tv = [[ProtoWriter alloc] init];
    [tv writeMessageField:2 message:repInts]; // ints = 2

    ProtoWriter *iv = [[ProtoWriter alloc] init];
    [iv writeMessageField:1 message:tv];

    NSArray *shape = scalar ? @[] : @[@(vals.count)];
    ProtoWriter *val = [[ProtoWriter alloc] init];
    [val writeMessageField:2 message:makeTensorType(kMIL_INT32, shape)];
    [val writeMessageField:3 message:iv];
    return val;
}

static ProtoWriter *makeFloat16ScalarValue(uint16_t bits) {
    NSData *b = [NSData dataWithBytes:&bits length:sizeof(uint16_t)];
    ProtoWriter *repBytes = [[ProtoWriter alloc] init];
    [repBytes writeBytesField:1 bytes:b];

    ProtoWriter *tv = [[ProtoWriter alloc] init];
    [tv writeMessageField:7 message:repBytes]; // bytes = 7

    ProtoWriter *iv = [[ProtoWriter alloc] init];
    [iv writeMessageField:1 message:tv];

    ProtoWriter *val = [[ProtoWriter alloc] init];
    [val writeMessageField:2 message:makeTensorType(kMIL_FLOAT16, @[])];
    [val writeMessageField:3 message:iv];
    return val;
}

static ProtoWriter *makeBytesTensorValue(uint32_t dataType, NSArray<NSNumber *> *shape, NSData *bytes) {
    ProtoWriter *repBytes = [[ProtoWriter alloc] init];
    [repBytes writeBytesField:1 bytes:bytes];

    ProtoWriter *tv = [[ProtoWriter alloc] init];
    [tv writeMessageField:7 message:repBytes];

    ProtoWriter *iv = [[ProtoWriter alloc] init];
    [iv writeMessageField:1 message:tv];

    ProtoWriter *val = [[ProtoWriter alloc] init];
    [val writeMessageField:2 message:makeTensorType(dataType, shape)];
    [val writeMessageField:3 message:iv];
    return val;
}

static ProtoWriter *makeNamedValueType(NSString *name, ProtoWriter *valueType) {
    ProtoWriter *nvt = [[ProtoWriter alloc] init];
    [nvt writeStringField:1 string:name];
    [nvt writeMessageField:2 message:valueType];
    return nvt;
}

static ProtoWriter *makeConstOp(NSString *name, ProtoWriter *valueType, ProtoWriter *value) {
    ProtoWriter *op = [[ProtoWriter alloc] init];
    [op writeStringField:1 string:@"const"];
    [op writeMessageField:3 message:makeNamedValueType(name, valueType)];

    // attr "name"
    ProtoWriter *attrName = [[ProtoWriter alloc] init];
    [attrName writeStringField:1 string:@"name"];
    [attrName writeMessageField:2 message:makeStringValue(name)];
    [op writeMessageField:5 message:attrName];

    // attr "val"
    ProtoWriter *attrVal = [[ProtoWriter alloc] init];
    [attrVal writeStringField:1 string:@"val"];
    [attrVal writeMessageField:2 message:value];
    [op writeMessageField:5 message:attrVal];
    return op;
}

static ProtoWriter *makeConstexprShiftScaleOp(NSString *outputName, NSArray<NSNumber *> *shape, NSData *int8Bytes, uint16_t scaleFP16) {
    ProtoWriter *op = [[ProtoWriter alloc] init];
    [op writeStringField:1 string:@"constexpr_blockwise_shift_scale"];

    // input "data"
    {
        ProtoWriter *val = makeBytesTensorValue(kMIL_INT8, shape, int8Bytes);
        ProtoWriter *binding = [[ProtoWriter alloc] init];
        [binding writeMessageField:2 message:val];

        ProtoWriter *arg = [[ProtoWriter alloc] init];
        [arg writeMessageField:1 message:binding];

        ProtoWriter *mapEntry = [[ProtoWriter alloc] init];
        [mapEntry writeStringField:1 string:@"data"];
        [mapEntry writeMessageField:2 message:arg];
        [op writeMessageField:2 message:mapEntry];
    }

    // input "scale" [Co, 1, 1, 1]
    {
        NSUInteger Co = shape[0].unsignedIntegerValue;
        NSMutableData *scaleData = [NSMutableData dataWithLength:Co * sizeof(uint16_t)];
        uint16_t *sPtr = (uint16_t *)scaleData.mutableBytes;
        for (NSUInteger i = 0; i < Co; i++) sPtr[i] = scaleFP16;

        NSArray *scaleShape = @[@(Co), @1, @1, @1];
        ProtoWriter *val = makeBytesTensorValue(kMIL_FLOAT16, scaleShape, scaleData);
        ProtoWriter *binding = [[ProtoWriter alloc] init];
        [binding writeMessageField:2 message:val];

        ProtoWriter *arg = [[ProtoWriter alloc] init];
        [arg writeMessageField:1 message:binding];

        ProtoWriter *mapEntry = [[ProtoWriter alloc] init];
        [mapEntry writeStringField:1 string:@"scale"];
        [mapEntry writeMessageField:2 message:arg];
        [op writeMessageField:2 message:mapEntry];
    }

    ProtoWriter *outType = makeTensorType(kMIL_FLOAT16, shape);
    [op writeMessageField:3 message:makeNamedValueType(outputName, outType)];

    ProtoWriter *attrName = [[ProtoWriter alloc] init];
    [attrName writeStringField:1 string:@"name"];
    [attrName writeMessageField:2 message:makeStringValue(outputName)];
    [op writeMessageField:5 message:attrName];
    return op;
}

static ProtoWriter *makeNonConstOp(NSString *type, NSString *outputName, ProtoWriter *outputType, NSDictionary<NSString *, NSString *> *inputs) {
    ProtoWriter *op = [[ProtoWriter alloc] init];
    [op writeStringField:1 string:type];
    for (NSString *key in inputs) {
        NSString *argName = inputs[key];
        ProtoWriter *binding = [[ProtoWriter alloc] init];
        [binding writeStringField:1 string:argName];

        ProtoWriter *arg = [[ProtoWriter alloc] init];
        [arg writeMessageField:1 message:binding];

        ProtoWriter *mapEntry = [[ProtoWriter alloc] init];
        [mapEntry writeStringField:1 string:key];
        [mapEntry writeMessageField:2 message:arg];
        [op writeMessageField:2 message:mapEntry];
    }
    [op writeMessageField:3 message:makeNamedValueType(outputName, outputType)];

    ProtoWriter *attrName = [[ProtoWriter alloc] init];
    [attrName writeStringField:1 string:@"name"];
    [attrName writeMessageField:2 message:makeStringValue(outputName)];
    [op writeMessageField:5 message:attrName];
    return op;
}

#pragma mark - Weight Generators

static NSData *generateWeightsFP16(MILConvChainConfig config) {
    size_t count = config.channelsOut * config.channelsIn * config.kernel * config.kernel;
    NSMutableData *data = [NSMutableData dataWithLength:count * sizeof(uint16_t)];
    uint16_t *p = (uint16_t *)data.mutableBytes;

    if (config.weightMode == MILWeightModeRepeat) {
        static const uint16_t pattern[4] = {kFP16PlusOneSixteenth, kFP16MinusOneSixteenth, kFP16PlusOneThirtySecond, kFP16MinusOneThirtySecond};
        for (size_t i = 0; i < count; i++) p[i] = pattern[i % 4];
    } else {
        uint64_t state = kWeightSeed;
        for (size_t i = 0; i < count; i++) {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            p[i] = (state & 1) ? kFP16MinusOneThirtySecond : kFP16PlusOneThirtySecond;
        }
    }
    return data;
}

static NSData *generateWeightsINT8(MILConvChainConfig config) {
    size_t count = config.channelsOut * config.channelsIn * config.kernel * config.kernel;
    NSMutableData *data = [NSMutableData dataWithLength:count * sizeof(int8_t)];
    int8_t *p = (int8_t *)data.mutableBytes;

    if (config.weightMode == MILWeightModeRepeat) {
        static const int8_t pattern[4] = {1, -1, 2, -2};
        for (size_t i = 0; i < count; i++) p[i] = pattern[i % 4];
    } else {
        uint64_t state = kWeightSeed;
        for (size_t i = 0; i < count; i++) {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            p[i] = (state & 1) ? -1 : 1;
        }
    }
    return data;
}

static NSData *generateWeightsFP8(MILConvChainConfig config) {
    size_t count = config.channelsOut * config.channelsIn * config.kernel * config.kernel;
    NSMutableData *data = [NSMutableData dataWithLength:count];
    uint8_t *p = (uint8_t *)data.mutableBytes;

    if (config.weightMode == MILWeightModeRepeat) {
        static const uint8_t pattern[4] = {0x30, 0xB0, 0x38, 0xB8};
        for (size_t i = 0; i < count; i++) p[i] = pattern[i % 4];
    } else {
        uint64_t state = kWeightSeed;
        for (size_t i = 0; i < count; i++) {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            p[i] = (state & 1) ? 0xB0 : 0x30; // ±0.5 in FP8 E4M3
        }
    }
    return data;
}

#pragma mark - Public API

NSString *MILConvChainInputName(void) { return @"x"; }

NSString *MILConvChainOutputName(MILConvChainConfig config) {
    return [NSString stringWithFormat:@"conv_%lu", (unsigned long)(config.layers - 1)];
}

NSString *MILPrecisionName(MILPrecision precision) {
    switch (precision) {
        case MILPrecisionINT8: return @"INT8";
        case MILPrecisionFP8:  return @"FP8";
        case MILPrecisionFP16:
        default:               return @"FP16";
    }
}

NSData *MILBuildConvChainSpec(MILConvChainConfig config, NSError **error) {
    if (config.batch == 0 || config.channelsIn == 0 || config.channelsOut == 0 ||
        config.height == 0 || config.width == 0 || config.kernel == 0 || config.layers == 0) {
        if (error) {
            *error = [NSError errorWithDomain:MILSpecBuilderErrorDomain code:1 userInfo:@{NSLocalizedDescriptionKey: @"Dimensions must be greater than zero."}];
        }
        return nil;
    }

    NSString *opset = (config.precision == MILPrecisionFP8) ? @"CoreML9" : @"CoreML8";
    int32_t specVersion = (config.precision == MILPrecisionFP8) ? 10 : 9;

    NSString *finalOut = MILConvChainOutputName(config);
    NSArray *wShape = @[@(config.channelsOut), @(config.channelsIn), @(config.kernel), @(config.kernel)];
    NSArray *actShape = @[@(config.batch), @(config.channelsOut), @(config.height), @(config.width)];
    NSArray *inShape = @[@(config.batch), @(config.channelsIn), @(config.height), @(config.width)];

    ProtoWriter *block = [[ProtoWriter alloc] init];
    [block writeStringField:2 string:finalOut]; // outputs = 2

    if (config.precision == MILPrecisionINT8) {
        NSData *wData = generateWeightsINT8(config);
        [block writeMessageField:3 message:makeConstexprShiftScaleOp(@"weights", wShape, wData, kFP16PlusOneThirtySecond)];

        [block writeMessageField:3 message:makeConstOp(@"act_scale", makeTensorType(kMIL_FLOAT16, @[]), makeFloat16ScalarValue(kFP16PlusOneThirtySecond))];
        [block writeMessageField:3 message:makeConstOp(@"dtype_int8", makeStringScalarType(), makeStringValue(@"int8"))];
    } else if (config.precision == MILPrecisionFP8) {
        NSData *wData = generateWeightsFP8(config);
        ProtoWriter *wVal = makeBytesTensorValue(kMIL_FLOAT8E4M3FN, wShape, wData);
        [block writeMessageField:3 message:makeConstOp(@"raw_weights", makeTensorType(kMIL_FLOAT8E4M3FN, wShape), wVal)];

        // w_scale: scalar fp16 1/16 (0x2C00)
        [block writeMessageField:3 message:makeConstOp(@"w_scale", makeTensorType(kMIL_FLOAT16, @[]), makeFloat16ScalarValue(kFP16PlusOneSixteenth))];

        ProtoWriter *fp16WeightsType = makeTensorType(kMIL_FLOAT16, wShape);
        [block writeMessageField:3 message:makeNonConstOp(@"dequantize", @"weights", fp16WeightsType, @{@"input": @"raw_weights", @"scale": @"w_scale"})];

        // act_scale: scalar fp16 1/32 (0x2800)
        [block writeMessageField:3 message:makeConstOp(@"act_scale", makeTensorType(kMIL_FLOAT16, @[]), makeFloat16ScalarValue(kFP16PlusOneThirtySecond))];
        [block writeMessageField:3 message:makeConstOp(@"dtype_fp8", makeStringScalarType(), makeStringValue(@"fp8e4m3fn"))];
    } else {
        NSData *wData = generateWeightsFP16(config);
        ProtoWriter *wVal = makeBytesTensorValue(kMIL_FLOAT16, wShape, wData);
        [block writeMessageField:3 message:makeConstOp(@"weights", makeTensorType(kMIL_FLOAT16, wShape), wVal)];
    }

    // Conv constants
    [block writeMessageField:3 message:makeConstOp(@"strides", makeTensorType(kMIL_INT32, @[@2]), makeInt32Value(@[@1, @1], NO))];
    [block writeMessageField:3 message:makeConstOp(@"dilations", makeTensorType(kMIL_INT32, @[@2]), makeInt32Value(@[@1, @1], NO))];
    [block writeMessageField:3 message:makeConstOp(@"pad", makeTensorType(kMIL_INT32, @[@4]), makeInt32Value(@[@0, @0, @0, @0], NO))];
    [block writeMessageField:3 message:makeConstOp(@"groups", makeTensorType(kMIL_INT32, @[]), makeInt32Value(@[@1], YES))];
    [block writeMessageField:3 message:makeConstOp(@"pad_type", makeStringScalarType(), makeStringValue(@"same"))];

    ProtoWriter *actType = makeTensorType(kMIL_FLOAT16, actShape);
    ProtoWriter *int8ActType = makeTensorType(kMIL_INT8, actShape);
    ProtoWriter *fp8ActType = makeTensorType(kMIL_FLOAT8E4M3FN, actShape);

    NSString *curr = MILConvChainInputName();
    for (NSUInteger i = 0; i < config.layers; i++) {
        NSString *convIn = curr;
        if (config.precision == MILPrecisionINT8) {
            NSString *qName = [NSString stringWithFormat:@"quant_%lu", (unsigned long)i];
            [block writeMessageField:3 message:makeNonConstOp(@"quantize", qName, int8ActType, @{@"input": curr, @"scale": @"act_scale", @"output_dtype": @"dtype_int8"})];
            NSString *dqName = [NSString stringWithFormat:@"dequant_%lu", (unsigned long)i];
            [block writeMessageField:3 message:makeNonConstOp(@"dequantize", dqName, actType, @{@"input": qName, @"scale": @"act_scale"})];
            convIn = dqName;
        } else if (config.precision == MILPrecisionFP8) {
            NSString *qName = [NSString stringWithFormat:@"quant_%lu", (unsigned long)i];
            [block writeMessageField:3 message:makeNonConstOp(@"quantize", qName, fp8ActType, @{@"input": curr, @"scale": @"act_scale", @"output_dtype": @"dtype_fp8"})];
            NSString *dqName = [NSString stringWithFormat:@"dequant_%lu", (unsigned long)i];
            [block writeMessageField:3 message:makeNonConstOp(@"dequantize", dqName, actType, @{@"input": qName, @"scale": @"act_scale"})];
            convIn = dqName;
        }

        NSString *cName = [NSString stringWithFormat:@"conv_%lu", (unsigned long)i];
        NSDictionary *inputs = @{
            @"x": convIn,
            @"weight": @"weights",
            @"strides": @"strides",
            @"pad_type": @"pad_type",
            @"pad": @"pad",
            @"dilations": @"dilations",
            @"groups": @"groups"
        };
        [block writeMessageField:3 message:makeNonConstOp(@"conv", cName, actType, inputs)];
        curr = cName;
    }

    // Function
    ProtoWriter *function = [[ProtoWriter alloc] init];
    [function writeMessageField:1 message:makeNamedValueType(MILConvChainInputName(), makeTensorType(kMIL_FLOAT16, inShape))];
    [function writeStringField:2 string:opset];

    ProtoWriter *bsEntry = [[ProtoWriter alloc] init];
    [bsEntry writeStringField:1 string:opset];
    [bsEntry writeMessageField:2 message:block];
    [function writeMessageField:3 message:bsEntry];

    // Program
    ProtoWriter *program = [[ProtoWriter alloc] init];
    [program writeVarintField:1 val:1];

    ProtoWriter *fnEntry = [[ProtoWriter alloc] init];
    [fnEntry writeStringField:1 string:@"main"];
    [fnEntry writeMessageField:2 message:function];
    [program writeMessageField:2 message:fnEntry];

    // ModelDescription
    ProtoWriter *desc = [[ProtoWriter alloc] init];
    {
        ProtoWriter *fd = [[ProtoWriter alloc] init];
        [fd writeStringField:1 string:MILConvChainInputName()];
        ProtoWriter *aft = [[ProtoWriter alloc] init];
        for (NSNumber *d in inShape) [aft writeVarintField:1 val:d.longLongValue];
        [aft writeVarintField:2 val:65552]; // FLOAT16
        ProtoWriter *ft = [[ProtoWriter alloc] init];
        [ft writeMessageField:5 message:aft];
        [fd writeMessageField:3 message:ft];
        [desc writeMessageField:1 message:fd];
    }
    {
        ProtoWriter *fd = [[ProtoWriter alloc] init];
        [fd writeStringField:1 string:finalOut];
        ProtoWriter *aft = [[ProtoWriter alloc] init];
        for (NSNumber *d in actShape) [aft writeVarintField:1 val:d.longLongValue];
        [aft writeVarintField:2 val:65552]; // FLOAT16
        ProtoWriter *ft = [[ProtoWriter alloc] init];
        [ft writeMessageField:5 message:aft];
        [fd writeMessageField:3 message:ft];
        [desc writeMessageField:10 message:fd];
    }

    // Metadata
    ProtoWriter *metadata = [[ProtoWriter alloc] init];
    NSString *summary = [NSString stringWithFormat:@"%lux conv%lux%lu on [%lu, %lu, %lu, %lu], %@",
                         (unsigned long)config.layers, (unsigned long)config.kernel, (unsigned long)config.kernel,
                         (unsigned long)config.batch, (unsigned long)config.channelsIn,
                         (unsigned long)config.height, (unsigned long)config.width,
                         MILPrecisionName(config.precision)];
    [metadata writeStringField:1 string:summary];

    NSDictionary<NSString *, NSString *> *userDefined = @{
        @"workload.batch": [NSString stringWithFormat:@"%lu", (unsigned long)config.batch],
        @"workload.channels_in": [NSString stringWithFormat:@"%lu", (unsigned long)config.channelsIn],
        @"workload.channels_out": [NSString stringWithFormat:@"%lu", (unsigned long)config.channelsOut],
        @"workload.height": [NSString stringWithFormat:@"%lu", (unsigned long)config.height],
        @"workload.width": [NSString stringWithFormat:@"%lu", (unsigned long)config.width],
        @"workload.kernel": [NSString stringWithFormat:@"%lu", (unsigned long)config.kernel],
        @"workload.layers": [NSString stringWithFormat:@"%lu", (unsigned long)config.layers],
        @"workload.precision": MILPrecisionName(config.precision),
        @"builder": @"MILSpecBuilder (native Objective-C, zero-dependency)"
    };
    for (NSString *key in userDefined) {
        ProtoWriter *entry = [[ProtoWriter alloc] init];
        [entry writeStringField:1 string:key];
        [entry writeStringField:2 string:userDefined[key]];
        [metadata writeMessageField:100 message:entry];
    }
    [desc writeMessageField:100 message:metadata];

    // Model
    ProtoWriter *model = [[ProtoWriter alloc] init];
    [model writeVarintField:1 val:specVersion];
    [model writeMessageField:2 message:desc];
    [model writeMessageField:502 message:program];

    return model.data;
}
