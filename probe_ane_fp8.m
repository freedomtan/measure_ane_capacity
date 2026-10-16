#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

@interface MPSGraphCompilationDescriptor (Private)
@property (nonatomic, assign) unsigned long long preferredDevice;
@end

int main(int argc, char *argv[]) {
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> q = [device newCommandQueue];

    void (^testOp)(NSString *, void(^)(MPSGraph *, MPSGraphTensor **, MPSGraphTensor **, NSArray **), MPSDataType, MPSDataType) =
    ^(NSString *testName, void(^buildGraph)(MPSGraph *, MPSGraphTensor **, MPSGraphTensor **, NSArray **), MPSDataType inType, MPSDataType outType) {
      @autoreleasepool {
        MPSGraph *graph = [MPSGraph new];
        MPSGraphTensor *inTensor = nil;
        MPSGraphTensor *outTensor = nil;
        NSArray *shape = nil;
        buildGraph(graph, &inTensor, &outTensor, &shape);

        MPSGraphCompilationDescriptor *cd = [MPSGraphCompilationDescriptor new];
        cd.preferredDevice = 2; // ANE
        cd.optimizationLevel = MPSGraphOptimizationLevel1;

        MPSGraphExecutable *exe = [graph compileWithDevice:[MPSGraphDevice deviceWithMTLDevice:device]
                                                     feeds:@{inTensor: [[MPSGraphShapedType alloc] initWithShape:shape dataType:inType]}
                                             targetTensors:@[outTensor]
                                          targetOperations:nil
                                     compilationDescriptor:cd];
        if (!exe) {
          NSLog(@"[%@] Compilation FAILED", testName);
          return;
        }

        size_t elemCount = 256 * 256;
        size_t inBytes = elemCount * (inType == MPSDataTypeFloat8e4m3 ? 1 : 2);
        size_t outBytes = elemCount * (outType == MPSDataTypeFloat8e4m3 ? 1 : 2);

        id<MTLBuffer> iBuf = [device newBufferWithLength:inBytes options:0];
        id<MTLBuffer> oBuf = [device newBufferWithLength:outBytes options:0];

        // Fill input with non-zero
        if (inType == MPSDataTypeFloat8e4m3) {
          uint8_t *p = (uint8_t *)iBuf.contents;
          for (size_t i = 0; i < elemCount; i++) p[i] = 0x38; // 1.0 in E4M3 (exp=7, mant=0)
        } else {
          uint16_t *p = (uint16_t *)iBuf.contents;
          for (size_t i = 0; i < elemCount; i++) p[i] = 0x3C00; // 1.0 in FP16
        }

        memset(oBuf.contents, 0xEE, outBytes); // sentinel fill

        MPSGraphTensorData *iData = [[MPSGraphTensorData alloc] initWithMTLBuffer:iBuf shape:shape dataType:inType];
        MPSGraphTensorData *oData = [[MPSGraphTensorData alloc] initWithMTLBuffer:oBuf shape:shape dataType:outType];

        MPSGraphExecutableExecutionDescriptor *ed = [MPSGraphExecutableExecutionDescriptor new];
        ed.waitUntilCompleted = YES;

        [exe runWithMTLCommandQueue:q inputsArray:@[iData] resultsArray:@[oData] executionDescriptor:ed];

        size_t zeroCount = 0;
        if (outType == MPSDataTypeFloat8e4m3) {
          uint8_t *p = (uint8_t *)oBuf.contents;
          for (size_t i = 0; i < elemCount; i++) if (p[i] == 0) zeroCount++;
          NSLog(@"[%@] outType=FP8 Check: [0x%02x, 0x%02x, 0x%02x, 0x%02x] zeros: %lu/%lu",
                testName, p[0], p[1], p[2], p[3], zeroCount, elemCount);
        } else {
          uint16_t *p = (uint16_t *)oBuf.contents;
          for (size_t i = 0; i < elemCount; i++) if (p[i] == 0) zeroCount++;
          NSLog(@"[%@] outType=FP16 Check: [0x%04x, 0x%04x, 0x%04x, 0x%04x] zeros: %lu/%lu",
                testName, p[0], p[1], p[2], p[3], zeroCount, elemCount);
        }
      }
    };

    NSArray *shape = @[@1, @1, @256, @256];

    // Test 1: FP8 placeholder -> Dequantize to FP16 -> Output FP16
    testOp(@"Test 1: FP8 in -> Dequant(scale=1.0) -> FP16 out", ^(MPSGraph *g, MPSGraphTensor **inT, MPSGraphTensor **outT, NSArray **s) {
      *s = shape;
      *inT = [g placeholderWithShape:shape dataType:MPSDataTypeFloat8e4m3 name:@"in"];
      *outT = [g dequantizeTensor:*inT scale:1.0 zeroPoint:0.0 dataType:MPSDataTypeFloat16 name:@"dequant"];
    }, MPSDataTypeFloat8e4m3, MPSDataTypeFloat16);

    // Test 2: FP16 in -> Quantize to FP8 (scale=1.0) -> Output FP8
    testOp(@"Test 2: FP16 in -> Quant(scale=1.0) -> FP8 out", ^(MPSGraph *g, MPSGraphTensor **inT, MPSGraphTensor **outT, NSArray **s) {
      *s = shape;
      *inT = [g placeholderWithShape:shape dataType:MPSDataTypeFloat16 name:@"in"];
      *outT = [g quantizeTensor:*inT scale:1.0 zeroPoint:0.0 dataType:MPSDataTypeFloat8e4m3 name:@"quant"];
    }, MPSDataTypeFloat16, MPSDataTypeFloat8e4m3);

    // Test 3: FP16 in -> Quant(scale=1.0) -> Dequant(scale=1.0) -> FP16 out
    testOp(@"Test 3: FP16 in -> Quant(scale=1.0) -> Dequant(scale=1.0) -> FP16 out", ^(MPSGraph *g, MPSGraphTensor **inT, MPSGraphTensor **outT, NSArray **s) {
      *s = shape;
      *inT = [g placeholderWithShape:shape dataType:MPSDataTypeFloat16 name:@"in"];
      MPSGraphTensor *fp8 = [g quantizeTensor:*inT scale:1.0 zeroPoint:0.0 dataType:MPSDataTypeFloat8e4m3 name:@"quant"];
      *outT = [g dequantizeTensor:fp8 scale:1.0 zeroPoint:0.0 dataType:MPSDataTypeFloat16 name:@"dequant"];
    }, MPSDataTypeFloat16, MPSDataTypeFloat16);

    // Test 4: FP8 in -> Dequant(scale=1.0) -> Quant(scale=1.0) -> FP8 out
    testOp(@"Test 4: FP8 in -> Dequant(scale=1.0) -> Quant(scale=1.0) -> FP8 out", ^(MPSGraph *g, MPSGraphTensor **inT, MPSGraphTensor **outT, NSArray **s) {
      *s = shape;
      *inT = [g placeholderWithShape:shape dataType:MPSDataTypeFloat8e4m3 name:@"in"];
      MPSGraphTensor *fp16 = [g dequantizeTensor:*inT scale:1.0 zeroPoint:0.0 dataType:MPSDataTypeFloat16 name:@"dequant"];
      *outT = [g quantizeTensor:fp16 scale:1.0 zeroPoint:0.0 dataType:MPSDataTypeFloat8e4m3 name:@"quant"];
    }, MPSDataTypeFloat8e4m3, MPSDataTypeFloat8e4m3);

    // Test 5: What about INT8 for comparison?
    testOp(@"Test 5 (Control): INT8 in -> Dequant(scale=1.0) -> FP16 out", ^(MPSGraph *g, MPSGraphTensor **inT, MPSGraphTensor **outT, NSArray **s) {
      *s = shape;
      *inT = [g placeholderWithShape:shape dataType:MPSDataTypeInt8 name:@"in"];
      *outT = [g dequantizeTensor:*inT scale:1.0 zeroPoint:0.0 dataType:MPSDataTypeFloat16 name:@"dequant"];
    }, MPSDataTypeInt8, MPSDataTypeFloat16);

    // Test 6: INT8 in -> Dequant(scale=1.0) -> Quant(scale=1.0) -> INT8 out
    testOp(@"Test 6 (Control): INT8 in -> Dequant(scale=1.0) -> Quant(scale=1.0) -> INT8 out", ^(MPSGraph *g, MPSGraphTensor **inT, MPSGraphTensor **outT, NSArray **s) {
      *s = shape;
      *inT = [g placeholderWithShape:shape dataType:MPSDataTypeInt8 name:@"in"];
      MPSGraphTensor *fp16 = [g dequantizeTensor:*inT scale:1.0 zeroPoint:0.0 dataType:MPSDataTypeFloat16 name:@"dequant"];
      *outT = [g quantizeTensor:fp16 scale:1.0 zeroPoint:0.0 dataType:MPSDataTypeInt8 name:@"quant"];
    }, MPSDataTypeInt8, MPSDataTypeInt8);

    return 0;
  }
}
