CC = clang
CFLAGS = -fobjc-arc -O3
FRAMEWORKS = -framework Foundation -framework Metal -framework MetalPerformanceShadersGraph
LDFLAGS = ${FRAMEWORKS}
ANE_FRAMEWORKS = -F/System/Library/PrivateFrameworks -framework AppleNeuralEngine -framework IOSurface -framework IOKit -framework Security

# measure_conv_coreml deliberately links only CoreML: pulling in MPSGraph would
# muddy a benchmark whose point is to compare the two frameworks.
COREML_FRAMEWORKS = -framework Foundation -framework CoreML

# measure_conv_coreml builds its MIL program at runtime from coremltools' own
# schema. coremltools ships no C++ builder (mb.program is Python only), but it
# does ship the .proto files -- vendored under third_party/coremltools/format --
# and checks in the protoc-generated C++ classes. We regenerate those locally
# because the checked-in copies are pinned to protobuf 3.x.
PROTOC = protoc
PROTO_DIR = third_party/coremltools/format
PROTO_OUT = build/proto
PROTO_SRCS = $(wildcard $(PROTO_DIR)/*.proto)
PROTO_CC = $(patsubst $(PROTO_DIR)/%.proto,$(PROTO_OUT)/%.pb.cc,$(PROTO_SRCS))
PROTO_OBJS = $(PROTO_CC:.cc=.o)
# protobuf 4+ links against abseil, so the library list is long and version
# dependent: ask pkg-config. The fallbacks cover Homebrew and MacPorts prefixes
# for an install without a .pc file.
PROTOBUF_INC = $(shell pkg-config --cflags protobuf-lite 2>/dev/null || \
	echo -I/opt/homebrew/include -I/opt/local/include -I/usr/local/include)
PROTOBUF_LIB = $(shell pkg-config --libs protobuf-lite 2>/dev/null || \
	echo -L/opt/homebrew/lib -L/opt/local/lib -L/usr/local/lib -lprotobuf-lite)
# protobuf 33's generated code trips absl's own deprecation attributes.
PROTO_CXXFLAGS = -std=c++17 -O2 -Wno-deprecated-declarations
TARGETS  = measure_conv_fp16 measure_conv measure_conv_universal measure_matmul_universal measure_conv_qdq measure_conv_swift measure_ane_pmu measure_conv_coreml measure_conv_fp8

all: ${TARGETS}

measure_ane_pmu: measure_ane_pmu.m
	$(CC) $(CFLAGS) $(FRAMEWORKS) $(ANE_FRAMEWORKS) measure_ane_pmu.m -o measure_ane_pmu
	codesign -s - --entitlements entitlements.plist -f measure_ane_pmu

measure_conv_fp8: measure_conv_fp8.m

measure_conv_fp16: measure_conv_fp16.m

measure_conv: measure_conv.m

measure_conv_universal: measure_conv_universal.m

measure_matmul_universal: measure_matmul_universal.m

measure_conv_qdq: measure_conv_qdq.m

measure_conv_swift: measure_conv.swift
	swiftc -O measure_conv.swift -o measure_conv_swift

# One protoc run emits all 33 files; the stamp keeps make from re-running it per
# target. Model.proto imports every other schema, so all of them must be built.
$(PROTO_OUT)/.stamp: $(PROTO_SRCS)
	@command -v $(PROTOC) >/dev/null || \
		{ echo "error: protoc not found. Install with: brew install protobuf"; exit 1; }
	@mkdir -p $(PROTO_OUT)
	$(PROTOC) --proto_path=$(PROTO_DIR) --cpp_out=$(PROTO_OUT) $(PROTO_SRCS)
	@touch $@

$(PROTO_CC): $(PROTO_OUT)/.stamp

$(PROTO_OUT)/%.pb.o: $(PROTO_OUT)/%.pb.cc
	$(CXX) $(PROTO_CXXFLAGS) -I$(PROTO_OUT) $(PROTOBUF_INC) -c $< -o $@

MILSpecBuilder.o: MILSpecBuilder.mm MILSpecBuilder.h $(PROTO_OUT)/.stamp
	$(CXX) $(PROTO_CXXFLAGS) -fobjc-arc -I$(PROTO_OUT) $(PROTOBUF_INC) \
		-c MILSpecBuilder.mm -o MILSpecBuilder.o

measure_conv_coreml: measure_conv_coreml.m MILSpecBuilder.o $(PROTO_OBJS)
	$(CC) $(CFLAGS) $(COREML_FRAMEWORKS) $(PROTOBUF_LIB) -lc++ \
		measure_conv_coreml.m MILSpecBuilder.o $(PROTO_OBJS) \
		-o measure_conv_coreml

.PHONY: all clean app

clean:
	rm -f ${TARGETS} measure_conv_ios MILSpecBuilder.o
	rm -rf packages/ models/ build/

app:
	xcodebuild -project ANECapacityApp/ANECapacityApp.xcodeproj -scheme ANECapacityApp -sdk iphoneos -destination 'generic/platform=iOS' CODE_SIGNING_ALLOWED=NO CODE_SIGN_IDENTITY="" CODE_SIGNING_REQUIRED=NO build

