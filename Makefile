CC = clang
CFLAGS = -fobjc-arc -O3
FRAMEWORKS = -framework Foundation -framework Metal -framework MetalPerformanceShadersGraph
LDFLAGS = ${FRAMEWORKS}
ANE_FRAMEWORKS = -F/System/Library/PrivateFrameworks -framework AppleNeuralEngine -framework IOSurface -framework IOKit -framework Security
# measure_conv_coreml deliberately links only CoreML: pulling in MPSGraph would
# muddy a benchmark whose point is to compare the two frameworks.
COREML_FRAMEWORKS = -framework Foundation -framework CoreML
PYTHON = python3
TARGETS  = measure_conv_fp16 measure_conv measure_conv_universal measure_matmul_universal measure_conv_qdq measure_conv_swift measure_ane_pmu measure_conv_coreml

all: ${TARGETS}

measure_ane_pmu: measure_ane_pmu.m
	$(CC) $(CFLAGS) $(FRAMEWORKS) $(ANE_FRAMEWORKS) measure_ane_pmu.m -o measure_ane_pmu
	codesign -s - --entitlements entitlements.plist -f measure_ane_pmu

measure_conv_fp16: measure_conv_fp16.m

measure_conv: measure_conv.m

measure_conv_universal: measure_conv_universal.m

measure_matmul_universal: measure_matmul_universal.m

measure_conv_qdq: measure_conv_qdq.m

measure_conv_swift: measure_conv.swift
	swiftc -O measure_conv.swift -o measure_conv_swift

measure_conv_coreml: measure_conv_coreml.m
	$(CC) $(CFLAGS) $(COREML_FRAMEWORKS) measure_conv_coreml.m -o measure_conv_coreml

# Generate the MIL-authored .mlpackage models that measure_conv_coreml runs.
models:
	@$(PYTHON) -c "import coremltools" 2>/dev/null || \
		{ echo "error: coremltools not found. Install with: $(PYTHON) -m pip install coremltools"; exit 1; }
	$(PYTHON) tools/gen_conv_mil.py $(MODEL_ARGS)

.PHONY: all clean app models

clean:
	rm -f ${TARGETS} measure_conv_ios
	rm -rf packages/ models/

app:
	xcodebuild -project ANECapacityApp/ANECapacityApp.xcodeproj -scheme ANECapacityApp -sdk iphoneos -destination 'generic/platform=iOS' CODE_SIGNING_ALLOWED=NO CODE_SIGN_IDENTITY="" CODE_SIGNING_REQUIRED=NO build

