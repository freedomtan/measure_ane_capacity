CC = clang
CFLAGS = -fobjc-arc -O3
FRAMEWORKS = -framework Foundation -framework Metal -framework MetalPerformanceShadersGraph
LDFLAGS = ${FRAMEWORKS}
ANE_FRAMEWORKS = -F/System/Library/PrivateFrameworks -framework AppleNeuralEngine -framework IOSurface -framework IOKit -framework Security
TARGETS  = measure_conv_fp16 measure_conv measure_conv_universal measure_conv_qdq measure_conv_swift measure_ane_pmu

all: ${TARGETS}

measure_ane_pmu: measure_ane_pmu.m
	$(CC) $(CFLAGS) $(FRAMEWORKS) $(ANE_FRAMEWORKS) measure_ane_pmu.m -o measure_ane_pmu
	codesign -s - --entitlements entitlements.plist -f measure_ane_pmu

measure_conv_fp16: measure_conv_fp16.m

measure_conv: measure_conv.m

measure_conv_universal: measure_conv_universal.m

measure_conv_qdq: measure_conv_qdq.m

measure_conv_swift: measure_conv.swift
	swiftc -O measure_conv.swift -o measure_conv_swift

clean:
	rm -f ${TARGETS} measure_conv_ios
	rm -rf packages/

app:
	xcodebuild -project ANECapacityApp/ANECapacityApp.xcodeproj -scheme ANECapacityApp -sdk iphoneos -destination 'generic/platform=iOS' CODE_SIGNING_ALLOWED=NO CODE_SIGN_IDENTITY="" CODE_SIGNING_REQUIRED=NO build

app-sim:
	xcodebuild -project ANECapacityApp/ANECapacityApp.xcodeproj -scheme ANECapacityApp -sdk iphonesimulator -destination 'generic/platform=iOS Simulator' build
