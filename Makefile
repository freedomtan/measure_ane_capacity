CC = clang
CFLAGS = -fobjc-arc -O3
FRAMEWORKS = -framework Foundation -framework Metal -framework MetalPerformanceShadersGraph
LDFLAGS = ${FRAMEWORKS}
TARGETS  = measure_conv_fp16 measure_conv measure_conv_universal measure_conv_swift

all: ${TARGETS}

measure_conv_fp16: measure_conv_fp16.m

measure_conv: measure_conv.m

measure_conv_universal: measure_conv_universal.m

measure_conv_swift: measure_conv.swift
	swiftc -O measure_conv.swift -o measure_conv_swift

clean:
	rm -f ${TARGETS} measure_conv_ios
