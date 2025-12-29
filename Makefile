CC = clang
CFLAGS = -fobjc-arc -O3
FRAMEWORKS = -framework Foundation -framework Metal -framework MetalPerformanceShadersGraph
LDFLAGS = ${FRAMEWORKS}
TARGETS  = measure_conv_fp16 measure_conv

all: ${TARGETS}

measure_conv_fp16: measure_conv_fp16.m

measure_conv: measure_conv.m

measure_conv_universal: measure_conv_universal.m

clean:
	rm -f ${TARGETS}
