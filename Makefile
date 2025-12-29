CC = clang
CFLAGS = -fobjc-arc -O3
FRAMEWORKS = -framework Foundation -framework Metal -framework MetalPerformanceShadersGraph
LDFLAGS = ${FRAMEWORKS}
TARGETS  = measure_conv_fp16

all: ${TARGETS}

measure_conv_fp16: measure_conv_fp16.m

clean:
	rm -f ${TARGETS}
