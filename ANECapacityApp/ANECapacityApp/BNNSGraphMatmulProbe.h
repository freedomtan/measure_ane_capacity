#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/// On-device port of the root repo's bnns_graph_matmul_probe.m (see
/// docs/CPU_Matrix_Convolution_Capability.md Finding 12): builds a matmul
/// whose weight is a genuine function input (via MILBuildMatmulWeightInputSpec),
/// compiles it with the normal CoreML compiler, then executes the resulting
/// .mlmodelc directly through BNNSGraphCompileFromFile + BNNSGraphContext*,
/// bypassing MLModel/CoreML's prediction path entirely. Triggered by
/// --native-bnns-graph-matmul (see ANECapacityApp.swift).
FOUNDATION_EXPORT void BNNSGraphMatmulProbeRun(void);

NS_ASSUME_NONNULL_END
