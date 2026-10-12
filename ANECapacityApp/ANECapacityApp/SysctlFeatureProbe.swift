import Foundation

/// Queries ARM feature sysctls directly on-device, to compare against the Mac's values
/// (see docs/CPU_Matrix_Convolution_Capability.md Finding 6) and specifically check
/// FEAT_SME_F8F16 on A20 Pro (iPhone 18 Pro), which Finding 6's follow-up suspects may be
/// the first chip in this repo to report it as supported.
enum SysctlFeatureProbe {
    static func run() {
        let names = [
            "hw.optional.neon",
            "hw.optional.arm.FEAT_SME",
            "hw.optional.arm.FEAT_SME2",
            "hw.optional.arm.FEAT_SME2p1",
            "hw.optional.arm.FEAT_SME_F8F16",
            "hw.optional.arm.FEAT_SME_F8F32",
        ]
        for name in names {
            var value: Int32 = -1
            var size = MemoryLayout<Int32>.size
            let rc = sysctlbyname(name, &value, &size, nil, 0)
            if rc == 0 {
                print("[SysctlFeatureProbe] \(name) = \(value)")
            } else {
                print("[SysctlFeatureProbe] \(name) = <not present> (errno \(errno))")
            }
        }
    }
}
