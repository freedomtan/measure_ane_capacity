import Foundation
import Metal
import MetalPerformanceShadersGraph

/// Live runtime comparison of the three places `supportsMXU` can be
/// answered, per docs/MTLDeviceFeatureQueries_initWithDevice_Full_Decompilation.md
/// and Findings 1/1b/2/2b:
///   1. `[device supportsMXU]` -- the real per-AGX-generation-driver-class
///      hardcoded override (ground truth: "mov w0,#imm; ret").
///   2. `[[device featureQueries] familySupportsMXU]` -- the cached byte on
///      the public `MTLDeviceFeatureQueries` object, populated from (1).
///   3. `[MPSGraphDeviceDescriptor supportsMXU]` -- MPSGraph's own
///      independent chip-codename string-reparse (Finding 1), which Finding
///      2b found returns NO when built via `initWithMPSGraphDevice:`.
enum SupportsMXUProbe {
    static func run() {
        guard let device = MTLCreateSystemDefaultDevice() else {
            print("[SupportsMXUProbe] No Metal device"); return
        }
        print("[SupportsMXUProbe] Device: \(device.name)")

        func rawBool(_ obj: AnyObject, _ selName: String) -> Bool? {
            let sel = Selector((selName))
            guard obj.responds(to: sel) else { return nil }
            typealias Fn = @convention(c) (AnyObject, Selector) -> Bool
            let imp = unsafeBitCast(class_getMethodImplementation(object_getClass(obj), sel), to: Fn.self)
            return imp(obj, sel)
        }
        func rawObj(_ obj: AnyObject, _ selName: String) -> AnyObject? {
            let sel = Selector((selName))
            guard obj.responds(to: sel) else { return nil }
            typealias Fn = @convention(c) (AnyObject, Selector) -> AnyObject?
            let imp = unsafeBitCast(class_getMethodImplementation(object_getClass(obj), sel), to: Fn.self)
            return imp(obj, sel)
        }

        // 1. Ground truth: the real AGX driver class's own hardcoded override.
        let deviceSupportsMXU = rawBool(device as AnyObject, "supportsMXU")
        let deviceSupportsFP8 = rawBool(device as AnyObject, "supportsFP8")
        print("[SupportsMXUProbe] [1] [device supportsMXU]=\(String(describing: deviceSupportsMXU)) [device supportsFP8]=\(String(describing: deviceSupportsFP8))")

        // 2. The cached byte on the public MTLDeviceFeatureQueries object.
        if let fq = rawObj(device as AnyObject, "featureQueries") {
            let familySupportsMXU = rawBool(fq, "familySupportsMXU")
            let familySupportsFP8 = rawBool(fq, "familySupportsFP8")
            print("[SupportsMXUProbe] [2] featureQueries.familySupportsMXU=\(String(describing: familySupportsMXU)) familySupportsFP8=\(String(describing: familySupportsFP8))")
        } else {
            print("[SupportsMXUProbe] [2] -featureQueries not available on this device")
        }

        let gDevice = MPSGraphDevice(mtlDevice: device)

        guard let descClass = NSClassFromString("MPSGraphDeviceDescriptor") as? NSObject.Type else {
            print("[SupportsMXUProbe] MPSGraphDeviceDescriptor class not found"); return
        }
        let allocSel = Selector(("alloc"))
        typealias AllocFn = @convention(c) (AnyObject, Selector) -> AnyObject
        let allocImp = unsafeBitCast(class_getMethodImplementation(object_getClass(descClass), allocSel), to: AllocFn.self)
        let allocated = allocImp(descClass, allocSel)

        let initSel = Selector(("initWithMPSGraphDevice:"))
        typealias InitFn = @convention(c) (AnyObject, Selector, AnyObject) -> AnyObject?
        guard let initMethod = class_getInstanceMethod(descClass, initSel) else {
            print("[SupportsMXUProbe] initWithMPSGraphDevice: not found"); return
        }
        let initImp = unsafeBitCast(method_getImplementation(initMethod), to: InitFn.self)
        guard let descInstance = initImp(allocated, initSel, gDevice as AnyObject) else {
            print("[SupportsMXUProbe] Failed to construct MPSGraphDeviceDescriptor"); return
        }

        func boolProp(_ name: String) -> Bool? {
            let sel = Selector((name))
            guard let m = class_getInstanceMethod(descClass, sel) else { return nil }
            typealias BoolFn = @convention(c) (AnyObject, Selector) -> Bool
            let imp = unsafeBitCast(method_getImplementation(m), to: BoolFn.self)
            return imp(descInstance, sel)
        }
        func stringProp(_ name: String) -> String? {
            let sel = Selector((name))
            guard let m = class_getInstanceMethod(descClass, sel) else { return nil }
            typealias StrFn = @convention(c) (AnyObject, Selector) -> AnyObject?
            let imp = unsafeBitCast(method_getImplementation(m), to: StrFn.self)
            return imp(descInstance, sel) as? String
        }

        let arch = stringProp("architecture") ?? "<unavailable>"
        let supportsMXU = boolProp("supportsMXU")
        let hasANE = boolProp("hasANE")
        print("[SupportsMXUProbe] [3] MPSGraphDeviceDescriptor: architecture=\(arch) supportsMXU=\(String(describing: supportsMXU)) hasANE=\(String(describing: hasANE))")
    }
}
