import Foundation
import Metal
import MetalPerformanceShadersGraph

/// Live runtime check of `-[MPSGraphDeviceDescriptor supportsMXU]` on this
/// device's actual MPSGraphDevice, as a cross-check against Findings 1/1b's
/// static disassembly (which only establishes what the code *would* do for
/// a given chip-codename string, not that this device's codename actually
/// parses the way assumed).
enum SupportsMXUProbe {
    static func run() {
        guard let device = MTLCreateSystemDefaultDevice() else {
            print("[SupportsMXUProbe] No Metal device"); return
        }
        print("[SupportsMXUProbe] Device: \(device.name)")

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
        print("[SupportsMXUProbe] architecture=\(arch) supportsMXU=\(String(describing: supportsMXU)) hasANE=\(String(describing: hasANE))")
    }
}
