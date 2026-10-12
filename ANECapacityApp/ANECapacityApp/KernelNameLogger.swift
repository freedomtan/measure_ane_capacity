import Foundation
import Metal
import ObjectiveC

/// Open-thread item from docs/GPU_MXU_And_Convolution_Dispatch_Investigation.md: with GPU-frame-
/// capture inspection blocked (`devicectl device copy --domain-type
/// appDataContainer` fails to pull `.gputrace` off a physical device, and Xcode 27
/// has no Window -> Devices and Simulators GUI fallback), this swizzles
/// `-[MTLLibrary newFunctionWithName:]` to log every compiled kernel function name
/// *any* MTLLibrary on this device is asked for -- including MPS's own internal
/// `default.metallib`, which this app never creates or touches directly.
///
/// This works because all `MTLLibrary` instances for a given `MTLDevice`/backend
/// share one concrete Objective-C class under the hood -- so swizzling once, via
/// a disposable library we create ourselves, intercepts lookups made by MPS's own
/// internal library instances too. No GPU capture, no file transfer, no Xcode GUI
/// needed: the log lines show up directly in `devicectl ... --console` output.
///
/// Global (not static-member) storage/functions throughout -- a closure or method
/// referencing a type's static var still counts as "capturing context" and can't
/// be used as a @convention(c) function pointer; only truly free functions and
/// file-scope globals are capture-free.

private var gOrigNewFunctionWithName: IMP?

private func swizzledNewFunctionWithName(_ selfObj: AnyObject, _ sel: Selector, _ name: AnyObject) -> AnyObject? {
    if let n = name as? String {
        print("[KernelNameLogger] newFunctionWithName: \(n)")
    }
    typealias Fn = @convention(c) (AnyObject, Selector, AnyObject) -> AnyObject?
    let orig = unsafeBitCast(gOrigNewFunctionWithName!, to: Fn.self)
    return orig(selfObj, sel, name)
}

private var gKernelNameLoggerInstalled = false

enum KernelNameLogger {
    static func install(on device: MTLDevice) {
        guard !gKernelNameLoggerInstalled else { return }
        gKernelNameLoggerInstalled = true

        let dummySource = "kernel void anecapacity_dummy_kernel() {}"
        guard let dummyLibrary = try? device.makeLibrary(source: dummySource, options: nil) else {
            print("[KernelNameLogger] Failed to create a throwaway library -- cannot determine the MTLLibrary class to swizzle")
            return
        }
        guard let cls = object_getClass(dummyLibrary) else {
            print("[KernelNameLogger] object_getClass on throwaway library returned nil"); return
        }
        print("[KernelNameLogger] Swizzling MTLLibrary concrete class: \(cls)")

        let sel = Selector(("newFunctionWithName:"))
        guard let m = class_getInstanceMethod(cls, sel) else {
            print("[KernelNameLogger] newFunctionWithName: not found on \(cls)"); return
        }
        gOrigNewFunctionWithName = method_getImplementation(m)
        typealias Fn = @convention(c) (AnyObject, Selector, AnyObject) -> AnyObject?
        let swizzled: Fn = swizzledNewFunctionWithName
        method_setImplementation(m, unsafeBitCast(swizzled, to: IMP.self))
    }
}
