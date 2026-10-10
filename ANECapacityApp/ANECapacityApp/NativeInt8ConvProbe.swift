import Foundation
import Metal
import MetalPerformanceShaders

/// Probe for whether the private, undocumented `MPSNDArrayConvolution2D`
/// class can dispatch a genuine int8 x int8 -> int32 convolution (i.e.
/// reach `conv2d_a18/a20_int32_int8_int8` in the metallib, per
/// docs/MPSGraph_MXU_SDPA_Investigation.md Finding 4h/4i). There is no
/// public header for this class -- it is driven here via KVC and raw
/// objc_msgSend against the real framework-loaded class/methods (confirmed
/// to exist via `nm -C` on the extracted framework binary), then the
/// resulting instance is force-downcast to the public
/// `MPSNDArrayMultiaryKernel` base class to call the normal, public, typed
/// `encodeToCommandBuffer:sourceArrays:destinationArray:`.
///
/// On this Mac (M4 Pro) this hits the hard, uncatchable
/// `MPSNDArrayConvolutionPreG13.mm:649` assertion -- the GPU-family
/// convolution-behavior dispatcher picks the legacy `PreG13` class there,
/// not `A18`, even though the metallib ships `conv2d_a18/a20_*` kernels
/// universally. Triggered by `--native-int8-conv` on launch.
enum NativeInt8ConvProbe {
    static func makeInstance(className: String, device: MTLDevice, desc: NSObject, sourceCount: Int? = nil) -> AnyObject? {
        guard let cls = NSClassFromString(className) as? NSObject.Type else { return nil }
        typealias AllocFn = @convention(c) (AnyObject, Selector) -> AnyObject
        let allocSel = Selector(("alloc"))
        let allocImp = unsafeBitCast(class_getMethodImplementation(object_getClass(cls), allocSel), to: AllocFn.self)
        let allocated = allocImp(cls, allocSel)

        if let sourceCount {
            let sel = Selector(("initWithDevice:ndArrayConvolution2DDescriptor:sourceCount:"))
            guard cls.instancesRespond(to: sel) else { return nil }
            typealias InitFn3 = @convention(c) (AnyObject, Selector, AnyObject, AnyObject, UInt) -> AnyObject?
            let imp = unsafeBitCast(class_getMethodImplementation(cls, sel), to: InitFn3.self)
            return imp(allocated, sel, device as AnyObject, desc, UInt(sourceCount))
        } else {
            let sel = Selector(("initWithDevice:ndArrayConvolution2DDescriptor:"))
            guard cls.instancesRespond(to: sel) else { return nil }
            typealias InitFn = @convention(c) (AnyObject, Selector, AnyObject, AnyObject) -> AnyObject?
            let imp = unsafeBitCast(class_getMethodImplementation(cls, sel), to: InitFn.self)
            return imp(allocated, sel, device as AnyObject, desc)
        }
    }

    static func run() {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
            print("[ConvProbe] No Metal device/queue"); return
        }
        print("[ConvProbe] Device: \(device.name)")

        guard let descClass = NSClassFromString("MPSNDArrayConvolution2DDescriptor") as? NSObject.Type else {
            print("[ConvProbe] MPSNDArrayConvolution2DDescriptor class not found"); return
        }
        let desc = descClass.init()

        // Cin alone going 2->16->32->64->128 never changed the "not enough
        // input channels" error at all, with H=W fixed at 2 throughout --
        // suspiciously consistent with dataFormat=0 NOT meaning NCHW, and
        // MPS instead reading the (always-2) H dimension as "channels".
        // Testing that directly: shrink Cin back down, grow H/W instead.
        // Descriptor readback confirmed KVC took effect correctly in every
        // prior attempt (Cin 2..256, H/W 2..32), and NONE of it ever moved
        // this error. The one parameter never touched across 6 rounds:
        // KH=KW=1. Testing that now. NOTE: the "expected" CPU reference
        // below is NOT updated for KH/KW>1 (still only sums over channels,
        // ignoring kernel spatial taps) -- this round only checks whether
        // the assertion itself changes, not numeric correctness.
        // Cin (2..256), Cout, H/W (2..32), and KH/KW (1 and 3) have each
        // independently made zero difference across 7 rounds -- strong sign
        // the issue isn't shape magnitude at all. Testing structure instead:
        // back to the simple known shapes, but with sourceArrays order
        // swapped below (weight first, then input).
        let Cin = 16, Cout = 16, H = 2, W = 2, KH = 1, KW = 1
        desc.setValue(KW, forKey: "kernelWidth")
        desc.setValue(KH, forKey: "kernelHeight")
        desc.setValue(Cin, forKey: "inputFeatureChannels")
        desc.setValue(Cout, forKey: "outputFeatureChannels")
        desc.setValue(1, forKey: "strideInPixelsX")
        desc.setValue(1, forKey: "strideInPixelsY")
        desc.setValue(1, forKey: "groups")
        desc.setValue(1, forKey: "dilationRateInX")
        desc.setValue(1, forKey: "dilationRateInY")
        desc.setValue(0, forKey: "dataFormat")
        desc.setValue(0, forKey: "weightsFormat")

        // Varying Cin (2..128) and separately H/W (2..32) never changed the
        // "not enough input channels" error at all -- check whether the KVC
        // setValue:forKey: calls above actually took effect, rather than
        // silently no-oping on a property-type mismatch.
        for key in ["kernelWidth", "kernelHeight", "inputFeatureChannels", "outputFeatureChannels",
                    "strideInPixelsX", "strideInPixelsY", "groups", "dilationRateInX", "dilationRateInY",
                    "dataFormat", "weightsFormat"] {
            print("[ConvProbe] desc.\(key) readback = \(desc.value(forKey: key) ?? "nil")")
        }

        guard let convInstance = makeInstance(className: "MPSNDArrayConvolution2D", device: device, desc: desc, sourceCount: 2) else {
            print("[ConvProbe] Failed to construct MPSNDArrayConvolution2D"); return
        }
        let kernel = unsafeDowncast(convInstance, to: MPSNDArrayMultiaryKernel.self)
        print("[ConvProbe] MPSNDArrayConvolution2D instance created: \(kernel)")

        func ndArray(dataType: MPSDataType, shape: [Int]) -> MPSNDArray {
            let d = MPSNDArrayDescriptor(dataType: dataType, shape: shape.map { NSNumber(value: $0) })
            return MPSNDArray(device: device, descriptor: d)
        }
        func writeInt8(_ arr: MPSNDArray, _ data: [Int8]) {
            var d = data
            d.withUnsafeMutableBytes { raw in arr.writeBytes(raw.baseAddress!, strideBytes: nil) }
        }
        func readInt32(_ arr: MPSNDArray, count: Int) -> [Int32] {
            var out = [Int32](repeating: 0, count: count)
            out.withUnsafeMutableBytes { raw in arr.readBytes(raw.baseAddress!, strideBytes: nil) }
            return out
        }

        var inputData = [Int8](repeating: 0, count: Cin * H * W)
        var weightData = [Int8](repeating: 0, count: Cout * Cin * KH * KW)
        // Bounded regardless of Cin/Cout (overflowed Int8 at Cin=32 with the
        // previous unbounded c*4+h*2+w+1 formula -- caught via a Swift
        // runtime trap, not an MPS assertion, on the real A20 Pro device).
        for c in 0..<Cin { for h in 0..<H { for w in 0..<W {
            inputData[c*H*W + h*W + w] = Int8((c % 7) - 3 + h + w)
        }}}
        for o in 0..<Cout { for c in 0..<Cin {
            weightData[o*Cin + c] = Int8(c % 2 == 0 ? 2 : -3)
        }}

        var expected = [Int32](repeating: 0, count: Cout * H * W)
        for o in 0..<Cout { for h in 0..<H { for w in 0..<W {
            var acc: Int32 = 0
            for c in 0..<Cin { acc += Int32(inputData[c*H*W + h*W + w]) * Int32(weightData[o*Cin + c]) }
            expected[o*H*W + h*W + w] = acc
        }}}
        print("[ConvProbe] Expected (CPU reference, NCHW/OIHW), first 16: \(expected.prefix(16))")

        let inputArr = ndArray(dataType: .int8, shape: [1, Cin, H, W])
        let weightArr = ndArray(dataType: .int8, shape: [Cout, Cin, KH, KW])
        writeInt8(inputArr, inputData)
        writeInt8(weightArr, weightData)
        let destArr = ndArray(dataType: .int32, shape: [1, Cout, H, W])

        let cb = queue.makeCommandBuffer()!
        kernel.encode(to: cb, sourceArrays: [weightArr, inputArr], destinationArray: destArr)
        cb.commit()
        cb.waitUntilCompleted()

        let out = readInt32(destArr, count: Cout * H * W)
        print("[ConvProbe] GPU result, first 16: \(out.prefix(16))")
        print("[ConvProbe] Match: \(out == expected)")
    }
}
