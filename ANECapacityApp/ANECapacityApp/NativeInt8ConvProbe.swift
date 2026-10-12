import Foundation
import Metal
import MetalPerformanceShaders

/// Finding 4o/4p's FP8-E4M3 in-range correctness test (restored from commit
/// 854bab6, with KernelNameLogger wired in to also reconfirm Finding 4p's
/// generic-plugin-fallback result on whatever device this runs on).
enum NativeInt8ConvProbe {
    static func run() {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
            print("[ConvProbe] No Metal device/queue"); return
        }
        print("[ConvProbe] Device: \(device.name)")
        KernelNameLogger.install(on: device)

        let Cin = 16, Cout = 16, H = 17, W = 17, KH = 16, KW = 16
        let outH = H - KH + 1, outW = W - KW + 1
        print("[ConvProbe] Cin=\(Cin) Cout=\(Cout) H=\(H) W=\(W) KH=\(KH) KW=\(KW) outH=\(outH) outW=\(outW)")

        // Raw 1-byte E4M3 encodings for a small cycling set of exactly-
        // representable, all-positive values, scaled so the full
        // Cin*KH*KW=4096-term reduction lands around ~50, comfortably inside
        // E4M3's ~448 max (Finding 4o's follow-up).
        let fp8Patterns: [(byte: UInt8, value: Float)] = [
            (0x20, 0.125), (0x18, 0.0625), (0x24, 0.1875), (0x10, 0.03125),
            (0x22, 0.15625), (0x1C, 0.09375), (0x26, 0.21875), (0x08, 0.015625),
        ]
        func fillFP8(count: Int) -> (bytes: [UInt8], values: [Float]) {
            var bytes = [UInt8](repeating: 0, count: count)
            var values = [Float](repeating: 0, count: count)
            for i in 0..<count {
                let p = fp8Patterns[i % fp8Patterns.count]
                bytes[i] = p.byte
                values[i] = p.value
            }
            return (bytes, values)
        }
        let (inputBytes, inputData) = fillFP8(count: Cin * H * W)
        let (weightBytes, weightData) = fillFP8(count: Cout * Cin * KH * KW)

        guard let descClass = NSClassFromString("MPSNDArrayConvolution2DDescriptor") as? NSObject.Type else {
            print("[ConvProbe] descriptor class not found"); return
        }
        let desc = descClass.init()
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
        desc.setValue(1, forKey: "channelMultiplier")

        guard let convClass = NSClassFromString("MPSNDArrayConvolution2D") as? NSObject.Type else {
            print("[ConvProbe] class not found"); return
        }
        let sel = Selector(("initWithDevice:ndArrayConvolution2DDescriptor:"))
        typealias AllocFn = @convention(c) (AnyObject, Selector) -> AnyObject
        let allocSel = Selector(("alloc"))
        let allocImp = unsafeBitCast(class_getMethodImplementation(object_getClass(convClass), allocSel), to: AllocFn.self)
        let allocated = allocImp(convClass, allocSel)
        typealias InitFn = @convention(c) (AnyObject, Selector, AnyObject, AnyObject) -> AnyObject?
        let imp = unsafeBitCast(class_getMethodImplementation(convClass, sel), to: InitFn.self)
        guard let convInstance = imp(allocated, sel, device as AnyObject, desc) else {
            print("[ConvProbe] Failed to construct MPSNDArrayConvolution2D"); return
        }
        let kernel = unsafeDowncast(convInstance, to: MPSNDArrayMultiaryKernel.self)
        print("[ConvProbe] MPSNDArrayConvolution2D (plain, FP8-E4M3) instance created: \(kernel)")

        func ndArray(dataType: MPSDataType, shape: [Int]) -> MPSNDArray {
            let d = MPSNDArrayDescriptor(dataType: dataType, shape: shape.map { NSNumber(value: $0) })
            return MPSNDArray(device: device, descriptor: d)
        }
        func writeBytes(_ arr: MPSNDArray, _ bytes: [UInt8]) {
            var b = bytes
            b.withUnsafeMutableBytes { raw in arr.writeBytes(raw.baseAddress!, strideBytes: nil) }
        }
        func readBytes(_ arr: MPSNDArray, count: Int) -> [UInt8] {
            var out = [UInt8](repeating: 0, count: count)
            out.withUnsafeMutableBytes { raw in arr.readBytes(raw.baseAddress!, strideBytes: nil) }
            return out
        }
        // E4M3 decode: 1 sign, 4 exponent (bias 7), 3 mantissa.
        func decodeE4M3(_ byte: UInt8) -> Float {
            let sign: Float = (byte & 0x80) != 0 ? -1 : 1
            let exp = Int((byte >> 3) & 0x0F)
            let mant = Int(byte & 0x07)
            if exp == 0 {
                return sign * Float(mant) / 8.0 * Float(exp2(Double(-6)))
            }
            if exp == 0b1111 && mant == 0b111 { return Float.nan }
            let value = (1.0 + Double(mant) / 8.0) * pow(2.0, Double(exp - 7))
            return sign * Float(value)
        }

        guard #available(iOS 27.0, *) else {
            print("[ConvProbe] FP8 requires iOS 27+"); return
        }
        let inputArr = ndArray(dataType: .float8e4m3, shape: [1, Cin, H, W])
        let weightArr = ndArray(dataType: .float8e4m3, shape: [Cout, Cin, KH, KW])
        writeBytes(inputArr, inputBytes)
        writeBytes(weightArr, weightBytes)
        let destArr = ndArray(dataType: .float8e4m3, shape: [1, Cout, outH, outW])

        let cb = queue.makeCommandBuffer()!
        kernel.encode(to: cb, sourceArrays: [inputArr, weightArr], destinationArray: destArr)
        cb.commit()
        cb.waitUntilCompleted()

        let outBytes = readBytes(destArr, count: Cout * outH * outW)
        let out = outBytes.map(decodeE4M3)
        let nonZero = outBytes.filter { $0 != 0 }.count
        print("[ConvProbe] Dispatch succeeded, no crash. GPU result (decoded): nonZero=\(nonZero)/\(outBytes.count) first16=\(out.prefix(16))")

        func inputOffset(_ layout: String, _ c: Int, _ h: Int, _ w: Int) -> Int {
            layout == "nchw" ? c*H*W + h*W + w : h*W*Cin + w*Cin + c
        }
        func weightOffset(_ layout: String, _ o: Int, _ c: Int, _ kh: Int, _ kw: Int) -> Int {
            switch layout {
            case "oihw": return o*Cin*KH*KW + c*KH*KW + kh*KW + kw
            case "iohw": return c*Cout*KH*KW + o*KH*KW + kh*KW + kw
            case "ohwi": return o*KH*KW*Cin + kh*KW*Cin + kw*Cin + c
            default /* hwio */: return kh*KW*Cin*Cout + kw*Cin*Cout + c*Cout + o
            }
        }
        func outputOffset(_ layout: String, _ o: Int, _ oh: Int, _ ow: Int) -> Int {
            layout == "nchw" ? o*outH*outW + oh*outW + ow : oh*outW*Cout + ow*Cout + o
        }
        func computeExpected(inLayout: String, wLayout: String, outLayout: String, flip: Bool) -> [Float] {
            var result = [Float](repeating: 0, count: Cout * outH * outW)
            for o in 0..<Cout { for oh in 0..<outH { for ow in 0..<outW {
                var acc: Float = 0
                for c in 0..<Cin { for kh in 0..<KH { for kw in 0..<KW {
                    let (ikh, ikw) = flip ? (KH-1-kh, KW-1-kw) : (kh, kw)
                    guard oh+ikh < H, ow+ikw < W else { continue }
                    let inVal = inputData[inputOffset(inLayout, c, oh+ikh, ow+ikw)]
                    let wVal = weightData[weightOffset(wLayout, o, c, kh, kw)]
                    acc += inVal * wVal
                }}}
                result[outputOffset(outLayout, o, oh, ow)] = acc
            }}}
            return result
        }
        func maxAbsDiff(_ candidate: [Float]) -> Float {
            zip(out, candidate).map { abs($0 - $1) }.max() ?? -1
        }

        var ranked: [(label: String, diff: Float)] = []
        for inLayout in ["nchw", "nhwc"] {
            for wLayout in ["oihw", "iohw", "ohwi", "hwio"] {
                for outLayout in ["nchw", "nhwc"] {
                    for flip in [false, true] {
                        let label = "in=\(inLayout) w=\(wLayout) out=\(outLayout) flip=\(flip)"
                        let diff = maxAbsDiff(computeExpected(inLayout: inLayout, wLayout: wLayout, outLayout: outLayout, flip: flip))
                        ranked.append((label, diff))
                    }
                }
            }
        }
        ranked.sort { $0.diff < $1.diff }
        print("[ConvProbe] Layout sweep (in-range FP8 data), best 5 by max-abs-diff (near-0 = found it):")
        for r in ranked.prefix(5) {
            print("[ConvProbe]   diff=\(r.diff)  \(r.label)")
        }
    }
}
