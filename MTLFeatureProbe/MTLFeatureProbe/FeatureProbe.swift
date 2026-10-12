import Foundation
import Metal
import ObjectiveC

/// Enumerates and calls every private `supports*`/`familySupports*` selector on
/// `MTLDevice` and its SPI `featureQueries` object, plus the `_featureInfo[]`
/// diagnostic table backing `_MTLDeviceFeatureQueries`. No private headers are
/// linked against - everything here goes through the Objective-C runtime, the
/// same way the static-disassembly investigation this app grew out of worked,
/// just live and on-device instead of offline against the dyld shared cache.

struct SelectorResult: Identifiable, Hashable {
    let id = UUID()
    let name: String
    let value: Bool
}

struct TableEntry: Identifiable, Hashable {
    let id = UUID()
    let index: Int
    let name: String
    let requirement: Int64
    let supported: Bool
}

enum MTLFeatureProbe {

    private typealias BoolIMP = @convention(c) (AnyObject, Selector) -> Bool
    private typealias IdIMP = @convention(c) (AnyObject, Selector) -> Unmanaged<AnyObject>?

    /// Walks `cls` and every superclass, collecting no-argument, BOOL-returning
    /// methods whose selector starts with `prefix`. Matching the exact type
    /// encoding (`B` = _Bool, `c` = signed char/old-style BOOL) avoids
    /// accidentally invoking something that isn't a simple capability getter.
    private static func boolSelectors(startingAt cls: AnyClass, matchingPrefix prefix: String) -> [Selector] {
        var seen = Set<String>()
        var out: [Selector] = []
        var current: AnyClass? = cls
        while let c = current {
            var count: UInt32 = 0
            if let list = class_copyMethodList(c, &count) {
                for i in 0..<Int(count) {
                    let m = list[i]
                    let sel = method_getName(m)
                    let name = NSStringFromSelector(sel)
                    guard name.hasPrefix(prefix) else { continue }
                    guard method_getNumberOfArguments(m) == 2 else { continue } // self, _cmd only
                    guard let enc = method_getTypeEncoding(m) else { continue }
                    guard let retCode = String(cString: enc).first, retCode == "B" || retCode == "c" else { continue }
                    if seen.insert(name).inserted {
                        out.append(sel)
                    }
                }
                free(list)
            }
            current = class_getSuperclass(c)
        }
        return out.sorted { NSStringFromSelector($0) < NSStringFromSelector($1) }
    }

    /// Calls a resolved, zero-arg BOOL selector directly through its IMP,
    /// bypassing `perform(_:)` (which misrepresents BOOL returns as `id`).
    private static func callBool(_ obj: AnyObject, _ sel: Selector) -> Bool? {
        guard let cls = object_getClass(obj), let method = class_getInstanceMethod(cls, sel) else { return nil }
        let fn = unsafeBitCast(method_getImplementation(method), to: BoolIMP.self)
        return fn(obj, sel)
    }

    /// Same idea for a zero-arg selector that returns an object (e.g. `featureQueries`).
    /// Returns an `Unmanaged` reference deliberately: the caller doesn't own it,
    /// and the owning object (`device`) is kept alive by the caller for as long
    /// as the result is used.
    private static func callId(_ obj: AnyObject, _ sel: Selector) -> AnyObject? {
        guard let cls = object_getClass(obj), let method = class_getInstanceMethod(cls, sel) else { return nil }
        let fn = unsafeBitCast(method_getImplementation(method), to: IdIMP.self)
        return fn(obj, sel)?.takeUnretainedValue()
    }

    static func probeDeviceSupports(_ device: MTLDevice) -> [SelectorResult] {
        let obj = device as AnyObject
        guard let cls = object_getClass(obj) else { return [] }
        return boolSelectors(startingAt: cls, matchingPrefix: "supports").map {
            SelectorResult(name: NSStringFromSelector($0), value: callBool(obj, $0) ?? false)
        }
    }

    static func featureQueries(_ device: MTLDevice) -> AnyObject? {
        let obj = device as AnyObject
        let sel = NSSelectorFromString("featureQueries")
        guard obj.responds(to: sel) else { return nil }
        return callId(obj, sel)
    }

    static func probeFamilySupports(_ device: MTLDevice) -> [SelectorResult] {
        guard let fq = featureQueries(device), let cls = object_getClass(fq) else { return [] }
        return boolSelectors(startingAt: cls, matchingPrefix: "familySupports").map {
            SelectorResult(name: NSStringFromSelector($0), value: callBool(fq, $0) ?? false)
        }
    }

    /// Reads the ivar backing `_MTLDeviceFeatureQueries._featureInfo` directly.
    /// It's a plain C-array ivar (no `@property`), so there's no selector/KVC
    /// path to it - this mirrors the live introspection done during the
    /// reverse-engineering investigation (`class_getInstanceVariable` +
    /// `ivar_getOffset`), but reads the element count out of the ivar's own
    /// runtime type encoding instead of hardcoding a count that could drift
    /// between OS versions.
    static func probeFeatureInfoTable(_ device: MTLDevice) -> [TableEntry] {
        guard let fq = featureQueries(device), let cls = object_getClass(fq) else { return [] }
        guard let ivar = class_getInstanceVariable(cls, "_featureInfo") else { return [] }
        let offset = ivar_getOffset(ivar)
        let count = arrayCount(of: ivar)
        guard count > 0, count < 10_000 else { return [] }

        let base = Unmanaged.passUnretained(fq).toOpaque().advanced(by: offset)
        let stride = 24 // { __unsafe_unretained NSString *name; long long requirement; BOOL supported; }

        var out: [TableEntry] = []
        out.reserveCapacity(count)
        for i in 0..<count {
            let entry = base.advanced(by: i * stride)
            let namePtr = entry.load(as: UInt.self)
            var name = ""
            if namePtr != 0, let p = UnsafeRawPointer(bitPattern: namePtr) {
                name = Unmanaged<NSString>.fromOpaque(p).takeUnretainedValue() as String
            }
            let requirement = entry.advanced(by: 8).load(as: Int64.self)
            let supportedByte = entry.advanced(by: 16).load(as: UInt8.self)
            if name.isEmpty && requirement == 0 && supportedByte == 0 { continue }
            out.append(TableEntry(index: i, name: name, requirement: requirement, supported: supportedByte != 0))
        }
        return out
    }

    /// Parses a leading `[N...` out of an ivar's type encoding to recover a
    /// fixed C-array's element count without hardcoding it.
    private static func arrayCount(of ivar: Ivar) -> Int {
        guard let enc = ivar_getTypeEncoding(ivar) else { return 0 }
        let s = String(cString: enc)
        guard s.hasPrefix("[") else { return 0 }
        let digits = s.dropFirst().prefix { $0.isNumber }
        return Int(digits) ?? 0
    }

    struct DiffEntry: Identifiable, Hashable {
        let id = UUID()
        let tableIndex: Int
        let tableName: String
        let tableSupported: Bool
        let matchedSelector: String
        let selectorSupported: Bool
        let matchScore: Int
    }

    private static let stopWords: Set<String> = [
        "support", "supports", "the", "and", "for", "of", "on", "in", "with",
        "from", "a", "to", "is", "this", "as", "or", "format", "formats",
    ]

    /// Splits a selector's CamelCase remainder (after stripping its
    /// `supports`/`familySupports` prefix) and a display-name string into
    /// lowercase word sets, dropping common filler words, so the two
    /// differently-styled naming schemes (`supportsBCTextureCompression` vs.
    /// "BC-format texture compression") can be compared.
    private static func wordSet(fromCamelCase s: String) -> Set<String> {
        var words: [String] = []
        var current = ""
        for ch in s {
            if ch.isUppercase || ch.isNumber {
                if !current.isEmpty, ch.isUppercase, let last = current.last, last.isLowercase {
                    words.append(current)
                    current = String(ch)
                } else {
                    current.append(ch)
                }
            } else {
                current.append(ch)
            }
        }
        if !current.isEmpty { words.append(current) }
        return Set(words.map { $0.lowercased() }).subtracting(stopWords).filter { $0.count >= 3 }
    }

    private static func wordSet(fromDisplayName s: String) -> Set<String> {
        let cleaned = s.lowercased().map { $0.isLetter || $0.isNumber ? $0 : " " }
        let words = String(cleaned).split(separator: " ").map(String.init)
        return Set(words).subtracting(stopWords).filter { $0.count >= 3 }
    }

    /// Best-effort name matching between the human-readable `_featureInfo[]`
    /// display strings and the live `familySupports*` selectors, then a
    /// direct boolean comparison -- the live selector is the real,
    /// post-construction answer; the table entry is the value baked in once
    /// during `-[_MTLDeviceFeatureQueries initWithDevice:]` itself (which
    /// this document already showed is stale/wrong for any capability an AGX
    /// driver class doesn't directly override; see the FP8/MXUNarrowTileSizes
    /// finding). This surfaces every other capability with the same symptom.
    static func fullTableDiff(_ device: MTLDevice) -> (diffs: [DiffEntry], unmatched: [String]) {
        guard let fq = featureQueries(device), let cls = object_getClass(fq) else { return ([], []) }
        let familySelectors = boolSelectors(startingAt: cls, matchingPrefix: "familySupports")
        var selectorWordSets: [(sel: Selector, name: String, words: Set<String>)] = []
        for sel in familySelectors {
            let name = NSStringFromSelector(sel)
            let remainder = String(name.dropFirst("familySupports".count))
            selectorWordSets.append((sel, name, wordSet(fromCamelCase: remainder)))
        }

        let table = probeFeatureInfoTable(device)
        var diffs: [DiffEntry] = []
        var unmatched: [String] = []

        for entry in table {
            let tableWords = wordSet(fromDisplayName: entry.name)
            guard !tableWords.isEmpty else { unmatched.append(entry.name); continue }

            var best: (sel: Selector, name: String, score: Int)?
            var secondBestScore = 0
            for candidate in selectorWordSets {
                let score = tableWords.intersection(candidate.words).count
                guard score > 0 else { continue }
                if best == nil || score > best!.score {
                    secondBestScore = best?.score ?? 0
                    best = (candidate.sel, candidate.name, score)
                } else if score > secondBestScore {
                    secondBestScore = score
                }
            }

            guard let match = best, match.score >= 1, match.score > secondBestScore else {
                unmatched.append(entry.name)
                continue
            }

            guard let value = callBool(fq, match.sel) else { continue }
            if value != entry.supported {
                diffs.append(DiffEntry(
                    tableIndex: entry.index, tableName: entry.name, tableSupported: entry.supported,
                    matchedSelector: match.name, selectorSupported: value, matchScore: match.score
                ))
            }
        }
        return (diffs, unmatched)
    }
}
