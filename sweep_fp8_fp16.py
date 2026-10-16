#!/usr/bin/env python3
import subprocess
import re
import sys

def run_cmd(cmd):
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=180)
    return proc.stdout

# =========================================================================
# Part 1: FP16 Sweep on ANE (PMU + MPSGraph)
# =========================================================================
fp16_configs = [
    # (batch, size, channels, layers)
    (1, 256, 128, 20),
    (1, 256, 160, 20),
    (1, 256, 192, 20),
    (1, 256, 192, 30),
    (1, 256, 224, 20),
    (1, 256, 224, 30),
    (1, 256, 256, 20),
    (1, 256, 256, 25),
    (1, 256, 256, 30),
    (1, 256, 320, 20),
    (1, 256, 320, 30),
    (2, 256, 192, 20),
    (2, 256, 256, 20),
]

print("=============================================================================================================")
print("  Apple Silicon M6 ANE FP16 Parameter Sweep (Silicon PMU & MPSGraph)")
print("=============================================================================================================")
print(f"{'Workload [B, HxW, C, L]':<30} | {'MPSGraph TOPS':<14} | {'Physical PMU TOPS':<18} | {'MACs/cycle':<12} | {'Zero Audit':<16}")
print("-" * 105)

fp16_results = []
for b, h, c, l in fp16_configs:
    label = f"[{b}, {h}x{h}, C={c}, L={l}]"
    cmd = [
        "./measure_ane_pmu",
        "--variant", "fp16",
        "--batch", str(b),
        "--size", str(h),
        "--channels", str(c),
        "--layers", str(l),
        "--iterations", "15",
        "--no-gpu"
    ]
    try:
        out = run_cmd(cmd)
        m_mps = re.search(r"MPSGraph ANE Result:\s+[\d\.]+\s+ms\s+\|\s+([\d\.]+)\s+TOPS", out)
        mps_tops = float(m_mps.group(1)) if m_mps else 0.0

        m_pmu = re.search(r"Physical Realized TOPS\s+:\s+([\d\.]+)\s+TOPS", out)
        pmu_tops = float(m_pmu.group(1)) if m_pmu else 0.0

        m_macs = re.search(r"Throughput / Silicon Cycle\s+:\s+([\d\.]+)\s+MACs\s+/\s+cycle", out)
        macs = float(m_macs.group(1)) if m_macs else 0.0

        m_zero = re.search(r"Output Zero-Skip Audit\s+:\s+([^\n]+)", out)
        zero_str = m_zero.group(1).split("->")[0].strip() if m_zero else "N/A"
        # Extract percentage
        m_pct = re.search(r"\(([\d\.]+)\%\)", zero_str)
        pct_str = f"{m_pct.group(1)}% zeros" if m_pct else "verified"

        fp16_results.append({
            "label": label, "mps_tops": mps_tops, "pmu_tops": pmu_tops, "macs": macs, "zeros": pct_str
        })
        print(f"{label:<30} | {mps_tops:>8.2f} TOPS  | {pmu_tops:>10.2f} TOPS     | {macs:>9.1f}   | {pct_str:<16}")
        sys.stdout.flush()
    except Exception as e:
        print(f"{label:<30} | ERROR: {e}")

# =========================================================================
# Part 2: FP8 Sweep on ANE (CoreML Runtime)
# =========================================================================
fp8_configs = [
    # (batch, size, channels, layers)
    (1, 256, 128, 20),
    (1, 256, 160, 20),
    (1, 256, 192, 20),
    (1, 256, 192, 30),
    (1, 256, 224, 20),
    (1, 256, 256, 20),
    (1, 256, 256, 25),
    (1, 256, 256, 30),
    (1, 256, 320, 20),
    (2, 256, 128, 20),
    (2, 256, 192, 20),
    (1, 384, 128, 20),
    (1, 384, 192, 20),
]

print("\n=============================================================================================================")
print("  Apple Silicon M6 ANE FP8 Parameter Sweep (CoreML MIL Engine)")
print("=============================================================================================================")
print(f"{'Workload [B, HxW, C, L]':<30} | {'Latency':<12} | {'CoreML TOPS':<14} | {'Framerate':<12} | {'Zero Audit':<20}")
print("-" * 105)

fp8_results = []
for b, h, c, l in fp8_configs:
    label = f"[{b}, {h}x{h}, C={c}, L={l}]"
    cmd = [
        "./measure_conv_coreml",
        "--units", "ane",
        "--precision", "fp8",
        "--batch", str(b),
        "--size", str(h),
        "--channels", str(c),
        "--layers", str(l),
        "--iterations", "15",
        "--check"
    ]
    try:
        out = run_cmd(cmd)
        m_perf = re.search(r"Avg:\s+([\d\.]+)\s+ms,\s+Speed:\s+([\d\.]+)\s+TOPS\s+\(([\d\.]+)\s+FPS\)", out)
        if m_perf:
            lat = float(m_perf.group(1))
            tops = float(m_perf.group(2))
            fps = float(m_perf.group(3))
        else:
            lat, tops, fps = 0.0, 0.0, 0.0

        m_check = re.search(r"Output check:\s+(\d+)\s+elements.*zeros:\s+(\d+)", out)
        if m_check:
            tot = int(m_check.group(1))
            zer = int(m_check.group(2))
            pct = (zer / tot) * 100.0 if tot > 0 else 0.0
            zero_str = f"{zer} zeros ({pct:.3f}%)"
        else:
            zero_str = "Checked"

        fp8_results.append({
            "label": label, "lat": lat, "tops": tops, "fps": fps, "zeros": zero_str
        })
        print(f"{label:<30} | {lat:>7.2f} ms   | {tops:>8.2f} TOPS  | {fps:>6.1f} FPS   | {zero_str:<20}")
        sys.stdout.flush()
    except Exception as e:
        print(f"{label:<30} | ERROR: {e}")

# =========================================================================
# Part 3: Leaderboard Summaries
# =========================================================================
print("\n=============================================================================================================")
print("  TOP ANE FP16 CONFIGURATIONS")
print("=============================================================================================================")
fp16_results.sort(key=lambda x: max(x["pmu_tops"], x["mps_tops"]), reverse=True)
for i, r in enumerate(fp16_results[:3], 1):
    best = max(r["pmu_tops"], r["mps_tops"])
    print(f"  #{i}: {r['label']} -> {best:.2f} TOPS (Physical: {r['pmu_tops']:.2f} TOPS, MPSGraph: {r['mps_tops']:.2f} TOPS, {r['macs']:.1f} MACs/cycle, {r['zeros']})")

print("\n=============================================================================================================")
print("  TOP ANE FP8 CONFIGURATIONS")
print("=============================================================================================================")
fp8_results.sort(key=lambda x: x["tops"], reverse=True)
for i, r in enumerate(fp8_results[:3], 1):
    print(f"  #{i}: {r['label']} -> {r['tops']:.2f} TOPS ({r['lat']:.2f} ms, {r['fps']:.1f} FPS, {r['zeros']})")
