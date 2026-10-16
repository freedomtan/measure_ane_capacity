#!/usr/bin/env python3
import subprocess
import re
import sys

# Configurations to sweep
# (batch, size, channels, layers, variant)
configs = [
    # 1. Channel sweep around optimal 192-256
    (1, 256, 128, 20, "int8"),
    (1, 256, 160, 20, "int8"),
    (1, 256, 192, 20, "int8"),
    (1, 256, 224, 20, "int8"),
    (1, 256, 256, 20, "int8"),

    # 2. Layer depth scaling on C=192
    (1, 256, 192, 30, "int8"),
    (1, 256, 192, 40, "int8"),
    (1, 256, 192, 50, "int8"),

    # 3. Layer depth scaling on C=256
    (1, 256, 256, 30, "int8"),
    (1, 256, 256, 40, "int8"),

    # 4. Larger spatial size
    (1, 384, 128, 20, "int8"),
    (1, 384, 192, 20, "int8"),
    (1, 512, 128, 10, "int8"),
    (1, 512, 192, 10, "int8"),

    # 5. Batch dimension scaling
    (2, 256, 128, 20, "int8"),
    (2, 256, 192, 20, "int8"),
    (2, 256, 192, 30, "int8"),
    (4, 256, 128, 20, "int8"),
    (4, 256, 192, 20, "int8"),

    # 6. FP16 Scaling
    (1, 256, 192, 20, "fp16"),
    (1, 256, 192, 30, "fp16"),
    (1, 256, 256, 20, "fp16"),
    (2, 256, 192, 20, "fp16"),
]

results = []

print("==========================================================================================")
print("  Apple Silicon M6 Neural Engine Capacity & TOPS Sweep")
print("==========================================================================================")
print(f"{'Config (B, HxW, C, L, Type)':<32} | {'MPSGraph TOPS':<14} | {'Physical PMU TOPS':<18} | {'MACs/cycle':<12}")
print("-" * 88)

for b, h, c, l, var in configs:
    label = f"[{b}, {h}x{h}, C={c}, L={l}, {var}]"
    cmd = [
        "./measure_ane_pmu",
        "--variant", var,
        "--batch", str(b),
        "--size", str(h),
        "--channels", str(c),
        "--layers", str(l),
        "--iterations", "15",
        "--no-gpu"
    ]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=120)
        out = proc.stdout
        
        # Parse MPSGraph TOPS
        mps_tops = 0.0
        m = re.search(r"MPSGraph ANE Result:\s+[\d\.]+\s+ms\s+\|\s+([\d\.]+)\s+TOPS", out)
        if m:
            mps_tops = float(m.group(1))

        # Parse Physical Realized TOPS
        pmu_tops = 0.0
        m = re.search(r"Physical Realized TOPS\s+:\s+([\d\.]+)\s+TOPS", out)
        if m:
            pmu_tops = float(m.group(1))

        # Parse MACs/cycle
        macs = 0.0
        m = re.search(r"Throughput / Silicon Cycle\s+:\s+([\d\.]+)\s+MACs\s+/\s+cycle", out)
        if m:
            macs = float(m.group(1))

        results.append({
            "config": label,
            "mps_tops": mps_tops,
            "pmu_tops": pmu_tops,
            "macs": macs,
            "b": b, "h": h, "c": c, "l": l, "var": var
        })

        print(f"{label:<32} | {mps_tops:>8.2f} TOPS  | {pmu_tops:>10.2f} TOPS     | {macs:>9.1f}")
        sys.stdout.flush()
    except subprocess.TimeoutExpired:
        print(f"{label:<32} | TIMEOUT")
    except Exception as e:
        print(f"{label:<32} | ERROR: {e}")

print("\n==========================================================================================")
print("  TOP 5 LEADERBOARD BY REALIZED PMU TOPS")
print("==========================================================================================")
results.sort(key=lambda x: max(x["pmu_tops"], x["mps_tops"]), reverse=True)
for i, r in enumerate(results[:5], 1):
    best_tops = max(r["pmu_tops"], r["mps_tops"])
    print(f"#{i}: {r['config']} -> {best_tops:.2f} TOPS (Physical: {r['pmu_tops']:.2f} TOPS, MPSGraph: {r['mps_tops']:.2f} TOPS, {r['macs']:.1f} MACs/cycle)")
