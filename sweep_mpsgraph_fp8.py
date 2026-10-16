#!/usr/bin/env python3
import subprocess
import re
import sys

def run_cmd(cmd):
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=180)
    return proc.stdout

# Parameter sweep configurations for MPSGraph FP8 on Apple Neural Engine (M6)
configs = [
    # 1. Channel sweep at L=20, B=1, 256x256
    (1, 256, 128, 20),
    (1, 256, 160, 20),
    (1, 256, 192, 20),
    (1, 256, 224, 20),
    (1, 256, 256, 20),
    (1, 256, 320, 20),
    (1, 256, 384, 20),
    (1, 256, 512, 10),

    # 2. Layer depth scaling on C=192 and C=256
    (1, 256, 192, 10),
    (1, 256, 192, 30),
    (1, 256, 192, 40),
    (1, 256, 256, 25),
    (1, 256, 256, 30),
    (1, 256, 256, 40),

    # 3. Spatial size scaling
    (1, 384, 128, 20),
    (1, 384, 192, 20),
    (1, 512, 128, 10),
    (1, 512, 192, 10),

    # 4. Batch scaling
    (2, 256, 128, 20),
    (2, 256, 192, 20),
    (2, 256, 256, 20),
    (4, 256, 128, 20),
    (4, 256, 192, 20),
]

print("=============================================================================================================")
print("  Apple Silicon M6 ANE FP8 Parameter Sweep (MPSGraph W8A8 QDQ Runner)")
print("=============================================================================================================")
print(f"{'Workload [B, HxW, C, L]':<30} | {'Latency':<12} | {'MPSGraph TOPS':<15} | {'Framerate':<12} | {'Zero Audit':<22}")
print("-" * 105)

results = []
for b, h, c, l in configs:
    label = f"[{b}, {h}x{h}, C={c}, L={l}]"
    cmd = [
        "./test_ane_fp8_app",
        "--batch", str(b),
        "--size", str(h),
        "--channels", str(c),
        "--layers", str(l),
    ]
    try:
        out = run_cmd(cmd)
        m_lat = re.search(r"Latency:\s+([\d\.]+)\s+ms", out)
        m_fps = re.search(r"Framerate:\s+([\d\.]+)\s+FPS", out)
        m_tops = re.search(r"Speed:\s+([\d\.]+)\s+TOPS", out)
        m_zero = re.search(r"Zero Audit:\s+([^\n]+)", out)

        lat = float(m_lat.group(1)) if m_lat else 0.0
        fps = float(m_fps.group(1)) if m_fps else 0.0
        tops = float(m_tops.group(1)) if m_tops else 0.0
        zero_str = m_zero.group(1).strip() if m_zero else "N/A"

        results.append({
            "label": label, "lat": lat, "tops": tops, "fps": fps, "zeros": zero_str
        })
        print(f"{label:<30} | {lat:>8.2f} ms  | {tops:>11.2f} TOPS  | {fps:>8.1f} FPS | {zero_str:<22}")
        sys.stdout.flush()
    except Exception as e:
        print(f"{label:<30} | ERROR: {e}")
        sys.stdout.flush()

print("-" * 105)
best = max(results, key=lambda x: x["tops"]) if results else None
if best:
    print(f"🏆 PEAK FP8 TOPS: {best['tops']:.2f} TOPS at {best['label']} ({best['lat']:.2f} ms, {best['fps']:.1f} FPS, {best['zeros']})")
print("=============================================================================================================")
