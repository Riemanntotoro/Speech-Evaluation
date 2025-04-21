# ───────────────────────────────────────────────────────────────────────────────
# File: SI_SDR.py
# 출처:
#   • Le Roux, J., Wisler, D., & Vincent, E. (2018).
#     "SDR – Half-Baked or Well Done? The SDR Metric Did Not Survive the
#     Blender," IEEE ICASSP 2018.
# ───────────────────────────────────────────────────────────────────────────────

import numpy as np
from scipy.io import wavfile
import math

def compute_si_sdr(clean_file, processed_file):
    # 1) WAV 읽기
    fs_clean, clean     = wavfile.read(clean_file)
    fs_proc, processed  = wavfile.read(processed_file)
    assert fs_clean == fs_proc, "샘플레이트가 일치하지 않습니다!"

    # 2) 모노 변환
    def to_mono(x):
        return x.mean(axis=1) if x.ndim > 1 else x
    clean     = to_mono(clean)
    processed = to_mono(processed)

    # 3) 길이 정렬
    N = min(len(clean), len(processed))
    clean     = clean[:N]
    processed = processed[:N]

    # 4) zero-mean
    clean_z   = clean - np.mean(clean)
    proc_z    = processed - np.mean(processed)

    # 5) scale-invariant projection
    alpha     = np.dot(proc_z, clean_z) / np.dot(clean_z, clean_z)
    e_true    = alpha * clean_z
    e_err     = proc_z - e_true

    # 6) SI-SDR 계산 (dB)
    score = 10 * math.log10(np.sum(e_true**2) / np.sum(e_err**2))
    return score

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python SI_SDR.py <clean.wav> <processed.wav>")
        sys.exit(1)
    clean_file, processed_file = sys.argv[1], sys.argv[2]
    s = compute_si_sdr(clean_file, processed_file)
    print(f"SI-SDR ({clean_file} vs {processed_file}): {s:.3f} dB")