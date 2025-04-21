# ───────────────────────────────────────────────────────────────────────────────
# File: PESQ.py
# 출처:
#   • GitHub: https://github.com/ludlows/python-pesq  
#   • ITU‑T 표준: P.862 "Perceptual evaluation of speech quality (PESQ)"
# ───────────────────────────────────────────────────────────────────────────────

from scipy.io import wavfile
from scipy.signal import resample
from pesq import pesq

def compute_pesq(clean_file, processed_file, target_fs=16000):
    # 1) WAV 읽기
    fs_clean, clean       = wavfile.read(clean_file)
    fs_proc, processed    = wavfile.read(processed_file)

    # 2) 리샘플링 (필요 시)
    if fs_clean != target_fs:
        clean = resample(clean, int(len(clean) * target_fs / fs_clean))
    if fs_proc != target_fs:
        processed = resample(processed, int(len(processed) * target_fs / fs_proc))

    # 3) 모노 변환
    if clean.ndim > 1:       clean = clean.mean(axis=1)
    if processed.ndim > 1:   processed = processed.mean(axis=1)

    # 4) 길이 정렬
    N = min(len(clean), len(processed))
    clean     = clean[:N]
    processed = processed[:N]

    # 5) PESQ 계산 ('wb' = wideband)
    score = pesq(target_fs, clean, processed, 'wb')
    return score

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python PESQ.py <clean.wav> <processed.wav>")
        sys.exit(1)
    clean_file, processed_file = sys.argv[1], sys.argv[2]
    p = compute_pesq(clean_file, processed_file)
    print(f"PESQ ({clean_file} vs {processed_file}): {p:.2f}")