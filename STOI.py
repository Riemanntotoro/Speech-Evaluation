# ───────────────────────────────────────────────────────────────────────────────
# File: STOI.py
# 출처:
#   • GitHub 레포지토리: https://github.com/mpariente/pystoi  
#   • 논문:
#       Taal, C. H., Hendriks, R. C., Heusdens, R., & Jensen, J. (2011).
#       "An Algorithm for Intelligibility Prediction of Time–Frequency Weighted Noisy Speech,"
#       IEEE Transactions on Audio, Speech, and Language Processing, 19(7), 2125–2136.
# ───────────────────────────────────────────────────────────────────────────────

from scipy.io import wavfile
from pystoi import stoi  # 구현 세부사항: see Taal et al. (2011) and pystoi repo

def compute_stoi(clean_file, processed_file):
    # 1) WAV 읽기
    fs_clean, clean     = wavfile.read(clean_file)
    fs_proc, processed  = wavfile.read(processed_file)

    # 2) 사전 조건 확인: 16 kHz 모노
    assert fs_clean == 16000 and fs_proc == 16000, "샘플레이트가 16kHz가 아닙니다!"
    if clean.ndim > 1:       clean = clean.mean(axis=1)
    if processed.ndim > 1:   processed = processed.mean(axis=1)

    # 3) 길이 정렬
    N = min(len(clean), len(processed))
    clean     = clean[:N]
    processed = processed[:N]

    # 4) STOI & ESTOI 계산
    score_stoi  = stoi(clean, processed, fs_clean, extended=False)  # classic STOI
    score_estoi = stoi(clean, processed, fs_clean, extended=True)   # ESTOI

    return score_stoi, score_estoi

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python STOI.py <clean.wav> <processed.wav>")
        sys.exit(1)
    clean_file, processed_file = sys.argv[1], sys.argv[2]
    s, e = compute_stoi(clean_file, processed_file)
    print(f"클래식 STOI      ({clean_file} vs {processed_file}): {s:.3f}")
    print(f"확장 STOI (ESTOI) ({clean_file} vs {processed_file}): {e:.3f}")