# ───────────────────────────────────────────────────────────────────────────────
# File: run_all_dirs.py
# 설명: clean/ 폴더와 processed/ 폴더의 wav 파일을 순서(order)대로 페어링하여
#       STOI, ESTOI, PESQ, SI-SDR를 순차적으로 계산합니다.
# ───────────────────────────────────────────────────────────────────────────────

#!/usr/bin/env python3
import os
import sys
from STOI import compute_stoi
from PESQ import compute_pesq
from SI_SDR import compute_si_sdr

def find_wav_files(dir_path):
    return sorted([
        f for f in os.listdir(dir_path)
        if os.path.isfile(os.path.join(dir_path, f)) and f.lower().endswith('.wav')
    ])

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python run_all_dirs.py <clean_folder> <processed_folder>")
        sys.exit(1)

    clean_dir, proc_dir = sys.argv[1], sys.argv[2]

    # 폴더 유효성 검사
    if not os.path.isdir(clean_dir) or not os.path.isdir(proc_dir):
        print("Error: 지정한 경로가 폴더가 아닙니다.")
        sys.exit(1)

    clean_files = find_wav_files(clean_dir)
    proc_files  = find_wav_files(proc_dir)

    if not clean_files or not proc_files:
        print("Error: wav 파일이 없습니다.")
        sys.exit(1)

    if len(clean_files) != len(proc_files):
        print("Warning: clean 폴더와 processed 폴더의 파일 수가 다릅니다. 가능한 만큼 페어링합니다.")

    # 파일명을 일치시킬 필요 없이, sorted 순서대로 페어링
    for clean_fname, proc_fname in zip(clean_files, proc_files):
        clean_path = os.path.join(clean_dir, clean_fname)
        proc_path  = os.path.join(proc_dir,  proc_fname)

        # 페어링 정보 출력
        print(
            f"\n>>> Processing pair:\n"
            f"    clean: {clean_fname}\n"
            f"    processed: {proc_fname}\n"
        )

        # 1) STOI & ESTOI
        stoi_score, estoi_score = compute_stoi(clean_path, proc_path)
        print(f"  • STOI:  {stoi_score:.3f}, ESTOI: {estoi_score:.3f}")

        # 2) PESQ
        pesq_score = compute_pesq(clean_path, proc_path)
        print(f"  • PESQ:  {pesq_score:.2f}")

        # 3) SI‑SDR
        si_sdr_score = compute_si_sdr(clean_path, proc_path)
        print(f"  • SI‑SDR: {si_sdr_score:.3f} dB")