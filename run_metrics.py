#!/usr/bin/env python3
"""
run_metrics_dnsmos_stoi_estoi.py

Evaluate STOI, ESTOI, and DNSMOS metrics for a set of audio files.
"""
import os
import argparse
import csv

import librosa
import torch

from stoi import stoi, estoi
from torchmetrics.audio import DeepNoiseSuppressionMeanOpinionScore


def main(clean_dir, denoised_dir, sr, output_path):
    # DNSMOS meter 초기화
    dnsmos_meter = DeepNoiseSuppressionMeanOpinionScore(fs=sr, personalized=False)

    results = []

    for fname in sorted(os.listdir(clean_dir)):
        if not fname.lower().endswith(".wav"):
            continue

        clean_path = os.path.join(clean_dir, fname)
        proc_path = os.path.join(denoised_dir, fname)
        if not os.path.exists(proc_path):
            print(f"WARNING: Denoised file not found for {fname}")
            continue

        # 오디오 로드
        clean_audio, _ = librosa.load(clean_path, sr=sr)
        proc_audio, _ = librosa.load(proc_path, sr=sr)

        # 1) STOI/ESTOI
        stoi_score = stoi(clean_path, proc_path)
        estoi_score = estoi(clean_path, proc_path)

        # 2) DNSMOS
        proc_t = torch.tensor(proc_audio)
        dnsmos_scores = dnsmos_meter(proc_t)
        p808_mos, mos_sig, mos_bak, mos_ovr = dnsmos_scores.tolist()

        results.append({
            "file": fname,
            "stoi": stoi_score,
            "estoi": estoi_score,
            "p808_mos": p808_mos,
            "mos_sig": mos_sig,
            "mos_bak": mos_bak,
            "mos_ovr": mos_ovr,
        })

    # 결과 저장
    if results:
        fieldnames = list(results[0].keys())
        with open(output_path, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"Metrics saved to {output_path}")
    else:
        print("No metrics computed. Check input directories.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate STOI, ESTOI, and DNSMOS for audio files"
    )
    parser.add_argument(
        "--clean_dir", type=str, default="clean", help="Directory of reference clean audio"
    )
    parser.add_argument(
        "--denoised_dir", type=str, default="denoised", help="Directory of denoised audio"
    )
    parser.add_argument(
        "--sr", type=int, default=16000, help="Sampling rate (Hz)"
    )
    parser.add_argument(
        "--output", type=str, default="metrics.csv", help="Output CSV file path"
    )
    args = parser.parse_args()
    main(args.clean_dir, args.denoised_dir, args.sr, args.output)
