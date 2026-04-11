#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$ROOT_DIR"

python train_demosaic_swinir.py \
  --train-dirs /home/qyura/SwinIR/data/DIV2K/DIV2K_train_HR/DIV2K_train_HR \
  --val-dirs /home/qyura/SwinIR/testsets/McMaster \
  --save-dir /home/qyura/SwinIR/experiments/demosaic_div2k_qrggb \
  --pattern QRGGB \
  --crop-size 128 \
  --batch-size 4 \
  --epochs 10 \
  --lr 2e-4 \
  --device cuda

python main_test_demosaic.py \
  --model-path /home/qyura/SwinIR/experiments/demosaic_div2k_qrggb/model_best.pth \
  --input-path /home/qyura/SwinIR/testsets/McMaster \
  --output-dir /home/qyura/SwinIR/results/pattern_compare/QRGGB \
  --device cpu \
  --max-images 10 \
  --error-scale 16 \
  --save-individual
