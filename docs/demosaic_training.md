# SwinIR Demosaicing Workflows

This repository now keeps three parallel demosaicing workflows:

1. Single baseline
   - one Bayer pattern per checkpoint
   - 3-channel masked mosaic input
   - entrypoints:
     - `train_demosaic_swinir.py`
     - `main_test_demosaic.py`

2. Existing ESUM
   - one shared checkpoint for `single / quad / nona`
   - 4-channel ESUM input = `raw mosaic (1)` + `CFA mask (3)`
   - entrypoints:
     - `train_demosaic_unified_swinir.py`
     - `main_test_demosaic_unified.py`

3. New all-in-one SwinIR
   - parallel workflow added without replacing the existing ESUM files
   - same all-in-one training idea, but isolated entrypoints and experiment directories
   - entrypoints:
     - `train_demosaic_allinone_swinir.py`
     - `main_test_demosaic_allinone.py`

## Recommended datasets

- Training:
  - DIV2K
  - Flickr2K
  - BSD500
- Validation:
  - McMaster
  - Kodak24

## 1. Single baseline training

```bash
python train_demosaic_swinir.py \
  --train-dirs /path/to/DIV2K /path/to/Flickr2K /path/to/BSD500 \
  --val-dirs /path/to/McMaster /path/to/Kodak24 \
  --save-dir experiments/demosaic_swinir_rggb \
  --pattern RGGB \
  --crop-size 128 \
  --batch-size 8 \
  --epochs 300 \
  --lr 2e-4 \
  --device cuda
```

## 1. Single baseline visualization

```bash
python main_test_demosaic.py \
  --model-path experiments/demosaic_swinir_rggb/model_best.pth \
  --input-path /path/to/McMaster \
  --output-dir results/demosaic_single_rggb \
  --pattern RGGB \
  --max-images 10 \
  --error-scale 8 \
  --device cpu \
  --save-individual
```

## 2. Existing ESUM training

```bash
python train_demosaic_unified_swinir.py \
  --train-dirs /path/to/DIV2K /path/to/Flickr2K /path/to/BSD500 \
  --val-dirs /path/to/McMaster /path/to/Kodak24 \
  --save-dir experiments/demosaic_esum_swinir \
  --base-pattern RGGB \
  --train-pattern-types single quad nona \
  --train-pattern-weights 1 1 1 \
  --val-pattern-types single quad nona \
  --crop-size 144 \
  --batch-size 8 \
  --epochs 300 \
  --lr 2e-4 \
  --loss-type mse \
  --maskout-enabled \
  --maskout-p-min 0.0 \
  --maskout-p-max 0.05 \
  --device cuda
```

## 2. Existing ESUM visualization

```bash
python main_test_demosaic_unified.py \
  --model-path experiments/demosaic_esum_swinir/model_best.pth \
  --input-path /path/to/McMaster \
  --output-dir results/demosaic_esum_quad \
  --pattern-type quad \
  --base-pattern RGGB \
  --max-images 10 \
  --error-scale 8 \
  --device cpu \
  --save-individual
```

## 3. New all-in-one SwinIR training

```bash
python train_demosaic_allinone_swinir.py \
  --train-dirs /path/to/DIV2K /path/to/Flickr2K /path/to/BSD500 \
  --val-dirs /path/to/McMaster /path/to/Kodak24 \
  --save-dir experiments/demosaic_allinone_swinir \
  --base-pattern RGGB \
  --train-pattern-types single quad nona \
  --train-pattern-weights 1 1 1 \
  --val-pattern-types single quad nona \
  --crop-size 144 \
  --batch-size 8 \
  --epochs 300 \
  --lr 2e-4 \
  --loss-type mse \
  --maskout-enabled \
  --maskout-p-min 0.0 \
  --maskout-p-max 0.05 \
  --device cuda
```

## 3. New all-in-one SwinIR visualization

```bash
python main_test_demosaic_allinone.py \
  --model-path experiments/demosaic_allinone_swinir/model_best.pth \
  --input-path /path/to/McMaster \
  --output-dir results/demosaic_allinone_nona \
  --pattern-type nona \
  --base-pattern RGGB \
  --max-images 10 \
  --error-scale 8 \
  --device cpu \
  --save-individual
```

## Notes

- Single baseline uses `in_chans=3` and does not change the all-in-one flows.
- Existing ESUM uses `in_chans=4`, one shared checkpoint, and per-pixel CFA one-hot masks.
- New all-in-one SwinIR also uses `in_chans=4`, but only inside its own new entrypoints and new experiment directories.
- Existing ESUM checkpoints and histories stay untouched.
- Both all-in-one workflows report balanced metrics and separate `single / quad / nona` validation metrics.
