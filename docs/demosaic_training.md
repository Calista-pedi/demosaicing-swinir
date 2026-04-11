# Synthetic Bayer Demosaicing with SwinIR

This setup uses ordinary RGB images as ground truth and creates synthetic Bayer mosaics on the fly:

1. Read RGB ground truth.
2. Sample with a Bayer pattern.
3. Build the masked mosaic input.
4. Predict full RGB with SwinIR.
5. Compute loss against the original RGB image.

The goal of this stage is to learn and validate the demosaicing training pipeline first, not to mimic real mobile RAW yet.

## Recommended datasets

- Training:
  - DIV2K
  - Flickr2K
  - BSD500
- Validation:
  - McMaster
  - Kodak24

If Kodak24 is not ready yet, you can validate on McMaster first.

## Expected command

```bash
python train_demosaic_swinir.py \
  --train-dirs /path/to/DIV2K /path/to/Flickr2K /path/to/BSD500 \
  --val-dirs /path/to/McMaster /path/to/Kodak24 \
  --save-dir experiments/demosaic_swinir_rggb \
  --pattern RGGB \
  --crop-size 128 \
  --batch-size 8 \
  --epochs 300 \
  --lr 2e-4
```

## Notes

- Input representation is a 3-channel masked Bayer mosaic.
- Default pattern is `RGGB`, but `BGGR`, `GRBG`, and `GBRG` are also supported.
- A standard Quad Bayer option is also supported as `QRGGB` using a repeating 4x4 tile with 2x2 same-color blocks.
- Validation uses full images and reports PSNR. SSIM is also reported when the optional metric dependency is available.
- For the first run, keep the task simple: synthetic Bayer only, no shot noise, no black level, no ISP simulation.
