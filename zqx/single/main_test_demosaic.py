import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image, ImageDraw

from data.demosaic_rgb_dataset import (
    VALID_BASE_PATTERNS,
    collect_image_paths,
    ensure_pattern_size,
    imread_rgb_float,
    rgb_to_masked_bayer_mosaic,
)
from models.network_swinir import SwinIR


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize single-pattern SwinIR demosaicing results.")
    parser.add_argument("--model-path", type=str, required=True, help="Checkpoint path, e.g. model_best.pth")
    parser.add_argument("--input-path", type=str, required=True, help="Image file or image folder")
    parser.add_argument("--output-dir", type=str, default="results/demosaic_preview")
    parser.add_argument("--pattern", type=str, default="", choices=["", *VALID_BASE_PATTERNS])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-images", type=int, default=4)
    parser.add_argument("--error-scale", type=float, default=4.0)
    parser.add_argument("--save-individual", action="store_true")
    return parser.parse_args()


def tensor_to_uint8_image(tensor: torch.Tensor) -> np.ndarray:
    image = tensor.detach().clamp(0, 1).cpu().numpy()
    image = np.transpose(image, (1, 2, 0))
    return np.round(image * 255.0).astype(np.uint8)


def numpy_to_tensor(image: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.transpose(np.ascontiguousarray(image), (2, 0, 1))).float().unsqueeze(0)


def read_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_config(model_path: Path) -> Dict:
    config_path = model_path.parent / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Could not find config.json next to checkpoint: {config_path}")
    return read_json(config_path)


def build_model_from_config(config: Dict, device: torch.device) -> SwinIR:
    model = SwinIR(
        img_size=config.get("crop_size", 128),
        patch_size=1,
        in_chans=3,
        out_chans=3,
        embed_dim=config.get("embed_dim", 96),
        depths=config.get("depths", [6, 6, 6, 6]),
        num_heads=config.get("num_heads", [6, 6, 6, 6]),
        window_size=config.get("window_size", 8),
        mlp_ratio=config.get("mlp_ratio", 2.0),
        upscale=1,
        img_range=1.0,
        upsampler="",
        resi_connection=config.get("resi_connection", "1conv"),
        use_checkpoint=config.get("use_checkpoint", False),
    )
    model.to(device)
    model.eval()
    return model


def load_checkpoint(model: SwinIR, model_path: Path):
    state = torch.load(model_path, map_location="cpu")
    state_dict = state.get("model") or state.get("params_ema") or state.get("params") or state
    model.load_state_dict(state_dict, strict=True)


def collect_inputs(input_path: Path, max_images: int) -> List[Path]:
    def natural_key(path: Path):
        parts = re.split(r"(\d+)", path.stem)
        key = []
        for part in parts:
            key.append(int(part) if part.isdigit() else part.lower())
        key.append(path.suffix.lower())
        return key

    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(collect_image_paths([str(input_path)]), key=natural_key)[:max_images]
    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def add_title(image: np.ndarray, title: str) -> np.ndarray:
    pil = Image.fromarray(image)
    canvas = Image.new("RGB", (pil.width, pil.height + 28), color=(255, 255, 255))
    canvas.paste(pil, (0, 28))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 6), title, fill=(0, 0, 0))
    return np.array(canvas)


def make_error_map(pred: np.ndarray, gt: np.ndarray, error_scale: float) -> np.ndarray:
    diff = np.abs(pred.astype(np.float32) - gt.astype(np.float32)) / 255.0
    diff = np.mean(diff, axis=2)
    diff = np.clip(diff * error_scale, 0.0, 1.0)
    error_rgb = np.stack([diff, np.zeros_like(diff), 1.0 - diff], axis=2)
    return np.round(error_rgb * 255.0).astype(np.uint8)


def save_image(path: Path, image: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)


@torch.no_grad()
def infer_one(
    model: SwinIR,
    image_path: Path,
    pattern: str,
    device: torch.device,
    error_scale: float,
):
    gt = imread_rgb_float(image_path)
    gt = ensure_pattern_size(gt, "single")
    mosaic = rgb_to_masked_bayer_mosaic(gt, pattern)

    pred = model(numpy_to_tensor(mosaic).to(device))[0]

    gt_u8 = np.round(gt * 255.0).astype(np.uint8)
    mosaic_u8 = np.round(np.clip(mosaic, 0, 1) * 255.0).astype(np.uint8)
    pred_u8 = tensor_to_uint8_image(pred)
    error_u8 = make_error_map(pred_u8, gt_u8, error_scale)
    return gt_u8, mosaic_u8, pred_u8, error_u8


def build_panel(
    gt: np.ndarray,
    mosaic: np.ndarray,
    pred: np.ndarray,
    error: np.ndarray,
    error_scale: float,
    pattern: str,
) -> np.ndarray:
    tiles = [
        add_title(gt, "GT"),
        add_title(mosaic, f"Mosaic ({pattern})"),
        add_title(pred, "Demosaiced"),
        add_title(error, f"Error x{error_scale:g}"),
    ]
    return np.concatenate(tiles, axis=1)


def main():
    args = parse_args()
    model_path = Path(args.model_path).expanduser().resolve()
    input_path = Path(args.input_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    device = torch.device(args.device)

    config = resolve_config(model_path)
    pattern = args.pattern or config.get("pattern", "RGGB")

    model = build_model_from_config(config, device)
    load_checkpoint(model, model_path)

    image_paths = collect_inputs(input_path, args.max_images)
    if not image_paths:
        raise ValueError(f"No images found in {input_path}")

    print(f"Using checkpoint: {model_path}")
    print(f"Using Bayer pattern: {pattern}")
    print(f"Saving to: {output_dir}")

    for image_path in image_paths:
        gt, mosaic, pred, error = infer_one(
            model=model,
            image_path=image_path,
            pattern=pattern,
            device=device,
            error_scale=args.error_scale,
        )
        stem = image_path.stem
        panel = build_panel(gt, mosaic, pred, error, args.error_scale, pattern)
        save_image(output_dir / f"{stem}_compare.png", panel)

        if args.save_individual:
            save_image(output_dir / f"{stem}_gt.png", gt)
            save_image(output_dir / f"{stem}_mosaic.png", mosaic)
            save_image(output_dir / f"{stem}_pred.png", pred)
            save_image(output_dir / f"{stem}_error.png", error)

        print(f"Saved visualization for {image_path.name}")


if __name__ == "__main__":
    main()
