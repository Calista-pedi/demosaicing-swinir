import argparse
import gc
import json
import math
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from data.demosaic_rgb_dataset import RGBToBayerDataset, RGBToBayerEvalDataset, VALID_BASE_PATTERNS
from models.network_swinir import SwinIR

try:
    from utils.util_calculate_psnr_ssim import calculate_ssim
except Exception:  # pragma: no cover - optional dependency path
    calculate_ssim = None


def parse_args():
    parser = argparse.ArgumentParser(description="Train a single-pattern SwinIR demosaicing model.")
    parser.add_argument("--train-dirs", nargs="*", default=[], help="Training image folders.")
    parser.add_argument("--val-dirs", nargs="*", default=[], help="Validation image folders.")
    parser.add_argument("--save-dir", type=str, default="experiments/demosaic_swinir_rggb")
    parser.add_argument("--pattern", type=str, default="RGGB", choices=VALID_BASE_PATTERNS)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--repeat", type=int, default=1, help="Virtual repeat factor for training set.")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--loss-type", type=str, default="mse", choices=["mse", "l1"])
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument(
        "--resume-model-only",
        action="store_true",
        help="Load model weights from --resume but reset optimizer/scheduler so new LR schedule takes effect.",
    )
    parser.add_argument("--pretrained", type=str, default="", help="Optional pretrained SwinIR weights.")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--val-every", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--embed-dim", type=int, default=96)
    parser.add_argument("--depths", type=int, nargs="+", default=[6, 6, 6, 6])
    parser.add_argument("--num-heads", type=int, nargs="+", default=[6, 6, 6, 6])
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--mlp-ratio", type=float, default=2.0)
    parser.add_argument("--resi-connection", type=str, default="1conv", choices=["1conv", "3conv"])
    parser.add_argument("--use-checkpoint", action="store_true")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tensor_to_uint8_image(tensor: torch.Tensor) -> np.ndarray:
    image = tensor.detach().clamp(0, 1).cpu().numpy()
    image = np.transpose(image, (1, 2, 0))
    return np.round(image * 255.0).astype(np.uint8)


def calculate_psnr_np(img1: np.ndarray, img2: np.ndarray) -> float:
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return 20.0 * np.log10(255.0 / np.sqrt(mse))


def build_criterion(loss_type: str) -> nn.Module:
    if loss_type == "mse":
        return nn.MSELoss()
    return nn.L1Loss()


def clear_cuda_memory(device: torch.device):
    if device.type == "cuda":
        gc.collect()
        torch.cuda.empty_cache()


def align_crop_size(crop_size: int, window_size: int) -> int:
    required_period = math.lcm(2, window_size)
    if crop_size < required_period:
        return required_period
    return int(math.ceil(crop_size / required_period) * required_period)


def create_model(args) -> nn.Module:
    model = SwinIR(
        img_size=args.crop_size,
        patch_size=1,
        in_chans=3,
        out_chans=3,
        embed_dim=args.embed_dim,
        depths=args.depths,
        num_heads=args.num_heads,
        window_size=args.window_size,
        mlp_ratio=args.mlp_ratio,
        upscale=1,
        img_range=1.0,
        upsampler="",
        resi_connection=args.resi_connection,
        use_checkpoint=args.use_checkpoint,
    )

    if args.pretrained:
        state = torch.load(args.pretrained, map_location="cpu")
        state_dict = state.get("model") or state.get("params_ema") or state.get("params") or state
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"Loaded pretrained weights from {args.pretrained}")
        print(f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
    return model


def build_dataloaders(args):
    train_loader = None
    if args.train_dirs:
        train_dataset = RGBToBayerDataset(
            roots=args.train_dirs,
            crop_size=args.crop_size,
            pattern=args.pattern,
            augment_data=True,
            repeat=args.repeat,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    val_loader = None
    if args.val_dirs:
        val_dataset = RGBToBayerEvalDataset(
            roots=args.val_dirs,
            pattern=args.pattern,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=max(1, min(2, args.num_workers)),
            pin_memory=True,
        )

    return train_loader, val_loader


def load_checkpoint_state(model: nn.Module, checkpoint_path: str) -> Dict:
    state = torch.load(checkpoint_path, map_location="cpu")
    state_dict = state.get("model") or state.get("params_ema") or state.get("params") or state
    model.load_state_dict(state_dict, strict=True)
    return state


def maybe_resume(args, model, optimizer, scheduler):
    start_epoch = 1
    best_psnr = -1.0
    if args.resume:
        state = load_checkpoint_state(model, args.resume)
        best_psnr = float(state.get("best_psnr", -1.0))
        checkpoint_epoch = int(state.get("epoch", 0))
        start_epoch = checkpoint_epoch + 1

        if args.resume_model_only:
            print(
                f"Loaded model weights from {args.resume} at epoch {start_epoch} "
                f"without restoring optimizer/scheduler. New LR schedule will follow current args."
            )
        else:
            if "optimizer" in state:
                optimizer.load_state_dict(state["optimizer"])
            if scheduler is not None and state.get("scheduler") is not None:
                scheduler.load_state_dict(state["scheduler"])
            restored_lr = optimizer.param_groups[0]["lr"]
            print(f"Resumed full training state from {args.resume} at epoch {start_epoch}")
            if not math.isclose(restored_lr, args.lr, rel_tol=1e-9, abs_tol=0.0):
                print(
                    f"Warning: optimizer LR restored from checkpoint ({restored_lr:.6g}) "
                    f"instead of current --lr ({args.lr:.6g})."
                )
            restored_scheduler = state.get("scheduler")
            if restored_scheduler is not None and "T_max" in restored_scheduler and restored_scheduler["T_max"] != args.epochs:
                print(
                    f"Warning: scheduler T_max restored as {restored_scheduler['T_max']} "
                    f"instead of current --epochs {args.epochs}. "
                    "If you changed training recipe, consider using --resume-model-only."
                )
    return start_epoch, best_psnr


@torch.no_grad()
def validate(model: nn.Module, val_loader: DataLoader, device: torch.device):
    model.eval()
    psnr_values: List[float] = []
    ssim_values: List[float] = []

    for batch in val_loader:
        lq = batch["lq"].to(device, non_blocking=True)
        gt = batch["gt"].to(device, non_blocking=True)

        amp_context = torch.autocast(device_type="cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()
        with amp_context:
            pred = model(lq).clamp(0, 1)
        pred_img = tensor_to_uint8_image(pred[0])
        gt_img = tensor_to_uint8_image(gt[0])

        psnr_values.append(calculate_psnr_np(pred_img, gt_img))
        if calculate_ssim is not None:
            ssim_values.append(calculate_ssim(pred_img, gt_img, crop_border=0, input_order="HWC"))

        del lq, gt, pred

    mean_psnr = float(np.mean(psnr_values)) if psnr_values else float("nan")
    mean_ssim = float(np.mean(ssim_values)) if ssim_values else float("nan")
    clear_cuda_memory(device)
    return {
        "psnr": mean_psnr,
        "ssim": mean_ssim,
        "count": len(psnr_values),
    }


def save_checkpoint(save_path: Path, epoch: int, model: nn.Module, optimizer, scheduler, best_psnr: float):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "best_psnr": best_psnr,
        },
        save_path,
    )


def print_validation_metrics(metrics: Dict):
    print(f"Validation: PSNR={metrics['psnr']:.4f} dB, SSIM={metrics['ssim']:.4f}, N={metrics['count']}")


def load_history(save_dir: Path) -> Dict[str, List]:
    history_path = save_dir / "history.json"
    if history_path.exists():
        with open(history_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "epoch": [],
        "train_loss": [],
        "epoch_seconds": [],
        "val_psnr": [],
        "val_ssim": [],
    }


def save_history_json(save_dir: Path, history: Dict[str, List]):
    with open(save_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=True)


def _svg_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def _format_tick_value(value: float, fmt: str) -> str:
    if fmt == "scientific":
        return f"{value:.1e}"
    return f"{value:.3f}"


def _save_line_plot_svg(
    save_path: Path,
    title: str,
    y_label: str,
    epochs: List[int],
    series: List[Tuple[str, List[float], str]],
    y_tick_format: str = "fixed",
):
    finite_series = []
    for name, values, color in series:
        points = [(epoch, float(value)) for epoch, value in zip(epochs, values) if np.isfinite(value)]
        if points:
            finite_series.append((name, points, color))

    if not finite_series:
        return

    width, height = 960, 540
    left, right, top, bottom = 80, 30, 55, 70
    plot_width = width - left - right
    plot_height = height - top - bottom

    x_min = min(epoch for _, points, _ in finite_series for epoch, _ in points)
    x_max = max(epoch for _, points, _ in finite_series for epoch, _ in points)
    y_min = min(value for _, points, _ in finite_series for _, value in points)
    y_max = max(value for _, points, _ in finite_series for _, value in points)
    if x_min == x_max:
        x_min -= 1
        x_max += 1
    if math.isclose(y_min, y_max):
        pad = max(1e-6, abs(y_min) * 0.05 + 1e-3)
        y_min -= pad
        y_max += pad
    else:
        pad = (y_max - y_min) * 0.08
        y_min -= pad
        y_max += pad

    def map_x(epoch: float) -> float:
        return left + (epoch - x_min) / (x_max - x_min) * plot_width

    def map_y(value: float) -> float:
        return top + plot_height - (value - y_min) / (y_max - y_min) * plot_height

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2:.1f}" y="28" text-anchor="middle" font-size="22" font-family="sans-serif">{_svg_escape(title)}</text>',
        f'<text x="{width/2:.1f}" y="{height-18}" text-anchor="middle" font-size="16" font-family="sans-serif">Epoch</text>',
        f'<text x="22" y="{height/2:.1f}" transform="rotate(-90 22 {height/2:.1f})" text-anchor="middle" font-size="16" font-family="sans-serif">{_svg_escape(y_label)}</text>',
        f'<rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" fill="none" stroke="#333" stroke-width="1.5"/>',
    ]

    for i in range(6):
        frac = i / 5
        grid_y = top + frac * plot_height
        value = y_max - frac * (y_max - y_min)
        svg.append(f'<line x1="{left}" y1="{grid_y:.2f}" x2="{left + plot_width}" y2="{grid_y:.2f}" stroke="#e5e7eb" stroke-width="1"/>')
        svg.append(
            f'<text x="{left - 10}" y="{grid_y + 5:.2f}" text-anchor="end" font-size="12" font-family="sans-serif" fill="#374151">{_format_tick_value(value, y_tick_format)}</text>'
        )

    x_tick_count = min(6, max(2, len(epochs)))
    for i in range(x_tick_count):
        frac = i / (x_tick_count - 1)
        epoch = x_min + frac * (x_max - x_min)
        grid_x = map_x(epoch)
        svg.append(f'<line x1="{grid_x:.2f}" y1="{top}" x2="{grid_x:.2f}" y2="{top + plot_height}" stroke="#f1f5f9" stroke-width="1"/>')
        svg.append(f'<text x="{grid_x:.2f}" y="{top + plot_height + 22}" text-anchor="middle" font-size="12" font-family="sans-serif" fill="#374151">{int(round(epoch))}</text>')

    legend_x = left + 12
    legend_y = top + 18
    for idx, (name, points, color) in enumerate(finite_series):
        y = legend_y + idx * 20
        svg.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 18}" y2="{y}" stroke="{color}" stroke-width="3"/>')
        svg.append(f'<text x="{legend_x + 24}" y="{y + 4}" font-size="13" font-family="sans-serif" fill="#111827">{_svg_escape(name)}</text>')

        path_cmd = " ".join(
            ("M" if point_idx == 0 else "L") + f" {map_x(epoch):.2f} {map_y(value):.2f}"
            for point_idx, (epoch, value) in enumerate(points)
        )
        svg.append(f'<path d="{path_cmd}" fill="none" stroke="{color}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>')
        for epoch, value in points:
            svg.append(f'<circle cx="{map_x(epoch):.2f}" cy="{map_y(value):.2f}" r="2.5" fill="{color}"/>')

    svg.append("</svg>")
    save_path.write_text("\n".join(svg), encoding="utf-8")


def save_training_curves(save_dir: Path, history: Dict[str, List]):
    epochs = history["epoch"]
    if not epochs:
        return

    _save_line_plot_svg(
        save_dir / "training_loss_curve.svg",
        title="Training Loss vs Epoch",
        y_label="Loss",
        epochs=epochs,
        series=[("train_loss", history["train_loss"], "#2563eb")],
        y_tick_format="scientific",
    )
    _save_line_plot_svg(
        save_dir / "validation_psnr_curve.svg",
        title="Validation PSNR vs Epoch",
        y_label="PSNR (dB)",
        epochs=epochs,
        series=[("val_psnr", history["val_psnr"], "#111827")],
        y_tick_format="fixed",
    )


def append_history(
    history: Dict[str, List],
    epoch: int,
    train_loss: float,
    epoch_seconds: float,
    metrics: Dict | None,
):
    history["epoch"].append(int(epoch))
    history["train_loss"].append(float(train_loss))
    history["epoch_seconds"].append(float(epoch_seconds))
    if metrics is None:
        history["val_psnr"].append(float("nan"))
        history["val_ssim"].append(float("nan"))
        return
    history["val_psnr"].append(float(metrics["psnr"]))
    history["val_ssim"].append(float(metrics["ssim"]))


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    device = torch.device(args.device)

    if len(args.depths) != len(args.num_heads):
        raise ValueError("--depths and --num-heads must have the same length.")
    if not args.eval_only and not args.train_dirs:
        raise ValueError("--train-dirs is required unless --eval-only is set.")
    if args.eval_only and not args.val_dirs:
        raise ValueError("--eval-only requires --val-dirs.")

    aligned_crop_size = align_crop_size(args.crop_size, args.window_size)
    if aligned_crop_size != args.crop_size:
        print(f"Adjust crop_size from {args.crop_size} to {aligned_crop_size} to match Bayer/window alignment.")
        args.crop_size = aligned_crop_size

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    config_path = save_dir / ("config_eval.json" if args.eval_only else "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=True)

    train_loader, val_loader = build_dataloaders(args)
    history = load_history(save_dir)
    model = create_model(args).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs), eta_min=args.lr * 0.1)
    criterion = build_criterion(args.loss_type)

    start_epoch, best_psnr = maybe_resume(args, model, optimizer, scheduler)

    print(f"Train images: {len(train_loader.dataset) if train_loader is not None else 0}")
    print(f"Val images: {len(val_loader.dataset) if val_loader is not None else 0}")
    print(f"Bayer pattern: {args.pattern}")

    if args.eval_only:
        metrics = validate(model, val_loader, device)
        print_validation_metrics(metrics)
        return

    if train_loader is None:
        raise ValueError("Training requires a train_loader.")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_start = time.time()

        for step, batch in enumerate(train_loader, start=1):
            lq = batch["lq"].to(device, non_blocking=True)
            gt = batch["gt"].to(device, non_blocking=True)

            pred = model(lq)
            loss = criterion(pred, gt)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            if step % args.log_every == 0 or step == len(train_loader):
                print(
                    f"Epoch [{epoch}/{args.epochs}] "
                    f"Step [{step}/{len(train_loader)}] "
                    f"Loss: {loss.item():.6f}"
                )

        scheduler.step()
        avg_loss = epoch_loss / max(1, len(train_loader))
        elapsed = time.time() - epoch_start
        print(f"Epoch {epoch} finished. Avg {args.loss_type.upper()}: {avg_loss:.6f}. Time: {elapsed:.1f}s")

        metrics = None
        if val_loader is not None and epoch % args.val_every == 0:
            del lq, gt, pred, loss, batch
            clear_cuda_memory(device)
            metrics = validate(model, val_loader, device)
            print_validation_metrics(metrics)
            if metrics["psnr"] > best_psnr:
                best_psnr = metrics["psnr"]
                save_checkpoint(save_dir / "model_best.pth", epoch, model, optimizer, scheduler, best_psnr)
                print(f"Saved new best checkpoint with PSNR {best_psnr:.4f} dB")

        append_history(history=history, epoch=epoch, train_loss=avg_loss, epoch_seconds=elapsed, metrics=metrics)
        save_history_json(save_dir, history)
        save_training_curves(save_dir, history)

        if epoch % args.save_every == 0 or epoch == args.epochs:
            save_checkpoint(save_dir / f"model_epoch_{epoch:04d}.pth", epoch, model, optimizer, scheduler, best_psnr)


if __name__ == "__main__":
    main()
