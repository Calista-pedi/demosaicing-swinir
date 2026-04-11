import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from data.demosaic_rgb_dataset import RGBToBayerDataset, RGBToBayerEvalDataset
from models.network_swinir import SwinIR

try:
    from utils.util_calculate_psnr_ssim import calculate_ssim
except Exception:  # pragma: no cover - optional dependency path
    calculate_ssim = None


def parse_args():
    parser = argparse.ArgumentParser(description="Train SwinIR for synthetic Bayer demosaicing from RGB datasets.")
    parser.add_argument("--train-dirs", nargs="+", required=True, help="Training image folders.")
    parser.add_argument("--val-dirs", nargs="*", default=[], help="Validation image folders.")
    parser.add_argument("--save-dir", type=str, default="experiments/demosaic_swinir")
    parser.add_argument("--pattern", type=str, default="RGGB", choices=["RGGB", "BGGR", "GRBG", "GBRG", "QRGGB"])
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--repeat", type=int, default=1, help="Virtual repeat factor for training set.")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--pretrained", type=str, default="", help="Optional pretrained SwinIR weights.")
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


def create_model(args) -> nn.Module:
    model = SwinIR(
        img_size=args.crop_size,
        patch_size=1,
        in_chans=3,
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
        state_dict = state.get("params_ema") or state.get("params") or state
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"Loaded pretrained weights from {args.pretrained}")
        print(f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
    return model


def build_dataloaders(args):
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
        val_dataset = RGBToBayerEvalDataset(roots=args.val_dirs, pattern=args.pattern)
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=max(1, min(2, args.num_workers)),
            pin_memory=True,
        )

    return train_loader, val_loader


def tensor_to_uint8_image(tensor: torch.Tensor) -> np.ndarray:
    image = tensor.detach().clamp(0, 1).cpu().numpy()
    image = np.transpose(image, (1, 2, 0))
    image = np.round(image * 255.0).astype(np.uint8)
    return image


def calculate_psnr_np(img1: np.ndarray, img2: np.ndarray) -> float:
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return 20.0 * np.log10(255.0 / np.sqrt(mse))


@torch.no_grad()
def validate(model: nn.Module, val_loader: DataLoader, device: torch.device):
    model.eval()
    psnr_list = []
    ssim_list = []

    for batch in val_loader:
        lq = batch["lq"].to(device, non_blocking=True)
        gt = batch["gt"].to(device, non_blocking=True)
        pred = model(lq).clamp(0, 1)

        pred_img = tensor_to_uint8_image(pred[0])
        gt_img = tensor_to_uint8_image(gt[0])
        psnr_list.append(calculate_psnr_np(pred_img, gt_img))
        if calculate_ssim is not None:
            ssim_list.append(calculate_ssim(pred_img, gt_img, crop_border=0, input_order="HWC"))

    mean_ssim = float(np.mean(ssim_list)) if ssim_list else float("nan")
    return float(np.mean(psnr_list)), mean_ssim


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


def maybe_resume(args, model, optimizer, scheduler):
    start_epoch = 1
    best_psnr = -1.0
    if args.resume:
        state = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        if scheduler is not None and state.get("scheduler") is not None:
            scheduler.load_state_dict(state["scheduler"])
        start_epoch = int(state["epoch"]) + 1
        best_psnr = float(state.get("best_psnr", -1.0))
        print(f"Resumed from {args.resume} at epoch {start_epoch}")
    return start_epoch, best_psnr


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    device = torch.device(args.device)

    if len(args.depths) != len(args.num_heads):
        raise ValueError("--depths and --num-heads must have the same length.")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=True)

    train_loader, val_loader = build_dataloaders(args)
    model = create_model(args).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.1)
    criterion = nn.L1Loss()

    start_epoch, best_psnr = maybe_resume(args, model, optimizer, scheduler)

    print(f"Train images: {len(train_loader.dataset)}")
    print(f"Val images: {len(val_loader.dataset) if val_loader is not None else 0}")
    print(f"Bayer pattern: {args.pattern}")

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
        print(f"Epoch {epoch} finished. Avg L1: {avg_loss:.6f}. Time: {elapsed:.1f}s")

        if val_loader is not None and epoch % args.val_every == 0:
            psnr, ssim = validate(model, val_loader, device)
            print(f"Validation Epoch {epoch}: PSNR={psnr:.4f} dB, SSIM={ssim:.4f}")
            if psnr > best_psnr:
                best_psnr = psnr
                save_checkpoint(save_dir / "model_best.pth", epoch, model, optimizer, scheduler, best_psnr)
                print(f"Saved new best checkpoint with PSNR {best_psnr:.4f} dB")

        if epoch % args.save_every == 0 or epoch == args.epochs:
            save_checkpoint(save_dir / f"model_epoch_{epoch:04d}.pth", epoch, model, optimizer, scheduler, best_psnr)


if __name__ == "__main__":
    main()
