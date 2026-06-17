import argparse
import csv
import json
import os
import time
from itertools import cycle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from component.dataset import CsvVolumeDataset
from utils.splits import split_train_val
from utils.volumes import list_nifti_files, load_nifti, volume_to_tensor


class UnlabeledVolumeDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.paths = list_nifti_files(folder)
        self.transform = transform
        if not self.paths:
            raise ValueError(f"No NIfTI files found in unlabeled folder: {folder}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        volume = volume_to_tensor(load_nifti(self.paths[idx]))
        if self.transform:
            volume = self.transform(volume)
        return volume


class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels),
            )

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)

        return self.relu(out + identity)


class ResNet3D(nn.Module):
    def __init__(self, block=BasicBlock3D, layers=(2, 2, 2, 2), output_size=1, base_channels=16, dropout=0.2):
        super().__init__()
        self.in_channels = base_channels
        self.stem = nn.Sequential(
            nn.Conv3d(1, base_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(block, base_channels, layers[0], stride=1)
        self.layer2 = self._make_layer(block, base_channels * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, base_channels * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, base_channels * 8, layers[3], stride=2)
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(base_channels * 8 * block.expansion, output_size)

        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm3d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _make_layer(self, block, out_channels, blocks, stride):
        layers = [block(self.in_channels, out_channels, stride)]
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)


def build_transform(augment):
    if not augment:
        return None
    import torchio as tio

    return tio.Compose([
        tio.RandomFlip(axes=(0, 1, 2)),
        tio.RandomAffine(scales=(0.95, 1.05), degrees=10, isotropic=True),
    ])


def make_loader(dataset, batch_size, shuffle, num_workers, device):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )


def rounded_predictions(outputs, min_label=0, max_label=3):
    return outputs.view(-1).detach().round().clamp(min_label, max_label).long()


def pseudo_targets_from_outputs(outputs, min_label=0, max_label=3):
    return outputs.view(-1).detach().round().clamp(min_label, max_label).float()


def run_epoch(
    model,
    loader,
    criterion,
    device,
    optimizer=None,
    amp=False,
    min_label=0,
    max_label=3,
    unlabeled_loader=None,
    pseudo_weight=0.3,
    pseudo_threshold=0.25,
):
    is_train = optimizer is not None
    model.train(is_train)
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")
    total_loss = 0.0
    correct = 0
    total = 0
    mae_sum = 0.0
    pseudo_used = 0
    unlabeled_iter = cycle(unlabeled_loader) if is_train and unlabeled_loader is not None else None

    for x, y in tqdm(loader, desc="Train" if is_train else "Eval", unit="batch"):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).float()

        with torch.set_grad_enabled(is_train):
            with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
                outputs = model(x).view(-1)
                loss = criterion(outputs, y.view(-1))

                if unlabeled_iter is not None:
                    x_unlabeled = next(unlabeled_iter).to(device, non_blocking=True)
                    with torch.no_grad():
                        pseudo_raw = model(x_unlabeled).view(-1)
                        pseudo_targets = pseudo_targets_from_outputs(
                            pseudo_raw,
                            min_label=min_label,
                            max_label=max_label,
                        )
                        distance_to_label = (pseudo_raw - pseudo_targets).abs()
                        mask = distance_to_label <= pseudo_threshold
                    if mask.any():
                        pseudo_outputs = model(x_unlabeled[mask]).view(-1)
                        loss = loss + pseudo_weight * criterion(pseudo_outputs, pseudo_targets[mask])
                        pseudo_used += int(mask.sum().item())

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        total_loss += loss.item() * x.size(0)
        preds = rounded_predictions(outputs, min_label=min_label, max_label=max_label)
        targets = y.view(-1).round().long()
        correct += (preds == targets).sum().item()
        mae_sum += (outputs.detach() - y.view(-1)).abs().sum().item()
        total += x.size(0)

    return {
        "loss": total_loss / max(total, 1),
        "mae": mae_sum / max(total, 1),
        "rounded_accuracy": correct / max(total, 1),
        "pseudo_used": pseudo_used,
    }


def save_checkpoint(path, model, optimizer, epoch, metrics, args):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "args": vars(args),
    }
    torch.save(checkpoint, path)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print("Using", "GPU" if device.type == "cuda" else "CPU")
    os.makedirs(args.save_dir, exist_ok=True)

    transform = build_transform(args.augment)
    dataset = CsvVolumeDataset(
        args.train_dir,
        args.label_csv,
        filename_column=args.filename_column,
        label_column=args.label_column,
        transform=transform,
        label_dtype=torch.float32,
        allowed_labels=range(args.min_label, args.max_label + 1),
    )
    train_dataset, val_dataset = split_train_val(dataset, args.val_split, args.random_state)
    train_loader = make_loader(train_dataset, args.batch_size, True, args.num_workers, device)
    val_loader = make_loader(val_dataset, args.batch_size, False, args.num_workers, device)

    unlabeled_loader = None
    if args.unlabeled_dir:
        unlabeled_dataset = UnlabeledVolumeDataset(args.unlabeled_dir, transform=transform)
        unlabeled_loader = make_loader(unlabeled_dataset, args.batch_size, True, args.num_workers, device)

    model = ResNet3D(output_size=1, base_channels=args.base_channels, dropout=args.dropout).to(device)
    criterion = nn.SmoothL1Loss() if args.loss == "smooth_l1" else nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    run_dir = Path(args.save_dir) / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "clf_3d_config.json", "w") as file:
        json.dump(vars(args), file, indent=4)

    best_val_loss = np.inf
    epochs_without_improvement = 0
    log_path = run_dir / "clf_3d_log.csv"
    with open(log_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "train_loss", "train_mae", "train_rounded_accuracy", "val_loss", "val_mae", "val_rounded_accuracy", "pseudo_used", "lr"])

    for epoch in range(args.epochs):
        active_unlabeled_loader = unlabeled_loader if epoch + 1 >= args.pseudo_start_epoch else None
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            amp=args.amp,
            min_label=args.min_label,
            max_label=args.max_label,
            unlabeled_loader=active_unlabeled_loader,
            pseudo_weight=args.pseudo_weight,
            pseudo_threshold=args.pseudo_threshold,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            amp=args.amp,
            min_label=args.min_label,
            max_label=args.max_label,
        )
        scheduler.step()
        lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch [{epoch + 1}/{args.epochs}] "
            f"Train loss: {train_metrics['loss']:.4f} mae: {train_metrics['mae']:.4f} "
            f"rounded acc: {train_metrics['rounded_accuracy']:.4f} | "
            f"Val loss: {val_metrics['loss']:.4f} mae: {val_metrics['mae']:.4f} "
            f"rounded acc: {val_metrics['rounded_accuracy']:.4f} | "
            f"Pseudo: {train_metrics['pseudo_used']}"
        )

        with open(log_path, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                epoch + 1,
                train_metrics["loss"],
                train_metrics["mae"],
                train_metrics["rounded_accuracy"],
                val_metrics["loss"],
                val_metrics["mae"],
                val_metrics["rounded_accuracy"],
                train_metrics["pseudo_used"],
                lr,
            ])

        save_checkpoint(run_dir / "latest.pth", model, optimizer, epoch + 1, val_metrics, args)
        improved = val_metrics["loss"] < best_val_loss - args.early_stop_min_delta
        if improved:
            best_val_loss = val_metrics["loss"]
            epochs_without_improvement = 0
            save_checkpoint(run_dir / "best.pth", model, optimizer, epoch + 1, val_metrics, args)
            print("New best regressor found")
        else:
            epochs_without_improvement += 1
            if args.early_stop_patience > 0:
                print(
                    f"No validation loss improvement for "
                    f"{epochs_without_improvement}/{args.early_stop_patience} epochs"
                )

        if 0 < args.early_stop_patience <= epochs_without_improvement:
            print(
                f"Early stopping at epoch {epoch + 1}. "
                f"Best validation loss: {best_val_loss:.4f}"
            )
            break

    print(f"Training complete. Outputs saved in: {run_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a 3D ResNet regressor on NIfTI volumes with rounded 0-3 outputs")
    parser.add_argument("--train_dir", type=str, required=True, help="folder containing all training NIfTI volumes")
    parser.add_argument("--label_csv", type=str, required=True, help="CSV with filename and target columns")
    parser.add_argument("--save_dir", type=str, required=True, help="directory for checkpoints and logs")
    parser.add_argument("--unlabeled_dir", type=str, default="", help="optional folder for regression pseudo-label training")
    parser.add_argument("--filename_column", type=str, default="filename", help="CSV filename column")
    parser.add_argument("--label_column", type=str, default="label", help="CSV target column")
    parser.add_argument("--min_label", type=int, default=0, help="minimum rounded output label")
    parser.add_argument("--max_label", type=int, default=3, help="maximum rounded output label")
    parser.add_argument("--base_channels", type=int, default=16, help="3D ResNet width; 16 is memory-friendly for 128^3")
    parser.add_argument("--dropout", type=float, default=0.2, help="regressor dropout")
    parser.add_argument("--loss", choices=("mse", "smooth_l1"), default="mse", help="regression loss")
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size for 128^3 volumes")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="AdamW weight decay")
    parser.add_argument("--val_split", type=float, default=0.3, help="validation split ratio")
    parser.add_argument("--random_state", type=int, default=42, help="random seed for train/validation split")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--pseudo_start_epoch", type=int, default=0, help="first epoch that uses pseudo labels")
    parser.add_argument("--pseudo_threshold", type=float, default=0.25, help="maximum distance from rounded label for pseudo labels")
    parser.add_argument("--pseudo_weight", type=float, default=0.3, help="loss weight for pseudo-labeled samples")
    parser.add_argument("--early_stop_patience", type=int, default=20, help="epochs without validation loss improvement before stopping; set 0 to disable")
    parser.add_argument("--early_stop_min_delta", type=float, default=0.0, help="minimum validation loss drop counted as improvement")
    parser.add_argument("--augment", action="store_true", help="enable light 3D augmentation with torchio")
    parser.add_argument("--amp", action="store_true", help="use CUDA automatic mixed precision")
    parser.add_argument("--cpu", action="store_true", help="force CPU training")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
