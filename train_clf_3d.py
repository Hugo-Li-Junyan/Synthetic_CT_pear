import argparse
import csv
import json
import os
import time
from itertools import cycle
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from utils.splits import split_train_val
from utils.volumes import list_nifti_files, load_nifti, volume_to_tensor


class FolderPairClassificationDataset(Dataset):
    def __init__(self, class0_dir, class1_dir, transform=None):
        self.samples = [(path, 0) for path in list_nifti_files(class0_dir)]
        self.samples += [(path, 1) for path in list_nifti_files(class1_dir)]
        self.transform = transform
        if not self.samples:
            raise ValueError("No NIfTI files found in the class folders")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        volume = volume_to_tensor(load_nifti(path))
        if self.transform:
            volume = self.transform(volume)
        return volume, torch.tensor(label, dtype=torch.long)


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


class CsvClassificationDataset(Dataset):
    def __init__(self, image_dir, csv_path, filename_column="filename", label_column="label", transform=None):
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.samples = self._read_samples(csv_path, filename_column, label_column)
        if not self.samples:
            raise ValueError(f"No labeled samples found in CSV: {csv_path}")

    def _read_samples(self, csv_path, filename_column, label_column):
        samples = []
        with open(csv_path, newline="") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames is None:
                raise ValueError(f"CSV has no header row: {csv_path}")
            if filename_column not in reader.fieldnames:
                raise ValueError(f"CSV is missing filename column: {filename_column}")
            if label_column not in reader.fieldnames:
                raise ValueError(f"CSV is missing label column: {label_column}")

            for row in reader:
                filename = row[filename_column].strip()
                label = int(row[label_column])
                path = self.image_dir / filename
                if not path.exists():
                    raise FileNotFoundError(f"CSV sample does not exist: {path}")
                samples.append((path, label))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        volume = volume_to_tensor(load_nifti(path))
        if self.transform:
            volume = self.transform(volume)
        return volume, torch.tensor(label, dtype=torch.long)


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
    def __init__(self, block=BasicBlock3D, layers=(2, 2, 2, 2), num_classes=2, base_channels=16, dropout=0.2):
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
        self.fc = nn.Linear(base_channels * 8 * block.expansion, num_classes)

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


def class_weights_from_dataset(dataset, num_classes, device):
    labels = []
    source = dataset.dataset if isinstance(dataset, Subset) else dataset
    indices = dataset.indices if isinstance(dataset, Subset) else range(len(dataset))
    for idx in indices:
        labels.append(source.samples[idx][1])

    counts = torch.bincount(torch.tensor(labels, dtype=torch.long), minlength=num_classes).float()
    weights = counts.sum() / (counts.clamp_min(1.0) * num_classes)
    return weights.to(device)


def run_epoch(model, loader, criterion, device, optimizer=None, amp=False, unlabeled_loader=None,
              pseudo_weight=0.3, pseudo_threshold=0.95):
    is_train = optimizer is not None
    model.train(is_train)
    scaler = torch.cuda.amp.GradScaler(enabled=amp and device.type == "cuda")
    total_loss = 0.0
    correct = 0
    total = 0
    pseudo_used = 0
    unlabeled_iter = cycle(unlabeled_loader) if is_train and unlabeled_loader is not None else None

    for x, y in tqdm(loader, desc="Train" if is_train else "Eval", unit="batch"):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.set_grad_enabled(is_train):
            with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
                logits = model(x)
                loss = criterion(logits, y)

                if unlabeled_iter is not None:
                    x_unlabeled = next(unlabeled_iter).to(device, non_blocking=True)
                    with torch.no_grad():
                        pseudo_probs = torch.softmax(model(x_unlabeled), dim=1)
                        confidence, pseudo_labels = pseudo_probs.max(dim=1)
                        mask = confidence >= pseudo_threshold
                    if mask.any():
                        pseudo_logits = model(x_unlabeled[mask])
                        loss = loss + pseudo_weight * criterion(pseudo_logits, pseudo_labels[mask])
                        pseudo_used += int(mask.sum().item())

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += x.size(0)

    return {
        "loss": total_loss / max(total, 1),
        "accuracy": correct / max(total, 1),
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
    dataset = FolderPairClassificationDataset(args.class0_dir, args.class1_dir, transform=transform)
    train_dataset, val_dataset = split_train_val(dataset, args.val_split, args.random_state)
    train_loader = make_loader(train_dataset, args.batch_size, True, args.num_workers, device)
    val_loader = make_loader(val_dataset, args.batch_size, False, args.num_workers, device)

    unlabeled_loader = None
    if args.unlabeled_dir:
        unlabeled_dataset = UnlabeledVolumeDataset(args.unlabeled_dir, transform=transform)
        unlabeled_loader = make_loader(unlabeled_dataset, args.batch_size, True, args.num_workers, device)

    model = ResNet3D(num_classes=args.num_classes, base_channels=args.base_channels, dropout=args.dropout).to(device)
    weights = class_weights_from_dataset(train_dataset, args.num_classes, device) if args.class_weighted_loss else None
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    run_dir = Path(args.save_dir) / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "clf_3d_config.json", "w") as file:
        json.dump(vars(args), file, indent=4)

    best_val_acc = 0.0
    epochs_without_improvement = 0
    log_path = run_dir / "clf_3d_log.csv"
    with open(log_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy", "pseudo_used", "lr"])

    for epoch in range(args.epochs):
        active_unlabeled_loader = unlabeled_loader if epoch + 1 >= args.pseudo_start_epoch else None
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            amp=args.amp,
            unlabeled_loader=active_unlabeled_loader,
            pseudo_weight=args.pseudo_weight,
            pseudo_threshold=args.pseudo_threshold,
        )
        val_metrics = run_epoch(model, val_loader, criterion, device, amp=args.amp)
        scheduler.step()
        lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch [{epoch + 1}/{args.epochs}] "
            f"Train loss: {train_metrics['loss']:.4f} acc: {train_metrics['accuracy']:.4f} | "
            f"Val loss: {val_metrics['loss']:.4f} acc: {val_metrics['accuracy']:.4f} | "
            f"Pseudo: {train_metrics['pseudo_used']}"
        )

        with open(log_path, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                epoch + 1,
                train_metrics["loss"],
                train_metrics["accuracy"],
                val_metrics["loss"],
                val_metrics["accuracy"],
                train_metrics["pseudo_used"],
                lr,
            ])

        save_checkpoint(run_dir / "latest.pth", model, optimizer, epoch + 1, val_metrics, args)
        improved = val_metrics["accuracy"] > best_val_acc + args.early_stop_min_delta
        if improved:
            best_val_acc = val_metrics["accuracy"]
            epochs_without_improvement = 0
            save_checkpoint(run_dir / "best.pth", model, optimizer, epoch + 1, val_metrics, args)
            print("New best classifier found")
        else:
            epochs_without_improvement += 1
            if args.early_stop_patience > 0:
                print(
                    f"No validation accuracy improvement for "
                    f"{epochs_without_improvement}/{args.early_stop_patience} epochs"
                )

        if args.early_stop_patience > 0 and epochs_without_improvement >= args.early_stop_patience:
            print(
                f"Early stopping at epoch {epoch + 1}. "
                f"Best validation accuracy: {best_val_acc:.4f}"
            )
            break

    if args.test_dir and args.test_csv:
        test_dataset = CsvClassificationDataset(
            args.test_dir,
            args.test_csv,
            filename_column=args.filename_column,
            label_column=args.label_column,
        )
        test_loader = make_loader(test_dataset, args.batch_size, False, args.num_workers, device)
        checkpoint = torch.load(run_dir / "best.pth", map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        test_metrics = run_epoch(model, test_loader, criterion, device, amp=args.amp)
        print(f"Test loss: {test_metrics['loss']:.4f} | Test accuracy: {test_metrics['accuracy']:.4f}")

    print(f"Training complete. Outputs saved in: {run_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a 3D ResNet classifier on NIfTI volumes")
    parser.add_argument("--class0_dir", type=str, required=True, help="folder for class 0 volumes")
    parser.add_argument("--class1_dir", type=str, required=True, help="folder for class 1 volumes")
    parser.add_argument("--save_dir", type=str, required=True, help="directory for checkpoints and logs")
    parser.add_argument("--unlabeled_dir", type=str, default="", help="optional folder for pseudo-label training")
    parser.add_argument("--test_dir", type=str, default="", help="optional test image folder")
    parser.add_argument("--test_csv", type=str, default="", help="optional CSV with test labels")
    parser.add_argument("--filename_column", type=str, default="filename", help="test CSV filename column")
    parser.add_argument("--label_column", type=str, default="label", help="test CSV label column")
    parser.add_argument("--num_classes", type=int, default=2, help="number of classes")
    parser.add_argument("--base_channels", type=int, default=16, help="3D ResNet width; 16 is memory-friendly for 128^3")
    parser.add_argument("--dropout", type=float, default=0.2, help="classifier dropout")
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size for 128^3 volumes")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="AdamW weight decay")
    parser.add_argument("--val_split", type=float, default=0.15, help="validation split ratio")
    parser.add_argument("--random_state", type=int, default=42, help="random seed for train/validation split")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--pseudo_start_epoch", type=int, default=5, help="first epoch that uses pseudo labels")
    parser.add_argument("--pseudo_threshold", type=float, default=0.95, help="minimum confidence for pseudo labels")
    parser.add_argument("--pseudo_weight", type=float, default=0.3, help="loss weight for pseudo-labeled samples")
    parser.add_argument("--early_stop_patience", type=int, default=20, help="epochs without validation accuracy improvement before stopping; set 0 to disable")
    parser.add_argument("--early_stop_min_delta", type=float, default=0.0, help="minimum validation accuracy gain counted as improvement")
    parser.add_argument("--class_weighted_loss", action="store_true", help="use inverse-frequency class weights")
    parser.add_argument("--augment", action="store_true", help="enable light 3D augmentation with torchio")
    parser.add_argument("--amp", action="store_true", help="use CUDA automatic mixed precision")
    parser.add_argument("--cpu", action="store_true", help="force CPU training")
    args = parser.parse_args()
    if bool(args.test_dir) != bool(args.test_csv):
        parser.error("--test_dir and --test_csv must be provided together")
    return args


if __name__ == "__main__":
    train(parse_args())
