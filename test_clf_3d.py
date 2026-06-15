import argparse
import csv
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm

from train_clf_3d import ResNet3D, make_loader, run_epoch
from utils.volumes import load_nifti, volume_to_tensor


class CsvClassificationDataset(Dataset):
    def __init__(self, image_dir, csv_path, filename_column="filename", label_column="label"):
        self.image_dir = Path(image_dir)
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
        return volume, torch.tensor(label, dtype=torch.long)


def checkpoint_args(checkpoint):
    args = checkpoint.get("args", {})
    return {
        "num_classes": int(args.get("num_classes", 2)),
        "base_channels": int(args.get("base_channels", 16)),
        "dropout": float(args.get("dropout", 0.2)),
    }


def confusion_matrix(model, loader, num_classes, device):
    matrix = torch.zeros((num_classes, num_classes), dtype=torch.long)
    model.eval()
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Confusion", unit="batch"):
            x = x.to(device, non_blocking=True)
            logits = model(x)
            preds = logits.argmax(dim=1).cpu()
            for target, pred in zip(y.view(-1), preds.view(-1)):
                matrix[int(target), int(pred)] += 1
    return matrix


def test(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print("Using", "GPU" if device.type == "cuda" else "CPU")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_kwargs = checkpoint_args(checkpoint)
    model = ResNet3D(**model_kwargs).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    dataset = CsvClassificationDataset(
        args.test_dir,
        args.test_csv,
        filename_column=args.filename_column,
        label_column=args.label_column,
    )
    loader = make_loader(dataset, args.batch_size, False, args.num_workers, device)
    criterion = nn.CrossEntropyLoss()

    metrics = run_epoch(model, loader, criterion, device, amp=args.amp)
    matrix = confusion_matrix(model, loader, model_kwargs["num_classes"], device)

    print(f"Test loss: {metrics['loss']:.4f}")
    print(f"Test accuracy: {metrics['accuracy']:.4f}")
    print("Confusion matrix rows=true labels, columns=predicted labels:")
    print(matrix.numpy())


def parse_args():
    parser = argparse.ArgumentParser(description="Test a trained 3D ResNet classifier checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="path to best.pth or another classifier checkpoint")
    parser.add_argument("--test_dir", type=str, required=True, help="folder containing test volumes")
    parser.add_argument("--test_csv", type=str, required=True, help="CSV with test filenames and labels")
    parser.add_argument("--filename_column", type=str, default="filename", help="test CSV filename column")
    parser.add_argument("--label_column", type=str, default="label", help="test CSV label column")
    parser.add_argument("--batch_size", type=int, default=2, help="test batch size")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--amp", action="store_true", help="use CUDA automatic mixed precision")
    parser.add_argument("--cpu", action="store_true", help="force CPU testing")
    return parser.parse_args()


if __name__ == "__main__":
    test(parse_args())
