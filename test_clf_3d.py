import argparse
import csv
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm

from train_clf_3d import ResNet3D, make_loader, rounded_predictions, run_epoch
from utils.volumes import load_nifti, volume_to_tensor


class CsvRegressionDataset(Dataset):
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
                label = float(row[label_column])
                if label not in (0.0, 1.0, 2.0, 3.0):
                    raise ValueError(f"Label must be one of 0, 1, 2, 3 for {filename}; got {label}")
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
        return volume, torch.tensor(label, dtype=torch.float32)


def checkpoint_args(checkpoint):
    args = checkpoint.get("args", {})
    return {
        "output_size": int(args.get("output_size", 1)),
        "base_channels": int(args.get("base_channels", 16)),
        "dropout": float(args.get("dropout", 0.2)),
    }


def confusion_matrix(model, loader, min_label, max_label, device):
    label_count = max_label - min_label + 1
    matrix = torch.zeros((label_count, label_count), dtype=torch.long)
    model.eval()
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Confusion", unit="batch"):
            x = x.to(device, non_blocking=True)
            outputs = model(x).view(-1)
            preds = rounded_predictions(outputs, min_label=min_label, max_label=max_label).cpu()
            for target, pred in zip(y.view(-1), preds.view(-1)):
                matrix[int(target.item()) - min_label, int(pred.item()) - min_label] += 1
    return matrix


def collect_predictions(model, loader, min_label, max_label, device):
    rows = []
    model.eval()
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Predict", unit="batch"):
            x = x.to(device, non_blocking=True)
            outputs = model(x).view(-1).cpu()
            preds = rounded_predictions(outputs, min_label=min_label, max_label=max_label).cpu()
            for target, output, pred in zip(y.view(-1), outputs.view(-1), preds.view(-1)):
                rows.append((float(target.item()), float(output.item()), int(pred.item())))
    return rows


def test(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print("Using", "GPU" if device.type == "cuda" else "CPU")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_kwargs = checkpoint_args(checkpoint)
    model = ResNet3D(**model_kwargs).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    dataset = CsvRegressionDataset(
        args.test_dir,
        args.test_csv,
        filename_column=args.filename_column,
        label_column=args.label_column,
    )
    loader = make_loader(dataset, args.batch_size, False, args.num_workers, device)
    criterion = nn.SmoothL1Loss() if args.loss == "smooth_l1" else nn.MSELoss()

    metrics = run_epoch(
        model,
        loader,
        criterion,
        device,
        amp=args.amp,
        min_label=args.min_label,
        max_label=args.max_label,
    )
    matrix = confusion_matrix(model, loader, args.min_label, args.max_label, device)

    print(f"Test loss: {metrics['loss']:.4f}")
    print(f"Test MAE: {metrics['mae']:.4f}")
    print(f"Test rounded accuracy: {metrics['rounded_accuracy']:.4f}")
    print(f"Confusion matrix rows=true labels {args.min_label}-{args.max_label}, columns=rounded predictions:")
    print(matrix.numpy())

    if args.predictions_csv:
        rows = collect_predictions(model, loader, args.min_label, args.max_label, device)
        with open(args.predictions_csv, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["target", "raw_prediction", "rounded_prediction"])
            writer.writerows(rows)
        print(f"Predictions saved to: {args.predictions_csv}")


def parse_args():
    parser = argparse.ArgumentParser(description="Test a trained 3D ResNet regression checkpoint with rounded 0-3 outputs")
    parser.add_argument("--checkpoint", type=str, required=True, help="path to best.pth or another regression checkpoint")
    parser.add_argument("--test_dir", type=str, required=True, help="folder containing test volumes")
    parser.add_argument("--test_csv", type=str, required=True, help="CSV with test filenames and 0-3 labels")
    parser.add_argument("--filename_column", type=str, default="filename", help="test CSV filename column")
    parser.add_argument("--label_column", type=str, default="label", help="test CSV label column")
    parser.add_argument("--min_label", type=int, default=0, help="minimum rounded output label")
    parser.add_argument("--max_label", type=int, default=3, help="maximum rounded output label")
    parser.add_argument("--loss", choices=("mse", "smooth_l1"), default="mse", help="regression loss")
    parser.add_argument("--predictions_csv", type=str, default="", help="optional CSV path for raw and rounded predictions")
    parser.add_argument("--batch_size", type=int, default=2, help="test batch size")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--amp", action="store_true", help="use CUDA automatic mixed precision")
    parser.add_argument("--cpu", action="store_true", help="force CPU testing")
    return parser.parse_args()


if __name__ == "__main__":
    test(parse_args())
