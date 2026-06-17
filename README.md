# Synthetic CT Pear

This repository trains and evaluates models for 3D pear CT volumes stored as NIfTI files (`.nii` or `.nii.gz`).

It currently contains three main workflows:

1. Train a VAE-GAN that compresses and reconstructs 3D CT volumes.
2. Train a latent diffusion model on the VAE latent space, then generate synthetic 3D pears.
3. Train a 3D ResNet classifier on real, synthetic, or pseudo-labeled CT volumes.

The default scripts assume single-channel `128 x 128 x 128` volumes.

## Repository Layout

```text
component/              Model and dataset components used by generation scripts
scripts/                Evaluation, interpolation, and latent-space utilities
slurm/                  Example SLURM jobs for cluster runs
utils/                  Shared helpers for metrics, model loading, volume I/O, and splitting
main.py                 Generate synthetic CT volumes with a trained VAE + diffuser
train_vae.py            Train the VAE-GAN reconstruction model
train_diffuser.py       Train latent diffusion on top of a trained VAE
train_clf_3d.py         Train a 3D ResNet classifier
test_clf_3d.py          Test a saved 3D ResNet classifier checkpoint
interpolation_line.py   Interpolate between two CT volumes in latent space
med3d_fid.py            Compute 3D FID-style metrics with a pretrained 3D ResNet
```

## Data Format

Labeled training/evaluation scripts expect one image folder plus a CSV file:

```text
data/
  images/
    A01.nii
    A02.nii
    B01.nii
  labels.csv
```

The CSV should contain at least a filename column and a label column:

```csv
filename,label
A01.nii,0
A02.nii,0
B01.nii,1
```

Files are loaded with `nibabel`, min-max normalized to `[0, 1]`, and converted to PyTorch tensors with shape:

```text
(channels, depth, height, width) = (1, 128, 128, 128)
```

The dataset loader only reads `.nii` and `.nii.gz` files.

## Installation

```bash
git clone https://github.com/Hugo-Li-Junyan/Synthetic_CT_pear.git
cd Synthetic_CT_pear

python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate

pip install -r requirements.txt
```

Install PyTorch separately if your CUDA setup requires a specific build.

## Train the VAE-GAN

The VAE-GAN learns to reconstruct 3D pear CT volumes. Its checkpoint is also used later by the diffusion model.

```bash
python train_vae.py \
  --image_dir /path/to/images \
  --labels_csv /path/to/labels.csv \
  --save_dir /path/to/models/vae \
  --epochs 500 \
  --batch_size 4
```

Each run creates a timestamped folder under `--save_dir` containing:

```text
checkpoint.pth
best.pth
vae_log.csv
vae_hyperparameter.json
```

## Train the Latent Diffusion Model

After training the VAE, train the diffuser using the VAE run folder as `--model_id`.

```bash
python train_diffuser.py \
  --image_dir /path/to/images \
  --labels_csv /path/to/labels.csv \
  --save_dir /path/to/models/vae \
  --model_id 20250626-021325 \
  --epochs 800 \
  --batch_size 16
```

The diffusion files are saved inside the same model folder:

```text
diffuser_checkpoint.pth
diffuser_best.pth
diffuser_log.csv
diffuser_hyperparameter.json
```

## Generate Synthetic CT Volumes

Use a folder that contains both the trained VAE checkpoint and diffuser checkpoint.

```bash
python main.py \
  --model_dir /path/to/models/vae/20250626-021325 \
  --save_dir /path/to/generated_pears \
  --batch_size 2 \
  --batches 3000
```

The script writes generated `.nii` files to `--save_dir`.

Total generated samples:

```text
batch_size * batches
```

## Latent Interpolation

Interpolate between one healthy and one defective CT volume.

```bash
python interpolation_line.py \
  --healthy_pth /path/to/healthy/A01.nii \
  --defective_pth /path/to/defective/B01.nii \
  --model_dir /path/to/models/vae/20250626-021325 \
  --save_dir /path/to/interpolation \
  --num_steps 11
```

Optional flags:

```bash
--diffusion      Denoise interpolated latents with the diffusion model
--show_latent    Save latent-space slices instead of decoded volumes
```

Outputs:

```text
interpolation_line.png
0.nii
1.nii
...
```

## Train a 3D ResNet Classifier

`train_clf_3d.py` trains a compact 3D ResNet for binary or multi-class volume classification. It is designed to be practical for `128 x 128 x 128` CT images and datasets in the range of hundreds to a few thousand samples.

Basic CSV-labeled training:

```bash
python train_clf_3d.py \
  --train_dir /path/to/images \
  --label_csv /path/to/labels.csv \
  --save_dir /path/to/models/classifier \
  --epochs 100 \
  --batch_size 2 \
  --augment \
  --class_weighted_loss
```

Training with an unlabeled folder for pseudo-labeling:

```bash
python train_clf_3d.py \
  --train_dir /path/to/images \
  --label_csv /path/to/labels.csv \
  --unlabeled_dir /path/to/unlabeled \
  --save_dir /path/to/models/classifier \
  --pseudo_start_epoch 5 \
  --pseudo_threshold 0.95 \
  --pseudo_weight 0.3
```

Run final test evaluation from a separate image folder and CSV after training:

```bash
python test_clf_3d.py \
  --checkpoint /path/to/models/classifier/20250626-021325/best.pth \
  --test_dir /path/to/test_images \
  --test_csv /path/to/test_labels.csv
```

The test CSV should contain at least:

```csv
filename,label
A01.nii,0
B01.nii,1
```

Useful classifier options:

```text
--base_channels 16          Model width. Increase if you have enough GPU memory.
--batch_size 2              Safe starting point for 128^3 volumes.
--amp                       Use CUDA mixed precision.
--early_stop_patience 20    Stop after this many stale validation epochs.
--early_stop_min_delta 0.0  Minimum validation accuracy gain counted as improvement.
--cpu                       Force CPU training.
```

Classifier outputs:

```text
clf_3d_config.json
clf_3d_log.csv
latest.pth
best.pth
```

## Evaluation

Evaluate a trained VAE reconstruction model:

```bash
python scripts/evaluate.py \
  --model_dir /path/to/models/vae/20250626-021325 \
  --image_dir /path/to/images \
  --labels_csv /path/to/labels.csv \
  --batch_size 1
```

The script reports MAE, SSIM, and PSNR on a validation split reconstructed by the VAE.

## Practical Notes

- Keep raw data, generated `.nii` files, checkpoints, and `.npy` activation caches outside git.
- Start with small classifier batches for `128^3` volumes. Increase batch size only after checking GPU memory.
- Use `--augment` for classifier training when the dataset is small.
- For pseudo-labeling, keep `--pseudo_threshold` high at first, such as `0.95`, to avoid reinforcing low-confidence mistakes.
- The VAE and diffusion scripts currently assume cubic 3D volumes and single-channel input.

## Typical Full Pipeline

```bash
# 1. Train VAE-GAN
python train_vae.py --image_dir data/images --labels_csv data/labels.csv --save_dir models/vae

# 2. Train latent diffusion using the VAE run ID
python train_diffuser.py --image_dir data/images --labels_csv data/labels.csv --save_dir models/vae --model_id 20250626-021325

# 3. Generate synthetic pears
python main.py --model_dir models/vae/20250626-021325 --save_dir outputs/generated --batch_size 2 --batches 3000

# 4. Train classifier with real data and optional pseudo-labeled data
python train_clf_3d.py --train_dir data/images --label_csv data/labels.csv --unlabeled_dir outputs/generated --save_dir models/classifier
```
