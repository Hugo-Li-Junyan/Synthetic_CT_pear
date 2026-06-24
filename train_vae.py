import json
import warnings
import torchio as tio
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import os
from component.dataset import CsvVolumeDataset
import torch
import csv
from component.vae import VAE
from component.gan import PatchGAN
import gc         # garbage collect library
import argparse
from tqdm import tqdm
from utils.metrics import mae, ssim, psnr
from utils.splits import split_train_val_test


def train(dataset, vae, save_dir, gan=None, vae_lr=1e-4, gan_lr=1e-4, epochs=500, batch_size=8,
          val_split=0.1, test_split=0.1, load_model_id=None, beta=1e-6, gamma=0.01,
          loss_criterion='MAE', random_state=42, amp=False, num_workers=0,
          lr_scheduler_patience=20, lr_scheduler_factor=0.5):
    # device ready
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using', 'GPU' if torch.cuda.is_available() else 'CPU')
    vae = vae.to(device)
    use_gan = gan is not None
    if use_gan:
        gan = gan.to(device)

    # components ready
    vae_optimizer = optim.Adam(vae.parameters(), lr=vae_lr)
    gan_optimizer = optim.Adam(gan.parameters(), betas=(0.5, 0.999), lr=gan_lr) if use_gan else None
    vae_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        vae_optimizer,
        mode='min',
        factor=lr_scheduler_factor,
        patience=lr_scheduler_patience,
    )
    gan_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        gan_optimizer,
        mode='min',
        factor=lr_scheduler_factor,
        patience=lr_scheduler_patience,
    ) if use_gan else None
    scaler = torch.amp.GradScaler('cuda', enabled=amp and device.type == 'cuda')

    # load from checkpoint
    start_epoch = 0
    if load_model_id:
        checkpoint_path = os.path.join(save_dir, load_model_id, 'checkpoint.pth')
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            random_state = checkpoint['random_state']
            vae.load_state_dict(checkpoint['vae_state_dict'])
            vae_optimizer.load_state_dict(checkpoint['vae_optimizer_state_dict'])
            if 'vae_scheduler_state_dict' in checkpoint:
                vae_scheduler.load_state_dict(checkpoint['vae_scheduler_state_dict'])
            if 'scaler_state_dict' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
            if use_gan:
                if 'gan_state_dict' in checkpoint and 'gan_optimizer_state_dict' in checkpoint:
                    gan.load_state_dict(checkpoint['gan_state_dict'])
                    gan_optimizer.load_state_dict(checkpoint['gan_optimizer_state_dict'])
                    if 'gan_scheduler_state_dict' in checkpoint:
                        gan_scheduler.load_state_dict(checkpoint['gan_scheduler_state_dict'])
                else:
                    warnings.warn('checkpoint does not contain GAN state, starting GAN from scratch')
            start_epoch = checkpoint['epoch']
            print(f'Loaded model from {checkpoint_path}')
            del checkpoint
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        else:
            warnings.warn(f'checkpoint path {checkpoint_path} not exists, skip loading')

    # train validation separation
    train_dataset, val_dataset, test_dataset = split_train_val_test(dataset, val_split, test_split, random_state)
    train_size = len(train_dataset)
    val_size = len(val_dataset)
    test_size = len(test_dataset)
    loader_kwargs = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': device.type == 'cuda',
        'persistent_workers': num_workers > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    # training settings
    hyperparameters = {'model': 'VAE-GAN' if use_gan else 'VAE', 'use_gan': use_gan,
                       'vae_latent_space': vae.latent_space,
                       'vae_featuremap_size': vae.featuremap_size, 'vae_base_channel': vae.base_channel,
                       'vae_use_residual': vae.with_residual, 'vae learning rate': vae_lr,
                       'loss_fn': loss_criterion, 'epochs': epochs, 'batch_size': batch_size,
                       'beta': beta, 'vae_optimizer': 'Adam', 'amp': amp, 'num_workers': num_workers,
                       'lr_scheduler': 'ReduceLROnPlateau',
                       'lr_scheduler_patience': lr_scheduler_patience,
                       'lr_scheduler_factor': lr_scheduler_factor}
    if use_gan:
        hyperparameters.update({'gan_optimizer': 'Adam', 'gan learning rate': gan_lr, 'gamma': gamma,
                                'gan_patch_size': gan.patch_size, 'gan_base_channel': gan.base_channel,
                                'gan_with_residual': gan.with_residual, 'gan_weight_function': gan.weight_fn})

    # for saving
    os.makedirs(save_dir, exist_ok=True)
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    model_dir = os.path.join(save_dir, timestamp)
    os.makedirs(model_dir, exist_ok=True)
    log_path = os.path.join(model_dir, 'vae_log.csv')
    checkpoint_path = os.path.join(model_dir, 'checkpoint.pth')
    best_model_path = os.path.join(model_dir, 'best.pth')
    hyperparameter_path = os.path.join(model_dir, 'vae_hyperparameter.json')
    with open(hyperparameter_path, 'w') as f:
        json.dump(hyperparameters, f, indent=4)

    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Train_Recon', 'Train_KL', 'Train_Adv', 'Train_GAN',
                         'Val_Recon', 'Val_KL', 'Val_Adv', 'Val_GAN', 'Beta', 'VAE_LR', 'GAN_LR'])

    # start training
    print('Training starts now')
    best_val_score = torch.inf
    for epoch in range(start_epoch, epochs + 1):
        # train
        vae.train()
        train_recon_loss, train_kl_loss = 0.0, 0.0
        train_adv_loss, train_gan_loss = 0.0, 0.0
        for data in train_loader:
            x, _ = data
            x = x.to(device, non_blocking=True)

            with torch.amp.autocast('cuda', enabled=amp and device.type == 'cuda'):
                reconstructed_x, z_mean, z_logvar = vae(x)

            if use_gan:
                # train gan
                gan.train()
                for param in gan.parameters():
                    param.requires_grad = True
                with torch.amp.autocast('cuda', enabled=amp and device.type == 'cuda'):
                    gan_loss = gan.loss_function(x, reconstructed_x.detach())
                gan_optimizer.zero_grad(set_to_none=True)
                scaler.scale(gan_loss).backward()
                scaler.unscale_(gan_optimizer)
                torch.nn.utils.clip_grad_norm_(gan.parameters(), 1.0)
                scaler.step(gan_optimizer)

            # train vae
            with torch.amp.autocast('cuda', enabled=amp and device.type == 'cuda'):
                recon_loss, kl_loss = vae.loss_function(reconstructed_x, x, z_mean, z_logvar, beta=beta,
                                                        criterion=loss_criterion)
                vae_loss = recon_loss + kl_loss
                if use_gan:
                    gan.eval()
                    for param in gan.parameters():
                        param.requires_grad = False
                    adv_loss = gan.adversarial_loss(x, reconstructed_x)
                    vae_loss = vae_loss + adv_loss * gamma
            vae_optimizer.zero_grad(set_to_none=True)
            scaler.scale(vae_loss).backward()
            scaler.unscale_(vae_optimizer)
            torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
            scaler.step(vae_optimizer)
            scaler.update()

            train_recon_loss += recon_loss.item() * x.size(0)
            train_kl_loss += kl_loss.item() * x.size(0)

            if use_gan:
                train_adv_loss += adv_loss.item() * x.size(0)
                train_gan_loss += gan_loss.item() * x.size(0)
        train_recon_loss /= train_size
        train_kl_loss /= train_size
        train_adv_loss /= train_size
        train_gan_loss /= train_size

        # validation
        vae.eval()
        if use_gan:
            gan.eval()
        val_recon_loss, val_kl_loss = 0.0, 0.0
        val_adv_loss, val_gan_loss = 0.0, 0.0

        with torch.no_grad():
            for data in val_loader:
                x, _ = data
                x = x.to(device, non_blocking=True)
                with torch.amp.autocast('cuda', enabled=amp and device.type == 'cuda'):
                    reconstructed_x, z_mean, z_logvar = vae(x)
                    recon_loss, kl_loss = vae.loss_function(
                        reconstructed_x, x, z_mean, z_logvar, beta=beta, criterion=loss_criterion
                    )

                val_recon_loss += recon_loss.item() * x.size(0)
                val_kl_loss += kl_loss.item() * x.size(0)
                if use_gan:
                    with torch.amp.autocast('cuda', enabled=amp and device.type == 'cuda'):
                        adv_loss = gan.adversarial_loss(x, reconstructed_x)
                        gan_loss = gan.loss_function(x, reconstructed_x)
                    val_adv_loss += adv_loss.item() * x.size(0)
                    val_gan_loss += gan_loss.item() * x.size(0)
        val_recon_loss /= val_size
        val_kl_loss /= val_size
        val_adv_loss /= val_size
        val_gan_loss /= val_size

        vae_scheduler.step(val_recon_loss)
        if use_gan:
            gan_scheduler.step(val_gan_loss)
        vae_current_lr = vae_optimizer.param_groups[0]['lr']
        gan_current_lr = gan_optimizer.param_groups[0]['lr'] if use_gan else 0.0

        print(f'Epoch [{epoch + 1}/{epochs}] | Train Recon: {train_recon_loss:.7f} | Train KL: {train_kl_loss:.7f} | '
              f'Val Recon: {val_recon_loss:.7f} | Val KL: {val_kl_loss:.7f}  | Beta: {beta:.7f}')
        print(f'Train Adv {train_adv_loss:.7f} | Train GAN: {train_gan_loss:.7f} | '
              f'Val Adv: {val_adv_loss:.7f} | Val GAN: {val_gan_loss:.7f} | '
              f'VAE LR: {vae_current_lr:.8f} | GAN LR: {gan_current_lr:.8f}')

        # log vae_loss
        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, train_recon_loss, train_kl_loss, train_adv_loss, train_gan_loss,
                             val_recon_loss, val_kl_loss, val_adv_loss, val_gan_loss, beta,
                             vae_current_lr, gan_current_lr])

        checkpoint_info = {'epoch': epoch + 1,
                           'random_state': random_state,
                           'vae_state_dict': vae.state_dict(),
                           'vae_optimizer_state_dict': vae_optimizer.state_dict(),
                           'vae_scheduler_state_dict': vae_scheduler.state_dict(),
                           'scaler_state_dict': scaler.state_dict(),
                           'use_gan': use_gan
                           }
        if use_gan:
            checkpoint_info.update({'gan_state_dict': gan.state_dict(),
                                    'gan_optimizer_state_dict': gan_optimizer.state_dict(),
                                    'gan_scheduler_state_dict': gan_scheduler.state_dict()})

        # Save latest model every epoch in case of crash.
        torch.save(checkpoint_info, checkpoint_path)

        # save best model
        if val_recon_loss < best_val_score:
            torch.save(checkpoint_info, best_model_path)
            best_val_score = val_recon_loss
            print('New best model found')

    with torch.no_grad():
        MAE, SSIM, PSNR = 0.0, 0.0, 0.0
        for data in tqdm(test_loader, desc='Validating', unit='batch'):
            x, _ = data
            x = x.to(device, non_blocking=True)
            with torch.amp.autocast('cuda', enabled=amp and device.type == 'cuda'):
                reconstructed_x, _, _ = vae(x)
            MAE += mae(x, reconstructed_x).item() * x.size(0)
            SSIM += ssim(x, reconstructed_x).item() * x.size(0)
            PSNR += psnr(x, reconstructed_x).item() * x.size(0)
        MAE /= test_size
        SSIM /= test_size
        PSNR /= test_size
    print(f'{100 * test_split}% test data with {test_size} instances')
    print(f'MAE = {MAE:.4f}, SSIM = {SSIM:.4f}, PSNR = {PSNR:.4F}')


def main():
    parser = argparse.ArgumentParser(description='Train Adversarial VAE model')
    # dataset parser
    parser.add_argument('--image_dir', type=str, required=True, help='folder containing labeled NIfTI volumes')
    parser.add_argument('--labels_csv', type=str, required=True, help='CSV containing filenames and labels')
    parser.add_argument('--filename_column', type=str, default='filename', help='CSV filename column')
    parser.add_argument('--label_column', type=str, default='label', help='CSV label column')
    parser.add_argument('--save_dir', type=str, required=True, help='dir for model saving')

    # VAE parser
    parser.add_argument('--vae_featuremap_size', type=int, default=32, help='VAE featuremap size')
    parser.add_argument('--vae_base_channel', type=int, default=256, help='VAE base channel')
    # GAN parser
    parser.add_argument('--disable_gan', action='store_true', help='train only the VAE without GAN/adversarial losses')
    parser.add_argument('--gan_patch_size', type=int, default=16, help='GAN featuremap size')
    parser.add_argument('--gan_base_channel', type=int, default=16, help='GAN base channel')
    parser.add_argument('--gan_weight_fn', type=str, default='weighted', help='GAN weight fn')
    parser.add_argument('--gamma', type=float, default=0.01, help='weight of adversarial loss')
    # train parser
    parser.add_argument('--vae_lr', type=float, default=1e-4, help='vae learning rate')
    parser.add_argument('--gan_lr', type=float, default=1e-4, help='gan learning rate')
    parser.add_argument('--epochs', type=int, default=500, help='epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--beta', type=float, default=1e-6, help='beta')
    parser.add_argument('--loss_criterion', type=str, default='MAE', help='loss_criterion')
    parser.add_argument('--random_state', type=int, default=42, help='random_state')
    parser.add_argument('--load_model_id', type=str, default='', help='load_model_id')
    parser.add_argument('--amp', action='store_true', help='use CUDA automatic mixed precision')
    parser.add_argument('--num_workers', type=int, default=0, help='DataLoader workers')
    parser.add_argument('--lr_scheduler_patience', type=int, default=20, help='ReduceLROnPlateau patience')
    parser.add_argument('--lr_scheduler_factor', type=float, default=0.5, help='ReduceLROnPlateau LR decay factor')

    args = parser.parse_args()
    transform = tio.Compose([
        tio.RandomFlip(axes=(0, 1)),
        tio.RandomAffine(
            scales=(0.9, 1.1),
            degrees=(0, 0, 0, 0, -30, 30),
            isotropic=True
        )
    ])
    dataset = CsvVolumeDataset(
        args.image_dir,
        args.labels_csv,
        filename_column=args.filename_column,
        label_column=args.label_column,
        transform=transform,
    )
    input_shape = (1, 128, 128, 128)
    vae = VAE(input_shape=input_shape, featuremap_size=args.vae_featuremap_size, base_channel=args.vae_base_channel,
              flatten_latent_dim=None, with_residual=True)
    gan = None if args.disable_gan else PatchGAN(input_shape, patch_size=args.gan_patch_size,
                                                 base_channel=args.gan_base_channel, with_residual=True,
                                                 weight_fn=args.gan_weight_fn)

    train(dataset, vae=vae, save_dir=args.save_dir, gan=gan, vae_lr=args.vae_lr, gan_lr=args.gan_lr,
          epochs=args.epochs, batch_size=args.batch_size, val_split=0.1, beta=args.beta, gamma=args.gamma,
          loss_criterion=args.loss_criterion, random_state=args.random_state, load_model_id=args.load_model_id,
          amp=args.amp, num_workers=args.num_workers,
          lr_scheduler_patience=args.lr_scheduler_patience,
          lr_scheduler_factor=args.lr_scheduler_factor)


if __name__ == '__main__':
    main()
