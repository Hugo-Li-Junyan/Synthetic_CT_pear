import json
import warnings
import numpy as np
import torchio as tio
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torch.optim.lr_scheduler as lr_scheduler
import time
import os
from component.dataset import MedicalImageDataset
import torch
import csv
from component.vae import VAE
from component.gan import PatchGAN
import math
import gc         # garbage collect library
import argparse


def train(dataset, vae, save_dir, gan, vae_lr=1e-4, gan_lr=1e-4, epochs=500, batch_size=8, val_split=0.1,
          load_model_id=None, beta=1e-6, loss_criterion='MAE', random_state=42):
    # device ready
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using', 'GPU' if torch.cuda.is_available() else 'CPU')
    vae = vae.to(device)
    gan = gan.to(device)

    # components ready
    vae_optimizer = optim.Adam(vae.parameters(), lr=vae_lr)
    gan_optimizer = optim.Adam(gan.parameters(), betas=(0.5, 0.999), lr=gan_lr)

    # load from checkpoint
    start_epoch = 0
    if load_model_id:
        checkpoint_path = os.path.join(save_dir, load_model_id, "checkpoint.pth")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location='cuda')
            random_state = checkpoint['random_state']
            vae.load_state_dict(checkpoint['vae_state_dict'])
            gan.load_state_dict(checkpoint['gan_state_dict'])
            vae_optimizer.load_state_dict(checkpoint['vae_optimizer_state_dict'])
            gan_optimizer.load_state_dict(checkpoint['gan_optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            print(f"Loaded model from {checkpoint_path}")
            del checkpoint
            gc.collect()
            torch.cuda.empty_cache()
        else:
            warnings.warn(f"checkpoint path {checkpoint_path} not exists, skip loading")

    # train validation separation
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(random_state)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    # training settings
    hyperparameters = {'model': 'VAE-GAN', 'vae_latent_space': vae.latent_space,
                       'vae_featuremap_size': vae.featuremap_size, 'vae_base_channel': vae.base_channel,
                       'vae_use_residual': vae.with_residual, 'vae learning rate': vae_lr, 'loss_fn': loss_criterion,
                       'epochs': epochs, 'batch_size': batch_size, 'beta': beta, 'vae_optimizer': 'Adam',
                       'gan_optimizer': 'Adam', 'gan learning rate': gan_lr, 'gamma': 0.01,
                       'gan_patch_size': gan.patch_size, 'gan_base_channel': gan.base_channel,
                       'gan_with_residual': gan.with_residual, 'gan_weight_function': gan.weight_fn}

    # for saving
    os.makedirs(save_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_dir = os.path.join(save_dir, timestamp)
    os.makedirs(model_dir, exist_ok=True)
    log_path = os.path.join(model_dir, "vae_log.csv")
    checkpoint_path = os.path.join(model_dir, "checkpoint.pth")
    best_model_path = os.path.join(model_dir, "best.pth")
    hyperparameter_path = os.path.join(model_dir, "vae_hyperparameter.json")
    with open(hyperparameter_path, "w") as f:
        json.dump(hyperparameters, f, indent=4)

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Train_Recon", "Train_KL", "Train_Adv", "Train_GAN",
                         "Val_Recon", "Val_KL", "Val_Adv", "Val_GAN", "Beta"])

    # start training
    print('Training starts now')
    best_val_score = torch.inf
    early_stop_count = 0
    for epoch in range(start_epoch, epochs+1):
        # train
        vae.train()
        train_recon_loss, train_kl_loss = 0.0, 0.0
        train_adv_loss, train_gan_loss = 0.0, 0.0
        avg_gamma = 0.0
        for data in train_loader:
            x, _ = data
            x = x.to(device)

            reconstructed_x, z_mean, z_logvar = vae(x)

            # train gan
            gan.train()
            for param in gan.parameters():
                param.requires_grad = True
            gan_loss = gan.loss_function(x, reconstructed_x.detach())
            gan_optimizer.zero_grad()
            gan_loss.backward()
            torch.nn.utils.clip_grad_norm_(gan.parameters(), 1.0)
            gan_optimizer.step()

            # train vae
            gan.eval()
            for param in gan.parameters():
                param.requires_grad = False
            recon_loss, kl_loss = vae.loss_function(reconstructed_x, x, z_mean, z_logvar, beta=beta,
                                                    criterion=loss_criterion)
            vae_loss = recon_loss + kl_loss
            adv_loss = gan.adversarial_loss(x, reconstructed_x)
            vae_loss = vae_loss + adv_loss * 0.01
            vae_optimizer.zero_grad()
            vae_loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
            vae_optimizer.step()

            train_recon_loss += recon_loss.item() * x.size(0)
            train_kl_loss += kl_loss.item() * x.size(0)

            train_adv_loss += adv_loss.item() * x.size(0)
            train_gan_loss += gan_loss.item() * x.size(0)
        train_recon_loss /= train_size
        train_kl_loss /= train_size
        train_adv_loss /= train_size
        train_gan_loss /= train_size

        # validation
        vae.eval()
        gan.eval()
        val_recon_loss, val_kl_loss = 0.0, 0.0
        val_adv_loss, val_gan_loss = 0.0, 0.0

        with torch.no_grad():
            for data in val_loader:
                x, _ = data
                x = x.to(device)
                reconstructed_x, z_mean, z_logvar = vae(x)
                recon_loss, kl_loss = vae.loss_function(reconstructed_x, x, z_mean, z_logvar, beta=beta, criterion=loss_criterion)
                adv_loss = gan.adversarial_loss(x, reconstructed_x)
                gan_loss = gan.loss_function(x, reconstructed_x)

                val_recon_loss += recon_loss.item() * x.size(0)
                val_kl_loss += kl_loss.item() * x.size(0)
                val_adv_loss += adv_loss.item() * x.size(0)
                val_gan_loss += gan_loss.item() * x.size(0)
        val_recon_loss /= val_size
        val_kl_loss /= val_size
        val_adv_loss /= val_size
        val_gan_loss /= val_size


        print(f"Epoch [{epoch + 1}/{epochs}] | Train Recon: {train_recon_loss:.7f} | Train KL: {train_kl_loss:.7f} | "
              f"Val Recon: {val_recon_loss:.7f} | Val KL: {val_kl_loss:.7f}  | Beta: {beta:.7f}")
        print(f"Train Adv {train_adv_loss:.7f} | Train GAN: {train_gan_loss:.7f} | Val Adv: {val_adv_loss:.7f} | Val GAN: {val_gan_loss:.7f}")


        # log vae_loss
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, train_recon_loss, train_kl_loss, train_adv_loss, train_gan_loss, val_recon_loss,
                             val_kl_loss, val_adv_loss, val_gan_loss, beta])

        # Save latest model (in case of crash)
        if (epoch+1) % 3 == 0:
            checkpoint_info = {'epoch': epoch+1,
                               'random_state': random_state,
                               'vae_state_dict': vae.state_dict(),
                               'gan_state_dict': gan.state_dict(),
                               'vae_optimizer_state_dict': vae_optimizer.state_dict(),
                               'gan_optimizer_state_dict': gan_optimizer.state_dict()
                               }
            torch.save(checkpoint_info, checkpoint_path)
        # save best model
        if val_recon_loss < best_val_score:
            checkpoint_info = {'epoch': epoch+1,
                               'random_state': random_state,
                               'vae_state_dict': vae.state_dict(),
                               'gan_state_dict': gan.state_dict(),
                               'vae_optimizer_state_dict': vae_optimizer.state_dict(),
                               'gan_optimizer_state_dict': gan_optimizer.state_dict()
                               }
            torch.save(checkpoint_info, best_model_path)
            best_val_score = val_recon_loss
            early_stop_count = 0
            print(f'New best model found')
        else:
            early_stop_count += 1
            if (early_stop_count + 1) % 10 == 0:
                print(f'Validation loss has not been reduced for {early_stop_count} epochs')


def main():
    parser = argparse.ArgumentParser(description="Train Adversarial VAE model")
    # dir parser
    parser.add_argument("--class1_dir", type=str, required=True, help="dir for healthy class")
    parser.add_argument("--class2_dir", type=str, required=True, help="dir for defective class")
    parser.add_argument("--save_dir", type=str, required=True, help="dir for model saving")

    # VAE parser
    parser.add_argument("--vae_featuremap_size", type=int, default=32, help="VAE featuremap size")
    parser.add_argument("--vae_base_channel", type=int, default=256, help="VAE base channel")
    # GAN parser
    parser.add_argument("--gan_patch_size", type=int, default=16, help="GAN featuremap size")
    parser.add_argument("--gan_base_channel", type=int, default=16, help="GAN base channel")
    parser.add_argument("--gan_weight_fn", type=str, default='weighted', help="GAN weight fn")
    # train parser
    parser.add_argument("--vae_lr", type=float, default=1e-4, help="vae learning rate")
    parser.add_argument("--gan_lr", type=float, default=1e-4, help="gan learning rate")
    parser.add_argument("--epochs", type=int, default=500, help="epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")
    parser.add_argument("--beta", type=float, default=1e-6, help="beta")
    parser.add_argument("--loss_criterion", type=str, default='MAE', help="loss_criterion")
    parser.add_argument("--random_state", type=int, default=42, help="random_state")
    parser.add_argument("--load_model_id", type=str, default='', help="load_model_id")

    args = parser.parse_args()
    transform = tio.Compose([
        tio.RandomFlip(axes=(0, 1)),
        tio.RandomAffine(
            scales=(0.9, 1.1),
            degrees=(0, 0, 0, 0, -30, 30),
            isotropic=True
        )
    ])
    dataset = MedicalImageDataset(args.class1_dir, args.class2_dir, transform=transform)
    input_shape = (1, 128, 128, 128)
    vae = VAE(input_shape=input_shape, featuremap_size=args.vae_featuremap_size, base_channel=args.vae_base_channel,
              flatten_latent_dim=None, with_residual=True)
    gan = PatchGAN(input_shape, patch_size=args.gan_patch_size, base_channel=args.gan_base_channel,
                   with_residual=True, weight_fn=args.gan_weight_fn)

    train(dataset, vae=vae, save_dir=args.save_dir, gan=gan, vae_lr=args.vae_lr, gan_lr=args.gan_lr, epochs=args.epochs,
          batch_size=args.batch_size, val_split=0.1, beta=args.beta, loss_criterion=args.loss_criterion, random_state=args.random_state, load_model_id=args.load_model_id)


if __name__ == '__main__':
    main()