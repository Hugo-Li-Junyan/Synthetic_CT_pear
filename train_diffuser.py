import argparse
import json
import torch
import torchio as tio
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from component.vae import VAE
from component.dataset import MedicalImageDataset
import os
import csv
from component.diffuser import LatentDiffusion


def train(dataset, vae, diffuser, model_dir, lr, epochs, batch_size, val_split=0.1, early_stop_patience=50,
          random_state=42):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using', 'GPU' if torch.cuda.is_available() else 'CPU')
    vae = vae.to(device)
    diffuser = diffuser.to(device)

    optimizer = optim.Adam(diffuser.parameters(), lr=lr)
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    # load diffuser if exists
    start_epoch = 0
    diffuser_checkpoint_path = os.path.join(model_dir, "diffuser_checkpoint.pth")
    if os.path.exists(diffuser_checkpoint_path):
        checkpoint = torch.load(diffuser_checkpoint_path, map_location=device)
        diffuser.load_state_dict(checkpoint['diffuser_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        random_state = checkpoint['random_state']
        print(f"Loaded diffusion model from {diffuser_checkpoint_path}")

    # train validation separation
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(random_state)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    # Setup Logger
    hyperparameters = {
        'model': f"{'MLP' if isinstance(diffuser.input_dim, int) else 'UNet'}",
        'input dimensionality': diffuser.input_dim,
        'base channel': diffuser.base_channel,
        'learning rate': lr,
        'epochs': epochs,
        'batch_size': batch_size,
        'timesteps': diffuser.timesteps,
        'optimizer': 'Adam',
        'learning rate scheduler': 'Cosine annealing warm restarts'
    }
    hyperparameter_file = os.path.join(model_dir, 'diffuser_hyperparameter.json')
    with open(hyperparameter_file, "w") as f:
        json.dump(hyperparameters, f, indent=4)
    log_path = os.path.join(model_dir, "diffuser_log.csv")
    best_model_path = os.path.join(model_dir, "diffuser_best.pth")
    checkpoint_path = os.path.join(model_dir, 'diffuser_checkpoint.pth')

    if not os.path.exists(diffuser_checkpoint_path):
        with open(log_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Epoch", "Train Loss", "Validation Loss", "Learning rate"])

    # start training
    vae.eval()  # freeze
    for param in vae.parameters():
        param.requires_grad = False

    best_val_score = torch.inf
    early_stop_count = 0
    print('Training starts now')
    #std_first_batch = None
    for epoch in range(start_epoch, epochs + 1):
        train_loss = 0.0
        diffuser.train()
        for data in train_loader:
            x, _ = data
            x = x.to(device)
            with torch.no_grad():
                mu, logvar = vae.encode(x)
                z = vae.reparameterize(mu, logvar)
                #if not std_first_batch:
                    #std_first_batch = z.flatten().std()
                    #print(std_first_batch.item(), 'has been saved as std')
                #z = z/std_first_batch  # rescaling trick
            diff_t = torch.randint(0, diffuser.timesteps, (z.size(0),), device=device)

            optimizer.zero_grad()
            loss = diffuser.p_losses(z, diff_t)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= train_size

        # Validation phase
        val_loss = 0.0
        diffuser.eval()
        with torch.no_grad():
            for data in val_loader:
                x, _ = data
                x = x.to(device)
                mu, logvar = vae.encode(x)
                z = vae.reparameterize(mu, logvar)
                #z = z / std_first_batch  # rescaling trick
                diff_t = torch.randint(0, diffuser.timesteps, (z.size(0),), device=device)
                loss = diffuser.p_losses(z, diff_t)
                val_loss += loss.item()
        val_loss /= val_size

        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch [{epoch + 1}/{epochs}] | Train Loss: {train_loss} | Val Loss: {val_loss} | LR: {current_lr}")
        scheduler.step()

        # Log to CSV
        with open(log_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, train_loss, val_loss, current_lr])

        # Save latest model (in case of crash)
        if (epoch + 1) % 3 == 0:
            checkpoint_info = {'epoch': epoch + 1,
                               'random_state': random_state,
                               'diffuser_state_dict': diffuser.state_dict(),
                               'optimizer_state_dict': optimizer.state_dict(),
                               'scheduler_state_dict': scheduler.state_dict(),
                               #'std_first_batch': std_first_batch
                               }
            torch.save(checkpoint_info, checkpoint_path)

        # save best model
        if val_loss < best_val_score:
            best_info = {'epoch': epoch + 1,
                         'random_state': random_state,
                         'diffuser_state_dict': diffuser.state_dict(),
                         'optimizer_state_dict': optimizer.state_dict(),
                         'scheduler_state_dict': scheduler.state_dict(),
                         #'std_first_batch': std_first_batch
                         }
            torch.save(best_info, best_model_path)
            best_val_score = val_loss
            early_stop_count = 0
            print(f'New best model found')
        else:
            early_stop_count += 1
            if (early_stop_count + 1) % 10 == 0:
                print(f'Validation loss has not been reduced for {early_stop_count} epochs')

        if early_stop_count > early_stop_patience:
            print('Stop earlier')
            checkpoint_info = {'epoch': epoch + 1,
                               'random_state': random_state,
                               'diffuser_state_dict': diffuser.state_dict(),
                               'optimizer_state_dict': optimizer.state_dict(),
                               'scheduler_state_dict': scheduler.state_dict(),
                               #'std_first_batch': std_first_batch
                               }
            torch.save(checkpoint_info, checkpoint_path)
            break

    print(f"🏁 Training complete. Best model saved at: {best_model_path}")
    print(f"📊 Best validation score: {best_val_score}")
    print(f"📊 Training log saved at: {log_path}")


def main():
    parser = argparse.ArgumentParser(description="Train diffusion model")
    # dir parser
    parser.add_argument("--class1_dir", type=str, required=True, help="dir for healthy class")
    parser.add_argument("--class2_dir", type=str, required=True, help="dir for defective class")
    parser.add_argument("--save_dir", type=str, required=True, help="dir for model saving")
    parser.add_argument("--model_id", type=str, required=True, help="model_id")

    # VAE parser
    parser.add_argument("--vae_featuremap_size", type=int, default=32, help="VAE featuremap size")
    parser.add_argument("--vae_base_channel", type=int, default=256, help="VAE base channel")
    # Diffusion parser
    parser.add_argument("--diffuser_emb_dim", type=int, default=512, help="diffuser emb dim")
    parser.add_argument("--diffuser_base_channel", type=int, default=128, help="diffuser base channel")

    # hyperparameter
    parser.add_argument("--lr", type=float, default=1e-5, help="gan learning rate")
    parser.add_argument("--epochs", type=int, default=300, help="epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--random_state", type=int, default=42, help="random_state")


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
    model_dir = os.path.join(args.save_dir, args.model_id)
    vae = VAE(input_shape=(1, 128, 128, 128),
              featuremap_size=args.vae_featuremap_size,
              base_channel=args.vae_base_channel,
              flatten_latent_dim=None)
    vae_checkpoint = torch.load(os.path.join(model_dir, 'checkpoint.pth'))
    vae.load_state_dict(vae_checkpoint['vae_state_dict'])

    # load diffuser
    diffuser = LatentDiffusion(input_dim=(1, 32, 32, 32), emb_dim=args.diffuser_emb_dim, base_channel=args.diffuser_base_channel)
    train(dataset, vae, diffuser, model_dir=model_dir, lr=args.lr, epochs=args.epochs, batch_size=args.batch_size, val_split=0.1,
          early_stop_patience=150)


if __name__ == '__main__':
    main()