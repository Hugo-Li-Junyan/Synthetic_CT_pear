import math
import torch.nn as nn
import torch


class ResidualBlock3D(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock3D, self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.block(x))


class Encoder(nn.Module):
    def __init__(self, input_shape, featuremap_size, base_channel, flatten_latent_dim, with_residual):
        super(Encoder, self).__init__()
        self.feature_layers = None
        self.mu_head = None
        self.logvar_head = None
        self._construct_layers(input_shape, featuremap_size, base_channel, flatten_latent_dim, with_residual)

    def forward(self, x):
        x = self.feature_layers(x)
        mu = self.mu_head(x)
        logvar = self.logvar_head(x)
        return mu, logvar

    def _construct_layers(self, input_shape, featuremap_size, base_channel, flatten_latent_dim, with_residual):
        def is_power_of_two(n):
            return n > 0 and (n & (n - 1)) == 0

        assert isinstance(input_shape, tuple), "input_shape must be (C, D, H, W)"
        assert len(input_shape) == 4, "input_shape must be (C, D, H, W)"
        assert input_shape[1] == input_shape[2] == input_shape[3], "only D==H==W is supported for input_shape"
        assert is_power_of_two(input_shape[1]), "input_shape must be the power of 2"
        assert is_power_of_two(featuremap_size), "featuremap_size must be the power of 2"

        feature_layers = []
        # decide how many feature_layers to include
        down_factor = input_shape[1] / featuremap_size
        num_blocks = int(math.log2(down_factor))

        # first block from 1 channel to 32 channels
        first_block = [
            nn.Conv3d(1, base_channel, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        feature_layers += first_block

        for i in range(num_blocks):
            block = self._downsampling_blocks(base_channel * 2**i, base_channel * 2**(i+1), with_residual)
            feature_layers += block

        self.feature_layers = nn.Sequential(*feature_layers)

        final_channel = base_channel * 2**num_blocks
        if flatten_latent_dim:
            self.mu_head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(final_channel * featuremap_size**3, flatten_latent_dim)
            )
            self.logvar_head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(final_channel * featuremap_size**3, flatten_latent_dim)
            )
        else:
            self.mu_head = nn.Conv3d(final_channel,1,kernel_size=3, stride=1, padding=1)
            self.logvar_head = nn.Conv3d(final_channel, 1, kernel_size=3, stride=1, padding=1)

    @staticmethod
    def _downsampling_blocks(in_channel, out_channel, with_residual):
        block = [
            nn.Conv3d(in_channel, out_channel, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm3d(out_channel),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        if with_residual:
            block.append(ResidualBlock3D(out_channel))
        return block


class Decoder(nn.Module):
    def __init__(self, output_shape, featuremap_size, base_channel, flatten_latent_dim, with_residual):
        super(Decoder, self).__init__()
        self.latent_to_feature = None
        self.feature_layers = None
        self.reconstruction_layer = None
        self._construct_layers(output_shape, featuremap_size, base_channel, flatten_latent_dim, with_residual)

    def forward(self, z):
        z = self.latent_to_feature(z)
        for block in self.feature_layers:
            z = block(z)
        z = self.reconstruction_layer(z)
        return z

    def _construct_layers(self, output_shape, featuremap_size, base_channel, flatten_latent_dim, with_residual):
        def is_power_of_two(n):
            return n > 0 and (n & (n - 1)) == 0
        assert isinstance(output_shape, tuple), "output_size must be a tuple (C, D, H, W)"
        assert len(output_shape) == 4, "output_shape must be a tuple (C, D, H, W)"
        assert output_shape[1] == output_shape[2] == output_shape[3], "only D==H==W is supported for output_shape"
        assert is_power_of_two(output_shape[1]), "output_shape must be power of 2"
        assert is_power_of_two(featuremap_size), "featuremap_size must be power of 2"

        feature_layers = []
        up_factor = output_shape[1] / featuremap_size
        num_blocks = int(math.log2(up_factor))
        first_channel = base_channel * 2**num_blocks
        if flatten_latent_dim:
            self.latent_to_feature = nn.Sequential(
                nn.Linear(flatten_latent_dim, first_channel * featuremap_size**3),
                nn.Unflatten(1, (first_channel, featuremap_size, featuremap_size, featuremap_size))
            )
        else:
            self.latent_to_feature = nn.Conv3d(1, first_channel, kernel_size=3, stride=1,padding=1)

        for i in range(num_blocks)[::-1]:
            block = self._construct_blocks(base_channel * 2**(i+1), int(base_channel * 2**i), with_residual)
            feature_layers += block

        self.reconstruction_layer = nn.Sequential(
            nn.Conv3d(base_channel, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        self.feature_layers = nn.Sequential(*feature_layers)

    @staticmethod
    def _construct_blocks(in_channel, out_channel, with_residual):
        block = [
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners = True),
            nn.Conv3d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(out_channel),
            nn.ReLU()
        ]
        if with_residual:
            block.append(ResidualBlock3D(out_channel))
        return block

class VAE(nn.Module):
    def __init__(self, input_shape, featuremap_size:int=16, base_channel:int=32, flatten_latent_dim=None, with_residual=True):
        """
        Create a VAE with 1-d or 3-d latent space shape
        :param input_shape: (channel, Depth, Height, Width)
        :param featuremap_size: the 1-d size of the last feature map of Encoder and first feature map of Decoder.
         Must be the power of 2.
        :param flatten_latent_dim: if None, use a 3-d latent space (featuremap_size, featuremap_size, featuremap_size);
        if int, specify the dimensionality of the latent space.
        """
        super(VAE, self).__init__()
        self.input_shape = input_shape
        self.featuremap_size = featuremap_size
        self.base_channel = base_channel
        self.with_residual = with_residual
        if flatten_latent_dim:
            self.latent_space = flatten_latent_dim
        else:
            self.latent_space = (1, featuremap_size, featuremap_size, featuremap_size)
        self.encoder = Encoder(input_shape,featuremap_size, base_channel, flatten_latent_dim, with_residual)
        self.decoder = Decoder(input_shape,featuremap_size, base_channel, flatten_latent_dim, with_residual)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(mu)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

    @staticmethod
    def loss_function(recon_x, x, mu, logvar, beta, criterion):
        # Mean Squared Error (MSE) Loss
        #weight = (x.detach() + 0.1).clamp(max=1.0)
        if criterion == 'MSE':
            recon_loss = ((recon_x - x).pow(2)).mean()
        elif criterion == 'MAE':
            recon_loss = ((recon_x - x).abs()).mean()
        else:
            raise ValueError('Other loss function not supported')
        # Combined reconstruction loss (Weighted sum of MSE & SSIM)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss, beta * kl_loss

if __name__ == '__main__':
    # Create dataset instance
    x = torch.randn((1,1,128,128,128))
    vae = VAE(input_shape=(1,128,128,128), featuremap_size=32, base_channel=256, flatten_latent_dim=None, with_residual=True)


    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


    print(count_parameters(vae))