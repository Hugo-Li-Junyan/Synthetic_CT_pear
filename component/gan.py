import torch.nn as nn
import math
import torch.nn.functional as F
import torch
from torch.nn.utils import spectral_norm
from vae import ResidualBlock3D


class PatchGAN(nn.Module):
    def __init__(self, x_shape, patch_size=32, base_channel=64, with_residual=True, weight_fn='weighted'):
        super(PatchGAN, self).__init__()
        # Initial conv layer
        self.layers = None
        self.weight_fn = weight_fn
        self.dim = x_shape[1]
        self.patch_size = patch_size
        self.base_channel = base_channel
        self.with_residual = with_residual
        self._construct_layers(x_shape, patch_size, base_channel, with_residual)

    def forward(self, x):
        x = self.layers(x)
        return x  # shape: (B, 1, D', H', W')

    def adversarial_loss(self, x, x_recon):
        weight = self.calculate_weight(x)
        gan_pred_fake = self(x_recon)
        #adv_loss = F.binary_cross_entropy_with_logits(gan_pred_fake, torch.full_like(gan_pred_fake, 0.95), weight=weight)
        adv_loss = -torch.mean(gan_pred_fake * weight)
        return adv_loss

    def loss_function(self, x, x_recon):
        weight = self.calculate_weight(x)

        gan_pred_real = self(x)
        gan_pred_fake = self(x_recon)

        #loss_real = F.binary_cross_entropy_with_logits(gan_pred_real, torch.full_like(gan_pred_real, 0.95), weight=weight)
        #loss_fake = F.binary_cross_entropy_with_logits(gan_pred_fake, torch.full_like(gan_pred_fake, 0.05), weight=weight)
        #gan_loss = (loss_real+loss_fake)/2
        real_loss = torch.mean(F.relu(1.0-gan_pred_real) * weight)
        fake_loss = torch.mean(F.relu(1.0+gan_pred_fake) * weight)
        gan_loss = 0.5 * (real_loss + fake_loss)
        #if self.training:
        #    lambda_gp = 10
        #    gp = self._gradient_penalty(x, x_recon)
        #    gan_loss = gan_loss + lambda_gp * gp
        return gan_loss

    def calculate_weight(self, x):
        kernel = self.patch_size
        with torch.no_grad():
            if self.weight_fn == 'max':
                weight = torch.clamp(F.max_pool3d(x, kernel_size=kernel, stride=kernel).detach(), max=1.0)
            elif self.weight_fn == 'range':
                weight = torch.clamp(range_pool3d(x, kernel_size=kernel, stride=kernel).detach(), max=1.0)
            elif self.weight_fn == 'weighted':
                weight = torch.clamp(weighted_pool3d(x, kernel_size=kernel, stride=kernel).detach(), max=1.0)
            else:
                raise ValueError('Other weight function not supported')
        return weight

    def _construct_layers(self, x_shape, patch_size, base_channel, with_residual):
        def is_power_of_two(n):
            return n > 0 and (n & (n - 1)) == 0

        assert isinstance(x_shape, (tuple, int)), "x_shape must be (C, D, H, W)"
        assert is_power_of_two(patch_size), "patch_size must be the power of 2"
        assert len(x_shape) == 4, "x_shape must be (C, D, H, W)"
        assert x_shape[1] == x_shape[2] == x_shape[3], "only D==H==W is supported for x_shape"
        assert is_power_of_two(x_shape[1]), "x_shape must be the power of 2"
        layers = []
        # decide how many feature_layers to include
        num_blocks = int(math.log2(patch_size))

        # first block from 1 channel to 32 channels
        first_block = [
            spectral_norm(nn.Conv3d(1, base_channel, kernel_size=3, stride=1, padding=1)),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        layers += first_block

        for i in range(num_blocks):
            block = self._downsampling_blocks(base_channel * 2 ** i, base_channel * 2**(i + 1), with_residual)
            layers += block

        last_block = [
            nn.Conv3d(base_channel * 2**num_blocks, 1, kernel_size=3, stride=1, padding=1)
        ]

        layers += last_block
        self.layers = nn.Sequential(*layers)


    @staticmethod
    def _downsampling_blocks(in_channel, out_channel, with_residual):
        block = [
            spectral_norm(nn.Conv3d(in_channel, out_channel, kernel_size=3, stride=2, padding=1)),
            # nn.InstanceNorm3d(out_channel),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        if with_residual:
           block.append(ResidualBlock3D(out_channel))
        return block

    def _gradient_penalty(self, real_images, fake_images):
        batch_size = real_images.size(0)
        epsilon = torch.rand(batch_size, 1, 1, 1, 1, device=real_images.device)
        epsilon = epsilon.expand_as(real_images)
        interpolated = (epsilon * real_images + (1 - epsilon) * fake_images).requires_grad_(True)

        interpolated_preds = self(interpolated).view(batch_size, -1).mean(1)
        grad_outputs = torch.ones_like(interpolated_preds)

        grads = torch.autograd.grad(
            outputs=interpolated_preds,
            inputs=interpolated,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        grads = grads.view(batch_size, -1)
        grad_norm = grads.norm(2, dim=1)

        # Penalty: (||grad|| - 1)^2
        gp = ((grad_norm - 1) ** 2).mean()
        return gp


def range_pool3d(input, kernel_size, stride=None, padding=0):
    if stride is None:
        stride = kernel_size
    # First, apply 3D Max Pooling
    max_pooled = F.max_pool3d(input, kernel_size, stride=stride, padding=padding)
    # Then, apply 3D Min Pooling (by negating input and max pooling)
    min_pooled = -F.max_pool3d(-input, kernel_size, stride=stride, padding=padding)
    # Range is simply max - min
    range_pooled = max_pooled - min_pooled
    return range_pooled


def weighted_pool3d(input, kernel_size, stride=None, padding=0):
    if stride is None:
        stride = kernel_size
    # First, apply 3D Max Pooling
    max_pooled = F.max_pool3d(input, kernel_size, stride=stride, padding=padding)
    # Then, apply 3D Min Pooling (by negating input and max pooling)
    min_pooled = -F.max_pool3d(-input, kernel_size, stride=stride, padding=padding)
    # Range is simply max - min
    weigted_pooled = max_pooled - 0.1 * min_pooled
    return weigted_pooled

if __name__ == '__main__':
    # Create dataset instance
    x = torch.randn((1,1,128,128,128))
    gan = PatchGAN(x_shape=(1,128,128,128), patch_size=16, base_channel=16, with_residual=True, weight_fn='max')
    y = gan(x)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


    print(count_parameters(gan))