import torch.nn as nn
import torch
import torch.nn.functional as F
import math


class LatentDiffusion(nn.Module):
    def __init__(self, input_dim, emb_dim, base_channel, timesteps=1000):
        super().__init__()
        self.input_dim = input_dim
        self.base_channel = base_channel
        self.timesteps = timesteps
        self.denoise_fn = None
        self._construct_layers(input_dim, emb_dim, base_channel)
        # Linear noise schedule
        self.register_buffer('betas', torch.linspace(1e-4, 0.02, timesteps))
        self.register_buffer('alphas', 1. - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod = self.alphas_cumprod[t].sqrt()
        sqrt_one_minus_alphas_cumprod = (1. - self.alphas_cumprod[t]).sqrt()
        x_noisy = self._smart_multiply(sqrt_alphas_cumprod, x_start) + self._smart_multiply(sqrt_one_minus_alphas_cumprod, noise)
        return x_noisy

    def p_losses(self, x_start, t):
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)
        t_input = t.unsqueeze(1).float() / self.timesteps  # Normalize timestep
        x_input = self._t_embedding(x_noisy, t_input)
        predicted_noise = self.denoise_fn(x_input)
        return F.mse_loss(predicted_noise, noise)

    def denoise(self, z_noisy, steps=50, save_steps:int=None):
        device = z_noisy.device

        # Get timestep schedule
        total_steps = self.timesteps
        step_indices = torch.linspace(0, total_steps - 1, steps, dtype=torch.long, device=device)
        steps_z = []
        for i in reversed(range(steps)):
            t = step_indices[i]
            t_int = t.long()
            t_input = t.unsqueeze(0).repeat(z_noisy.size(0)).unsqueeze(1) / self.timesteps  # normalized
            z_input = self._t_embedding(z_noisy, t_input)

            pred_noise = self.denoise_fn(z_input)
            # DDIM
            alpha_t = self.alphas_cumprod[t_int]
            alpha_prev = self.alphas_cumprod[step_indices[i - 1].long()] if i > 0 else torch.tensor(1.0, device=device)

            # DDIM update (deterministic version for eta=0)
            sqrt_alpha_t = alpha_t.sqrt()
            sqrt_one_minus_alpha_t = (1 - alpha_t).sqrt()

            pred_z0 = (z_noisy - sqrt_one_minus_alpha_t * pred_noise) / sqrt_alpha_t
            dir_xt = (1 - alpha_prev).sqrt() * pred_noise
            z_noisy = alpha_prev.sqrt() * pred_z0 + dir_xt
            if save_steps:
                if (i+1)%save_steps == 0:
                    steps_z.append(z_noisy)
        if save_steps:
            return steps_z
        else:
            return z_noisy

    @staticmethod
    def _smart_multiply(a, b):
        # Assume a.shape == (batch_size, 1)
        while a.dim() < b.dim():
            a = a.unsqueeze(-1)
        return a * b

    @staticmethod
    def _smart_minus(a, b):
        # Assume a.shape == (batch_size, 1)
        while a.dim() < b.dim():
            a = a.unsqueeze(-1)
        return a - b

    def _t_embedding(self, x, t):
        if isinstance(self.input_dim, int):
            return torch.cat([x, t], dim=1)
        else:
            return x, t

    def _construct_layers(self, input_dim, emb_dim, base_channel):
        def is_power_of_two(n):
            return n > 0 and (n & (n - 1)) == 0
        assert isinstance(input_dim, (int, tuple)), "input_dim must be a tuple (C, D, H, W) or int"
        if isinstance(input_dim, int):
            assert is_power_of_two(input_dim), "input_dim must be power of 2"
            self.denoise_fn = nn.Sequential(
            nn.Linear(input_dim + 1, base_channel),
            nn.ReLU(),
            nn.Linear(base_channel, base_channel),
            nn.ReLU(),
            nn.Linear(base_channel, input_dim),
        )

        elif isinstance(input_dim, tuple):
            assert len(input_dim) == 4, "input_dim must be a tuple (C, D, H, W)"
            assert input_dim[1] == input_dim[2] == input_dim[3], "only D==H==W is supported for input_dim"
            assert is_power_of_two(input_dim[1]), "input_dim must be power of 2"
            assert input_dim[1] >= 16, "input_dim must be larger than 16"
            self.denoise_fn = UNet3D(emb_dim = emb_dim, base_channel=base_channel)


# Sinusoidal timestep embedding like DDPM
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        emb = emb.to(device=timesteps.device)
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        emb = emb.view(emb.size(0), -1)
        return emb


class ResBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_ch)
        )
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            nn.SiLU(),
            nn.Conv3d(in_ch, out_ch, 3, padding=1)
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_ch),
            nn.SiLU(),
            nn.Conv3d(out_ch, out_ch, 3, padding=1)
        )
        self.skip = nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.block1(x)
        t = self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        h = h + t
        h = self.block2(h)
        return h + self.skip(x)


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.q = torch.nn.Conv3d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv3d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv3d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv3d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)


    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,d,h,w = q.shape
        q = q.reshape(b,c,h*w*d)
        q = q.permute(0,2,1)   # b,hwd,c
        k = k.reshape(b,c,h*w*d) # b,c,hwd
        w_ = torch.bmm(q,k)     # b,hwd,hwd    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b,c,h*w*d)
        w_ = w_.permute(0,2,1)   # b,hwd,hwd (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hwd (hwd of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b,c,d,h,w)

        h_ = self.proj_out(h_)

        return x+h_


# The full UNet3D with timestep conditioning
class UNet3D(nn.Module):
    def __init__(self, emb_dim, base_channel):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(emb_dim//4),
            nn.Linear(emb_dim//4, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim)
        )

        self.conv_in = torch.nn.Conv3d(1,base_channel,kernel_size=3,stride=1,padding=1)
        # Encoder
        self.enc1 = ResBlock3D(base_channel, base_channel*2, emb_dim)
        self.down1 = nn.Conv3d(base_channel*2, base_channel*2, 3,2,1)

        self.enc2 = ResBlock3D(base_channel*2, base_channel*4, emb_dim)
        self.down2 = nn.Conv3d(base_channel*4, base_channel*4, 3,2,1)

        self.enc3 = ResBlock3D(base_channel*4, base_channel*8, emb_dim)
        self.down3 = nn.Conv3d(base_channel*8, base_channel*8, 3,2,1)

        # Bottleneck
        self.mid_1 = ResBlock3D(base_channel*8, base_channel*8, emb_dim)
        self.mid_attn = AttnBlock(base_channel*8)
        self.mid_2 = ResBlock3D(base_channel*8, base_channel*8, emb_dim)

        # Decoder
        self.up3 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv3d(base_channel*8, base_channel*8, kernel_size=3, stride=1, padding=1))
        self.dec3 = ResBlock3D(base_channel*16, base_channel*4, emb_dim)

        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                 nn.Conv3d(base_channel * 4, base_channel * 4, kernel_size=3, stride=1, padding=1))
        self.dec2 = ResBlock3D(base_channel*8, base_channel*2, emb_dim)

        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                 nn.Conv3d(base_channel * 2, base_channel * 2, kernel_size=3, stride=1, padding=1))
        self.dec1 = ResBlock3D(base_channel*4, base_channel, emb_dim)

        self.norm_out = torch.nn.GroupNorm(num_groups=32, num_channels=base_channel, eps=1e-6, affine=True)
        self.nonlinear_out = nn.SiLU()
        self.conv_out = nn.Conv3d(base_channel, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x_input):
        x, t = x_input
        t_emb = self.time_mlp(t)  # shape: (B, emb_dim)
        x = self.conv_in(x)

        # Encoder
        enc1 = self.enc1(x, t_emb)
        x = self.down1(enc1)

        enc2 = self.enc2(x, t_emb)
        x = self.down2(enc2)

        enc3 = self.enc3(x, t_emb)
        x = self.down3(enc3)

        # Bottleneck
        x = self.mid_1(x, t_emb)
        x = self.mid_attn(x)
        x = self.mid_2(x, t_emb)

        # Decoder
        x = self.up3(x)
        x = torch.cat([x, enc3], dim=1)
        x = self.dec3(x, t_emb)

        x = self.up2(x)
        x = torch.cat([x, enc2], dim=1)
        x = self.dec2(x, t_emb)

        x = self.up1(x)
        x = torch.cat([x, enc1], dim=1)
        x = self.dec1(x, t_emb)

        x = self.norm_out(x)
        x = self.nonlinear_out(x)
        x = self.conv_out(x)

        return x



if __name__ == '__main__':
    # Create dataset instance
    x = torch.randn((1,1,128,128,128))
    vae = UNet3D(512, 128)


    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


    print(count_parameters(vae))