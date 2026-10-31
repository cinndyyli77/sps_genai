import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy


class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()

        self.layers = nn.Sequential(
            nn.Linear(32 * 32 * 3, 200),
            nn.ReLU(),
            nn.Linear(200, 150),
            nn.ReLU(),
            nn.Linear(150, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.layers(x)
        return x


class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class Critic(nn.Module):
    def __init__(self):        
        super(Critic, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(128)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)
        #self.drop2 = nn.Dropout(0.3)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.batchnorm3 = nn.BatchNorm2d(256)
        self.act3 = nn.LeakyReLU(0.2, inplace=True)
        #self.drop3 = nn.Dropout(0.3)

        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False)
        self.batchnorm4 = nn.BatchNorm2d(512)
        self.act4 = nn.LeakyReLU(0.2, inplace=True)
        #self.drop4 = nn.Dropout(0.3)

        self.conv5 = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.act2(x)
        #x = self.drop2(x)

        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = self.act3(x)
        #x = self.drop3(x)

        x = self.conv4(x)
        x = self.batchnorm4(x)
        x = self.act4(x)
        #x = self.drop4(x)

        x = self.conv5(x)
        x = self.flatten(x)

        return x

class Generator(nn.Module):
    def __init__(self, z_dim):
        super(Generator, self).__init__()
        self.z_dim = z_dim

        self.reshape = lambda x: x.view(x.size(0), z_dim, 1, 1)

        self.deconv1 = nn.ConvTranspose2d(z_dim, 512, kernel_size=4, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(512, momentum=0.9)
        self.act1 = nn.ReLU(True) # nn.LeakyReLU(0.2, inplace=True)

        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256, momentum=0.9)
        self.act2 = nn.ReLU(True) # nn.LeakyReLU(0.2, inplace=True)

        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128, momentum=0.9)
        self.act3 = nn.ReLU(True) # nn.LeakyReLU(0.2, inplace=True)

        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(64, momentum=0.9)
        self.act4 = nn.ReLU(True) # nn.LeakyReLU(0.2, inplace=True)

        self.deconv5 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.reshape(x)

        x = self.deconv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.deconv2(x)
        x = self.bn2(x)
        x = self.act2(x)

        x = self.deconv3(x)
        x = self.bn3(x)
        x = self.act3(x)

        x = self.deconv4(x)
        x = self.bn4(x)
        x = self.act4(x)

        x = self.deconv5(x)
        x = self.tanh(x)

        return x


class GAN(nn.Module):
    def __init__(self, z_dim=100):
        super().__init__()
        self.z_dim = z_dim
        self.generator = Generator(z_dim)
        self.critic = Critic()

    def forward(self, z):
        return self.generator(z)


def get_model(model_name):
    if model_name == "Diffusion":
        return DiffusionModel(
            UNet(image_size=64, num_channels=3),
            offset_cosine_diffusion_schedule
        )
    elif model_name == "GAN":
        return GAN()
    elif model_name == "CNN":
        return CNN()
    elif model_name in ("FCNN", "MLP"):
        return MLP()
    else:
        raise ValueError(f"Unknown model: {model_name}")


class SinusoidalEmbedding(nn.Module):
    def __init__(self, num_frequencies=16):
        super().__init__()
        self.num_frequencies = num_frequencies
        frequencies = torch.exp(
            torch.linspace(
                math.log(1.0),
                math.log(1000.0),
                num_frequencies
            )
        )
        self.register_buffer(
            "angular_speeds",
            2.0 * math.pi * frequencies.view(1, 1, 1, -1)
        )

    def forward(self, x):
        x = x.expand(-1, 1, 1, self.num_frequencies)
        sin_part = torch.sin(self.angular_speeds * x)
        cos_part = torch.cos(self.angular_speeds * x)
        return torch.cat([sin_part, cos_part], dim=-1)


def offset_cosine_diffusion_schedule(
    diffusion_times,
    min_signal_rate=0.02,
    max_signal_rate=0.95
):
    start_angle = torch.acos(
        torch.tensor(
            max_signal_rate,
            dtype=torch.float32,
            device=diffusion_times.device
        )
    )
    end_angle = torch.acos(
        torch.tensor(
            min_signal_rate,
            dtype=torch.float32,
            device=diffusion_times.device
        )
    )
    diffusion_angles = (
        start_angle
        + diffusion_times * (end_angle - start_angle)
    )
    signal_rates = torch.cos(diffusion_angles)
    noise_rates = torch.sin(diffusion_angles)
    return noise_rates, signal_rates


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1
        )
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1
        )

    def forward(self, x):
        residual = self.proj(x)
        x = self.conv1(x)
        x = x * torch.sigmoid(x)
        x = self.conv2(x)
        return x + residual


class DownBlock(nn.Module):
    def __init__(self, width, block_depth, in_channels):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(block_depth):
            self.blocks.append(ResidualBlock(in_channels, width))
            in_channels = width
        self.pool = nn.AvgPool2d(kernel_size=2)

    def forward(self, x, skips):
        for block in self.blocks:
            x = block(x)
            skips.append(x)
        return self.pool(x)


class UpBlock(nn.Module):
    def __init__(self, width, block_depth, in_channels):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(block_depth):
            self.blocks.append(
                ResidualBlock(in_channels + width, width)
            )
            in_channels = width

    def forward(self, x, skips):
        x = F.interpolate(
            x,
            scale_factor=2,
            mode="bilinear",
            align_corners=False
        )
        for block in self.blocks:
            skip = skips.pop()
            x = torch.cat([x, skip], dim=1)
            x = block(x)
        return x


class UNet(nn.Module):
    def __init__(self, image_size=64, num_channels=3):
        super().__init__()
        self.num_channels = num_channels
        self.image_size = image_size

        self.initial = nn.Conv2d(
            num_channels, 32, kernel_size=1
        )
        self.embedding = SinusoidalEmbedding(
            num_frequencies=16
        )

        self.down1 = DownBlock(
            width=32, block_depth=2, in_channels=64
        )
        self.down2 = DownBlock(
            width=64, block_depth=2, in_channels=32
        )
        self.down3 = DownBlock(
            width=96, block_depth=2, in_channels=64
        )

        self.mid1 = ResidualBlock(96, 128)
        self.mid2 = ResidualBlock(128, 128)

        self.up1 = UpBlock(
            width=96, block_depth=2, in_channels=128
        )
        self.up2 = UpBlock(
            width=64, block_depth=2, in_channels=96
        )
        self.up3 = UpBlock(
            width=32, block_depth=2, in_channels=64
        )

        self.final = nn.Conv2d(
            32, num_channels, kernel_size=1
        )
        nn.init.zeros_(self.final.weight)

    def forward(self, noisy_images, noise_variances):
        skips = []
        x = self.initial(noisy_images)

        noise_emb = self.embedding(noise_variances)
        noise_emb = noise_emb.permute(0, 3, 1, 2)
        noise_emb = F.interpolate(
            noise_emb,
            size=noisy_images.shape[-2:],
            mode="nearest"
        )
        x = torch.cat([x, noise_emb], dim=1)

        x = self.down1(x, skips)
        x = self.down2(x, skips)
        x = self.down3(x, skips)

        x = self.mid1(x)
        x = self.mid2(x)

        x = self.up1(x, skips)
        x = self.up2(x, skips)
        x = self.up3(x, skips)

        return self.final(x)


class DiffusionModel(nn.Module):
    def __init__(self, model, schedule_fn):
        super().__init__()
        self.network = model
        self.ema_network = copy.deepcopy(model)
        self.ema_network.requires_grad_(False)
        self.ema_network.eval()
        self.ema_decay = 0.8
        self.schedule_fn = schedule_fn
        self.normalizer_mean = 0.0
        self.normalizer_std = 1.0

    def set_normalizer(self, mean, std):
        self.normalizer_mean = mean
        self.normalizer_std = std

    def denormalize(self, images):
        return (
            images * self.normalizer_std
            + self.normalizer_mean
        ).clamp(0, 1)

    def forward(self, noisy_images, noise_variances):
        return self.network(noisy_images, noise_variances)

    def denoise(
        self, noisy_images, noise_rates, signal_rates,
        training=False
    ):
        if training:
            network = self.network
            network.train()
        else:
            network = self.ema_network
            network.eval()

        pred_noises = network(noisy_images, noise_rates ** 2)
        pred_images = (
            noisy_images - noise_rates * pred_noises
        ) / signal_rates
        return pred_noises, pred_images

    @torch.no_grad()
    def reverse_diffusion(self, initial_noise, diffusion_steps):
        current_images = initial_noise
        step_size = 1.0 / diffusion_steps

        for step in range(diffusion_steps):
            diffusion_times = torch.full(
                (initial_noise.size(0), 1, 1, 1),
                1.0 - step * step_size,
                device=initial_noise.device
            )
            noise_rates, signal_rates = self.schedule_fn(
                diffusion_times
            )
            pred_noises, pred_images = self.denoise(
                current_images, noise_rates, signal_rates
            )

            next_times = (diffusion_times - step_size).clamp(0, 1)
            next_noise_rates, next_signal_rates = self.schedule_fn(
                next_times
            )
            current_images = (
                next_signal_rates * pred_images
                + next_noise_rates * pred_noises
            )

        return pred_images

    @torch.no_grad()
    def generate(self, num_images, diffusion_steps=100):
        device = next(self.network.parameters()).device
        initial_noise = torch.randn(
            num_images,
            self.network.num_channels,
            self.network.image_size,
            self.network.image_size,
            device=device
        )
        images = self.reverse_diffusion(
            initial_noise, diffusion_steps
        )
        return self.denormalize(images)

    def train_step(self, images, optimizer, loss_fn):
        images = (
            images - self.normalizer_mean
        ) / self.normalizer_std
        noises = torch.randn_like(images)
        diffusion_times = torch.rand(
            images.size(0), 1, 1, 1,
            device=images.device
        )
        noise_rates, signal_rates = self.schedule_fn(
            diffusion_times
        )
        noisy_images = (
            signal_rates * images + noise_rates * noises
        )
        pred_noises, _ = self.denoise(
            noisy_images, noise_rates, signal_rates,
            training=True
        )
        loss = loss_fn(pred_noises, noises)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            for ema_param, param in zip(
                self.ema_network.parameters(),
                self.network.parameters()
            ):
                ema_param.mul_(self.ema_decay)
                ema_param.add_(
                    param, alpha=1.0 - self.ema_decay
                )

        return loss.item()