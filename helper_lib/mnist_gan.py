import torch.nn as nn


class MNISTGenerator(nn.Module):
    """Generate 1×28×28 images from 100-dimensional noise."""

    def __init__(self):
        super().__init__()
        self.z_dim = 100
        self.fc = nn.Linear(self.z_dim, 128 * 7 * 7)

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(
                128, 64, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                64, 1, kernel_size=4, stride=2, padding=1
            ),
            nn.Tanh()
        )

    def forward(self, noise):
        x = self.fc(noise)
        x = x.reshape(noise.size(0), 128, 7, 7)
        return self.layers(x)


class MNISTDiscriminator(nn.Module):
    """Return the probability that each input image is real."""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(
                1, 64, kernel_size=4, stride=2, padding=1
            ),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(
                64, 128, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1),
            nn.Sigmoid()
        )

    def forward(self, images):
        return self.layers(images)


def initialize_weights(module):
    if isinstance(
        module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)
    ):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight, mean=1.0, std=0.02)
        nn.init.zeros_(module.bias)