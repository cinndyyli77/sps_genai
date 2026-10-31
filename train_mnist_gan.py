import argparse
import csv
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from helper_lib.mnist_gan import (
    MNISTGenerator,
    MNISTDiscriminator,
    initialize_weights,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.0002)
    args = parser.parse_args()

    torch.manual_seed(42)
    root = Path(__file__).resolve().parent
    output_dir = root / "outputs" / "mnist_gan"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}", flush=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    dataset = datasets.MNIST(
        root=root / "data",
        train=True,
        download=True,
        transform=transform,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    generator = MNISTGenerator().to(device)
    discriminator = MNISTDiscriminator().to(device)
    generator.apply(initialize_weights)
    discriminator.apply(initialize_weights)

    criterion = nn.BCELoss()
    optimizer_g = torch.optim.Adam(
        generator.parameters(),
        lr=args.lr,
        betas=(0.5, 0.999),
    )
    optimizer_d = torch.optim.Adam(
        discriminator.parameters(),
        lr=args.lr,
        betas=(0.5, 0.999),
    )
    fixed_noise = torch.randn(64, 100, device=device)

    with (output_dir / "losses.csv").open(
        "w", newline=""
    ) as log_file:
        writer = csv.writer(log_file)
        writer.writerow([
            "epoch", "discriminator_loss", "generator_loss"
        ])

        for epoch in range(args.epochs):
            generator.train()
            discriminator.train()
            total_d = 0.0
            total_g = 0.0
            count = 0

            for step, (real_images, _) in enumerate(
                loader, start=1
            ):
                real_images = real_images.to(device)
                batch_size = real_images.size(0)
                real_labels = torch.ones(
                    batch_size, 1, device=device
                )
                fake_labels = torch.zeros(
                    batch_size, 1, device=device
                )

                # Train the discriminator.
                discriminator.requires_grad_(True)
                optimizer_d.zero_grad(set_to_none=True)

                noise = torch.randn(
                    batch_size, 100, device=device
                )
                fake_images = generator(noise)

                loss_real = criterion(
                    discriminator(real_images), real_labels
                )
                loss_fake = criterion(
                    discriminator(fake_images.detach()),
                    fake_labels,
                )
                loss_d = loss_real + loss_fake
                loss_d.backward()
                optimizer_d.step()

                # Train the generator.
                discriminator.requires_grad_(False)
                optimizer_g.zero_grad(set_to_none=True)

                loss_g = criterion(
                    discriminator(fake_images), real_labels
                )
                loss_g.backward()
                optimizer_g.step()
                discriminator.requires_grad_(True)

                total_d += loss_d.item() * batch_size
                total_g += loss_g.item() * batch_size
                count += batch_size

                if step % 100 == 0:
                    print(
                        f"Epoch {epoch + 1}/{args.epochs} "
                        f"Batch {step}/{len(loader)} "
                        f"D: {loss_d.item():.4f} "
                        f"G: {loss_g.item():.4f}",
                        flush=True,
                    )

            avg_d = total_d / count
            avg_g = total_g / count
            writer.writerow([epoch + 1, avg_d, avg_g])
            log_file.flush()

            generator.eval()
            with torch.no_grad():
                samples = generator(fixed_noise).cpu()

            save_image(
                (samples + 1) / 2,
                output_dir / f"epoch_{epoch + 1:03d}.png",
                nrow=8,
            )

            torch.save(
                {
                    "epoch": epoch + 1,
                    "z_dim": 100,
                    "generator_state_dict": {
                        key: value.detach().cpu()
                        for key, value
                        in generator.state_dict().items()
                    },
                    "discriminator_state_dict": {
                        key: value.detach().cpu()
                        for key, value
                        in discriminator.state_dict().items()
                    },
                },
                root / "mnist_gan_checkpoint.pth",
            )

            print(
                f"Epoch {epoch + 1} complete: "
                f"D={avg_d:.4f}, G={avg_g:.4f}",
                flush=True,
            )


if __name__ == "__main__":
    main()