"""Train the Assignment 2 CNN on CIFAR-10 and save cnn_checkpoint.pth.

Usage:
    python train_cnn.py --epochs 10 --lr 0.001
"""
import argparse

import torch
import torch.nn as nn

from helper_lib import (
    CNN,
    get_data_loader,
    train_model,
    evaluate_model,
    save_checkpoint,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--output", default="./cnn_checkpoint.pth")
    args = parser.parse_args()

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    # CIFAR-10 images are resized to 64x64 inside get_data_loader
    train_loader = get_data_loader(args.data_dir, args.batch_size, train=True)
    test_loader = get_data_loader(args.data_dir, args.batch_size, train=False)

    model = CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    model = train_model(
        model, train_loader, criterion, optimizer, device,
        num_epochs=args.epochs,
    )
    evaluate_model(model, test_loader, device)

    save_checkpoint(model, optimizer, args.epochs, args.output)
    print(f"Saved checkpoint to {args.output}")


if __name__ == "__main__":
    main()
