import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def get_data_loader(data_dir, batch_size=32, train=True):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=train,
        download=True,
        transform=transform
    )

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train
    )

    return data_loader
