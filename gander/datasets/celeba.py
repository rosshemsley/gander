import torch
from torchvision.transforms import (
    Compose,
    Normalize,
    ToTensor,
    Resize,
    RandomHorizontalFlip,
)
from torchvision.datasets import ImageFolder

# Normalize data to [-1, 1]
MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]

# MEAN = [0, 0, 0]


class CelebA(ImageFolder):
    def __init__(self, root_dir: str):
        transform = Compose(
            [
                ToTensor(),
                Normalize(mean=MEAN, std=STD),
                Resize((128, 128)),
                RandomHorizontalFlip(),
            ]
        )
        super(CelebA, self).__init__(root=root_dir, transform=transform)


def denormalize(x: torch.Tensor) -> torch.Tensor:
    return Normalize(
        mean=[-MEAN[0] / STD[0], -MEAN[1] / STD[1], -MEAN[2] / STD[2]],
        std=[1 / STD[0], 1 / STD[1], 1 / STD[2]],
    )(x)
