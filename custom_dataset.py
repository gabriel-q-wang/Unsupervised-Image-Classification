import torch
import numpy as np
import torch.nn as nn
from PIL import Image
from scipy import ndimage, misc
import matplotlib.pyplot as plt
import torchvision.transforms as TF
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToPILImage
from typing import Any, Callable, Optional, Tuple

# Need to upsample the image because otherwise it is too small for alexnet
height = 32*2
width = 32*2

brightness = (0.8, 1.2)
contrast = (0.8, 1.2)
saturation = (0.8, 1.2)
hue = (-0.1, 0.1)
color_jitter = TF.ColorJitter.get_params(
    brightness, contrast, saturation, hue)
perspective = TF.RandomPerspective(distortion_scale=0.1, p=0.5, interpolation=3, fill=0)

train_transformation = TF.Compose([TF.CenterCrop(30),
                                   TF.Resize((height,width)),
                                   TF.RandomHorizontalFlip(),
                                   color_jitter,
                                   TF.ToTensor()])

test_transformation = TF.Compose([TF.Resize((height,width)),
                                  TF.ToTensor()])

class Sobel(nn.Module):
    def __init__(self):
        super(Sobel, self).__init__()
        self.grayscale = nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0)
        self.grayscale.weight.data.fill_(1.0 / 3.0)
        self.grayscale.bias.data.zero_()

        self.sobel = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1)
        self.sobel.weight.data[0, 0].copy_(
            torch.FloatTensor([[1, 0, -1],
                            [2, 0, -2],
                            [1, 0, -1]])
        )
        self.sobel.weight.data[1, 0].copy_(
            torch.FloatTensor([[1, 2, 1],
                            [0, 0, 0],
                            [-1, -2, -1]])
        )
        self.sobel.bias.data.zero_()

    def forward(self, img):
        img = img.unsqueeze(0)
        img = self.grayscale(img)
        sobel_img = self.sobel(img)
        return sobel_img


class AugmentedCIFAR10(CIFAR10):
    def __init__(self, *args, **kwargs):
        super(AugmentedCIFAR10, self).__init__(*args, **kwargs)
        self.sobel = Sobel()

    def __getitem__(self, index: int, show_images=False) -> Tuple[Any, Any]:
        img, target = super().__getitem__(index)

        if self.train:
            img = train_transformation(img)
        else:
            img = test_transformation(img)

        return img, target


if __name__ == "__main__":
    ds = AugmentedCIFAR10('dataset', download=True)
    ds.__getitem__(24, show_images=True)