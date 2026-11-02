"""Feature-only torchvision-compatible VGG16-BN layout; no downloads or FC allocation."""
from torch import nn


def make_features():
    layers, channels = [], 3
    for width in (64,64,'M',128,128,'M',256,256,256,'M',512,512,512,'M',512,512,512,'M'):
        if width == 'M':
            layers.append(nn.MaxPool2d(2, 2))
        else:
            layers.extend([nn.Conv2d(channels, width, 3, padding=1),
                           nn.BatchNorm2d(width), nn.ReLU(inplace=True)])
            channels = width
    return nn.Sequential(*layers)
