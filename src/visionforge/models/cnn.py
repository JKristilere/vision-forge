"""Convolutional Neural Network (CNN) model for image classification."""
from torch import nn


class CNN(nn.Module):
    def __init__(self, num_classes=10, channels=(32, 64, 128), hidden=256, dropout=0.3):
        super().__init__()
        layers, in_ch = [], 3
        for out_ch in channels:
            layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            in_ch = out_ch
        self.features = nn.Sequential(*layers)
        # 32 / 2^len(channels) spatial size after pooling
        flat = channels[-1] * (32 // 2 ** len(channels)) ** 2
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, hidden, bias=False),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))