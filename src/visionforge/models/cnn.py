"""Convolutional Neural Network (CNN) model for image classification."""
import torch.nn as nn

# class CNN(nn.Module):
#     def __init__(self, num_classes=10):
#         super().__init__()
#         self.features = nn.Sequential(
#             # Block 1: Conv2d -> BatchNorm2d -> ReLU -> Pool
#             nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(num_features=16),  # Matches out_channels of Conv2d
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),

#             # Block 2
#             nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(num_features=32),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),

#             # Block 3
#             nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(num_features=64),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#         )

#         self.classifier = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(1024, 128, bias=False),
#             nn.BatchNorm1d(128),  # Use BatchNorm1d for 2D linear outputs
#             nn.ReLU(inplace=True),
#             nn.Linear(128, num_classes)
#         )

#     def forward(self, x):
#         x = self.features(x)
#         return self.classifier(x)


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