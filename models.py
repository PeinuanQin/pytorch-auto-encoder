"""
 @file: models.py
 @Time    : 2023/1/11
 @Author  : Peinuan qin
 """
from torch import nn


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # (b, 1, 28, 28)
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 在2x2空间下采样
        )
        # (b, 16, 14, 14)
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # (b, 32, 7, 7)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        # (b, 32, 7, 7)
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.Upsample((14, 14))
        )

        # (b, 16, 14, 14)
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=1,
                kernel_size=3,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.Upsample((28, 28))
        )
        # (b, 1, 28, 28)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class EncoderDecoder(nn.Module):
    """
    combine encoder and decoder
    """
    def __init__(self):
        super(EncoderDecoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x