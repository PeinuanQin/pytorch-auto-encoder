"""
 @file: models.py
 @Time    : 2023/1/11
 @Author  : Peinuan qin
 """

import torch
from torch import nn
from torch.distributions.normal import Normal


"""
Auto encoder for image denoise
"""
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




"""
Variance Auto encoder for generation task
"""

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = V_Encoder()
        self.decoder = V_Decoder()

    def forward(self, x):
        # get the distribution from encoder for encoder loss calculation
        distribution = self.encoder(x)
        # sample from the encoder distribution
        z = distribution.rsample()
        # use this sample to decoder and optimizer the re-construction loss
        output = self.decoder(z)
        return distribution, output


class V_Encoder(nn.Module):
    """
    VAE encoder
    """
    def __init__(self):
        super(V_Encoder, self).__init__()
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
        # assume the feature are constructed an 8-dim tensor
        # and this tensor ~ normal distribution
        self.mean_linear = nn.Linear(32*7*7, 8)
        self.var_linear = nn.Linear(32*7*7, 8)

    def forward(self, x):
        b, c, h, w = x.size()
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(b, -1)
        mean = self.mean_linear(x)
        log_var = self.var_linear(x)
        # reparameterize
        var = torch.exp(log_var)
        # return a distribution
        return Normal(mean, var)

class V_Decoder(nn.Module):
    def __init__(self):
        super(V_Decoder, self).__init__()
        # (b, 32, 7, 7)
        self.linear = nn.Linear(8, 32 * 7 * 7)
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=32,
                out_channels=16,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
        )

        # (b, 16, 14, 14)
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=16,
                out_channels=1,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.Sigmoid()
        )

    def forward(self, x):
        # (b, f)
        b, f = x.size()
        x = self.linear(x)
        # (b, 32* 7* 7)
        x = x.view(b, 32, 7, 7)
        # (b, 32, 7, 7)
        x = self.conv1(x)
        # (b, 16, 14, 14)
        x = self.conv2(x)
        # (b, 1, 28, 28)
        return x