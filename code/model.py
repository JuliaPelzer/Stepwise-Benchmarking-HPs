import torch.nn as nn
from torch import Tensor, cat

class UNetDummy(nn.Module):
    def __init__(self, in_channels:int, out_channels:int):
        super().__init__()
        depth = 3
        kernel_size = 3
        features = 16
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        for _ in range(depth):
            self.encoders.append(UNetDummy._block(in_channels, features, kernel_size))
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = features
            features *= 2
        self.encoders.append(UNetDummy._block(in_channels, features, kernel_size))
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for _ in range(depth):
            self.upconvs.append(nn.ConvTranspose2d(features, features//2, kernel_size=2, stride=2))
            self.decoders.append(UNetDummy._block(features, features//2, kernel_size))
            features = features // 2

        self.last_conv = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        encodings = []
        for encoder, pool in zip(self.encoders, self.pools):
            x = encoder(x)
            encodings.append(x)
            x = pool(x)
        x = self.encoders[-1](x)

        for upconv, decoder, encoding in zip(self.upconvs, self.decoders, reversed(encodings)):
            x = upconv(x)
            x = cat((x, encoding), dim=1)
            x = decoder(x)

        return self.last_conv(x)
    
    @staticmethod
    def _conv(in_channels, out_channels, kernel_size):
        return nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding="same",
                    bias=True,
                )
    
    @staticmethod
    def _block(in_channels, features, kernel_size): 
        return nn.Sequential(
            UNetDummy._conv(in_channels, features, kernel_size),
            nn.ReLU(inplace=True),
            UNetDummy._conv(features, features, kernel_size),
            nn.BatchNorm2d(num_features=features),
            nn.ReLU(inplace=True),
            UNetDummy._conv(features, features, kernel_size),
            nn.ReLU(inplace=True),
        )
    
class Step1(nn.Module):
    pass

class Step2(nn.Module):
    pass

class Step3(nn.Module):
    pass