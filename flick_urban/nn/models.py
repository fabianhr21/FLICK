"""
models.py — Neural network architectures for urban wind field prediction.

Classes
-------
ResidualBlock2D : Basic residual block (Conv → BN → PReLU → Conv + skip).
Generator2D     : ResNet-style generator (encoder → residual blocks → decoder).
                  Input:  (B, n_inputs, H, W)  — e.g. MASK, HEGT, WDST
                  Output: (B, n_targets, H, W) — e.g. U, V
Discriminator   : PatchGAN discriminator for adversarial training.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ResidualBlock2D(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock2D, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bnorm = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.reset_parameters(self.modules())

    def reset_parameters(self, m) -> None:
        for layer in m:
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_normal_(layer.weight)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bnorm(out)
        out = self.prelu(out)
        out = self.conv2(out)
        return out + residual


###########################################
##            Generator 2D               ##
###########################################
class Generator2D(nn.Module):
    def __init__(self, args):

        self._num_input_features = len(args.x_features)
        self._num_target_features = len(args.y_features)
        self._num_res_blocks = args.num_res_blocks

        super(Generator2D, self).__init__()
        # Initial convolutional layer — kernel_size=3
        self.conv1 = nn.Conv2d(self._num_input_features, 64, kernel_size=3, stride=1, padding=1)
        self.prelu1 = nn.PReLU(64)

        # Downsampling layer
        self.down1 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.prelu_down1 = nn.PReLU(128)

        # Residual blocks
        self.residual_blocks = nn.Sequential(*[ResidualBlock2D(128) for _ in range(self._num_res_blocks)])

        # Decoder: bilinear upsampling
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.PReLU()
        )
        # Output layers
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(64, self._num_target_features, kernel_size=3, stride=1, padding=1)

        self.reset_parameters(self.modules())

    def reset_parameters(self, m) -> None:
        for layer in m:
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_normal_(layer.weight)

    def forward(self, x):
        x1 = self.prelu1(self.conv1(x))
        d1 = self.prelu_down1(self.down1(x1))
        r = self.residual_blocks(d1)
        u1 = self.up1(r)
        out = self.conv2(x1 + u1)
        out = self.conv_out(out)
        return torch.sigmoid(out)


###########################################
##           Discriminator 2D            ##
###########################################
class Discriminator(nn.Module):
    """PatchGAN discriminator for adversarial training."""

    def __init__(self, args):
        super(Discriminator, self).__init__()

        self.args = args
        self._num_input_features = len(args.x_features)
        self._num_target_features = len(args.y_features)
        self._input_feature_xdim = args.input_xdim
        self._input_feature_ydim = args.input_ydim
        self._target_feature_xdim = args.target_xdim
        self._target_feature_ydim = args.target_ydim

        in_ch = self._num_target_features + self._num_input_features
        self.conv2D_input_64   = nn.Conv2d(in_ch, 64,  kernel_size=4, stride=2, padding=1, bias=False)
        self.conv2D_64_128     = nn.Conv2d(64,  128, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv2D_128_256    = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv2D_256_512    = nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1, bias=False)
        self.conv2D_512_output = nn.Conv2d(512, 1,   kernel_size=4, stride=1, padding=1, bias=False)

        self.BatchNorm_128 = torch.nn.BatchNorm2d(128, eps=1e-03, momentum=0.99)
        self.BatchNorm_256 = torch.nn.BatchNorm2d(256, eps=1e-03, momentum=0.99)
        self.BatchNorm_512 = torch.nn.BatchNorm2d(512, eps=1e-03, momentum=0.99)
        self.LeakyReLU = nn.LeakyReLU(0.2, True)

        self.reset_parameters(self.modules())

    def reset_parameters(self, m) -> None:
        for layer in m:
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, mean=0.0, std=0.02)

    def forward(self, x):
        x = self.LeakyReLU(self.conv2D_input_64(x))
        x = self.LeakyReLU(self.BatchNorm_128(self.conv2D_64_128(x)))
        x = self.LeakyReLU(self.BatchNorm_256(self.conv2D_128_256(x)))
        x = self.LeakyReLU(self.BatchNorm_512(self.conv2D_256_512(x)))
        return self.conv2D_512_output(x)
