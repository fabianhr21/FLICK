#!/bin/env python
#
# Unet architecture module
#

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

        self.reset_parameters(self.modules()) #self.modules returns an interable to the idfferent layers in the model class.

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
##             GENERADOR 2D              ##
###########################################
class Generator2D(nn.Module):
    def __init__(self, args):

        self._num_input_features=len(args.x_features)
        self._num_target_features=len(args.y_features)
        self._num_res_blocks=args.num_res_blocks

        super(Generator2D, self).__init__()
        # Initial Convolutional layer
        self.conv1 = nn.Conv2d(self._num_input_features, 64, kernel_size=3, stride=1, padding=1) # (Vaig canviar el kernel size a 3 en comptes de 9 )
        self.prelu1 = nn.PReLU(64)
        
        # Second Convolutional layer  (Aquesta la he afegit perque en el paper no redueixen dimensionalitat i nosaltres si ho fem)
        self.down1 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.prelu_down1 = nn.PReLU(128)

        # 32 Residual blocks 
        self.residual_blocks = nn.Sequential(*[ResidualBlock2D(128) for _ in range(self._num_res_blocks)])
        
        # Decoder: Upsampling + Bilineal Interpolation  (Aquesta part en el 3D Reconstruction sera bastant diferent, pero nomes per millorar la resolució i 
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.PReLU()
        )
        # Last Filters
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(64, self._num_target_features, kernel_size=3, stride=1, padding=1)

        self.reset_parameters(self.modules()) #self.modules returns an interable to the idfferent layers in the model class.

    def reset_parameters(self, m) -> None:
        
        for layer in m:
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_normal_(layer.weight)

    def forward(self, x):
        # Initial Convolutional layer
        x1 = self.conv1(x)
        x1 = self.prelu1(x1)

        # Second Convolutional layer Downsampling
        d1 = self.down1(x1)
        d1 = self.prelu_down1(d1)

        # Residual blocks
        r = self.residual_blocks(d1)
        
        # Upsampling
        u1 = self.up1(r)

        # Skip connection
        out = x1 + u1

        # Last Filters
        out = self.conv2(out)
        out = self.conv_out(out)
        return torch.sigmoid(out)

###########################################
##          DISCRIMINATOR 2D             ##
###########################################
class Discriminator2D(nn.Module):   # (Aquesta part també variará una mica respecte al 3D)
    def __init__(self, args):

        self._num_target_features=len(args.y_features)
        
        super(Discriminator2D, self).__init__()
        self.model = nn.Sequential(
            # 128x256 -> 64x128
            nn.Conv2d(self._num_target_features, 64, kernel_size=4, stride=2, padding=1),  # [B, 64, 64, 128]
            nn.LeakyReLU(0.2, inplace=True),
            # 64x128 -> 32x64
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),          # [B, 128, 32, 64]
            nn.LeakyReLU(0.2, inplace=True),
            # 32x64 -> 16x32
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),         # [B, 256, 16, 32]
            nn.LeakyReLU(0.2, inplace=True),
            # 16x32 -> 8x16
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),         # [B, 512, 8, 16]
            nn.LeakyReLU(0.2, inplace=True)
        )
        # (Aquí aplano la sortida per la capa fully-connected amb Linear ja que la funció Dense en Pytorch no existeix i Linear es la mes senblant)
        self.fc = nn.Linear(512 * 16 * 16, 1)  # 512*16*16 = 131072

        self.reset_parameters(self.modules()) #self.modules returns an interable to the idfferent layers in the model class.

    def reset_parameters(self, m) -> None:
        
        for layer in m:
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_normal_(layer.weight)

    def forward(self, x):
        out = self.model(x)             # Output: [B, 512, 8, 16]
        out = out.view(out.size(0), -1) # Output: [B, 65536]
        out = self.fc(out)              # Output: [B, 1]
        return torch.sigmoid(out)    
