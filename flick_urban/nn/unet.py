"""
unet.py — UNet-style encoder-decoder for urban wind field prediction.

UNet_wind architecture (pix2pix-inspired):
  - Encoder: 5× Conv2d with stride-2 downsampling
  - Decoder: 5× ConvTranspose2d with skip connections from encoder
  - Input:  (B, n_inputs, H, W)  — MASK, HEGT, WDST
  - Output: (B, n_targets, H, W) — U, V velocity components
  - Weights initialised with normal(mean=0, std=0.02)
  - BatchNorm (eps=1e-3, momentum=0.99), LeakyReLU(0.3), Dropout2d(0.5)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class UNet_wind(nn.Module):

    def __init__(self, args):
        """Build UNet_wind from an args namespace.

        Required args attributes
        ------------------------
        x_features     : list of input field names (e.g. ['MASK', 'HEGT', 'WDST'])
        y_features     : list of target field names (e.g. ['U', 'V'])
        input_xdim     : input spatial width
        input_ydim     : input spatial height
        target_xdim    : output spatial width
        target_ydim    : output spatial height
        verbose        : int ≥ 0 — print progress if > 0
        """
        super(UNet_wind, self).__init__()
        self.args = args
        self._input_feature_keys  = args.x_features
        self._target_feature_keys = args.y_features
        self._num_input_features  = len(args.x_features)
        self._num_target_features = len(args.y_features)
        self._input_feature_xdim  = args.input_xdim
        self._input_feature_ydim  = args.input_ydim
        self._target_feature_xdim = args.target_xdim
        self._target_feature_ydim = args.target_ydim

        if self.args.verbose > 0:
            print("[WIND NN Model] Creating UNet_wind model")

        # Encoder
        self.conv2D_input_64 = torch.nn.Conv2d(self._num_input_features, 64,  kernel_size=4, stride=2, padding=1, bias=False)
        self.conv2D_64_128   = torch.nn.Conv2d(64,  128, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv2D_128_256  = torch.nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv2D_256_512  = torch.nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv2D_512_512  = torch.nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1, bias=False)

        # Decoder
        self.conv2DTrans_512_512   = torch.nn.ConvTranspose2d(512,  512, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv2DTrans_1024_512  = torch.nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv2DTrans_1024_256  = torch.nn.ConvTranspose2d(1024, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv2DTrans_512_128   = torch.nn.ConvTranspose2d(512,  128, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv2DTrans_256_64    = torch.nn.ConvTranspose2d(256,  64,  kernel_size=4, stride=2, padding=1, bias=False)
        self.conv2DTrans_128_output = torch.nn.ConvTranspose2d(128, self._num_target_features, kernel_size=4, stride=2, padding=1, bias=False)

        # Activations and regularisation
        self.LeakyReLU = torch.nn.LeakyReLU(0.3, False)
        self.ReLU      = torch.nn.ReLU()
        self.Sigmoid   = torch.nn.Sigmoid()
        self.Dropout   = torch.nn.Dropout2d(0.5)

        # Batch normalisation
        self.BatchNorm_128 = torch.nn.BatchNorm2d(128, eps=1e-03, momentum=0.99)
        self.BatchNorm_256 = torch.nn.BatchNorm2d(256, eps=1e-03, momentum=0.99)
        self.BatchNorm_512 = torch.nn.BatchNorm2d(512, eps=1e-03, momentum=0.99)

        self.reset_parameters(self.modules())

    def reset_parameters(self, m) -> None:
        """Initialise Conv2d and ConvTranspose2d weights with normal(0, 0.02)."""
        for layer in m:
            if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(layer.weight, mean=0.0, std=0.02)

    def forward(self, x):
        """Forward pass with skip connections (U-Net style)."""
        skip = []

        # Encoder
        x = self.LeakyReLU(self.conv2D_input_64(x));  skip.append(x)
        x = self.LeakyReLU(self.BatchNorm_128(self.conv2D_64_128(x)));  skip.append(x)
        x = self.LeakyReLU(self.BatchNorm_256(self.conv2D_128_256(x))); skip.append(x)
        x = self.LeakyReLU(self.BatchNorm_512(self.conv2D_256_512(x))); skip.append(x)
        x = self.LeakyReLU(self.BatchNorm_512(self.conv2D_512_512(x))); skip.append(x)
        x = self.LeakyReLU(self.BatchNorm_512(self.conv2D_512_512(x))); skip.append(x)
        x = self.LeakyReLU(self.BatchNorm_512(self.conv2D_512_512(x))); skip.append(x)
        x = self.LeakyReLU(self.conv2D_512_512(x))  # bottleneck

        # Decoder with skip connections and dropout on first 3 layers
        x = self.ReLU(self.Dropout(self.conv2DTrans_512_512(x)));  x = torch.cat((x, skip[6]), dim=1)
        x = self.ReLU(self.Dropout(self.conv2DTrans_1024_512(x))); x = torch.cat((x, skip[5]), dim=1)
        x = self.ReLU(self.Dropout(self.conv2DTrans_1024_512(x))); x = torch.cat((x, skip[4]), dim=1)
        x = self.ReLU(self.conv2DTrans_1024_512(x));               x = torch.cat((x, skip[3]), dim=1)
        x = self.ReLU(self.conv2DTrans_1024_256(x));               x = torch.cat((x, skip[2]), dim=1)
        x = self.ReLU(self.conv2DTrans_512_128(x));                x = torch.cat((x, skip[1]), dim=1)
        x = self.ReLU(self.conv2DTrans_256_64(x));                 x = torch.cat((x, skip[0]), dim=1)
        x = self.Sigmoid(self.conv2DTrans_128_output(x))

        return x
