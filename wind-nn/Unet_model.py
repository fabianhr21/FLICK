#!/bin/env python
#
# Unet architecture module
#

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class UNet_wind(nn.Module):
    
    def __init__(self,args):

        """
        Specifies the network layers, not required to be in order, and saves model parameters passed in `args`.
        The first layer input will have dimensions specified by args.x_dim, and the last layer output dimension
        should be specified by args.y_dim. Everything in-between can have h_dims, ie: the number of hidden features
        or convolutional kernels in the model.
        ToDo:
         - Implementar la inicialització dels pesos de la xarxa amb una dsitribució normal de parametres m=0 i sigma=0.02.  això es fa a la reset function.
         cada tipus de layer s'inicialitza diferent (conv2D,batchnorm etc..) Veure com esta inicialitzat amb keras i reproduir https://www.tensorflow.org/tutorials/generative/pix2pix
         - Acabar d'ajustar els parameteres del batch norm layer a veure si realment són els que toca del pix2pix.
         - El padding a la versió original en keras que em va passar el jaime estva com a 'same'. Aquesta opció en pytorch és incompatible amb un stride >1. A la versió
         pix2pix amb pytorch estava posat un padding de 1 per tant he deixat aquest valor. 
         -implementar la loss function del jaime que és 0.5mse+0.5mae. No és la original del pix2pix que fa servir cross entropy. Veure si amb aquesta segona millora
         -s'ha de normalitzar l'input i l'output! ARA MATEIX HI HA UN SCALING FET PER PROVAR LO DEL LOSS REDUCTION PERÒ ESTA FET A SACO AL LLEGIR EL DATASET!!
         -crear dataset real. Potser es pot començar per relacionar pressió i velocitat?
         -FALTA INTRODUIR ELS SKIPS DE LA U-NET. VEURE TUTORIAL PIX2PIX COM FER-HO!!!
        """


        super(UNet_wind, self).__init__()
        self.args = args
        self._input_feature_keys=args.x_features
        self._target_feature_keys=args.y_features
        self._num_input_features=len(args.x_features)
        self._num_target_features=len(args.y_features)
        self._input_feature_xdim=args.input_xdim
        self._input_feature_ydim=args.input_ydim
        self._target_feature_xdim=args.target_xdim
        self._target_feature_ydim=args.target_ydim

        if self.args.verbose > 0: print("[WIND NN Model] Creating Unet_wind model")

        #torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)

        self.conv2D_input_64=torch.nn.Conv2d(in_channels=self._num_input_features, out_channels=64, kernel_size=4,stride=2,padding=1,bias=False)
        self.conv2D_64_128=torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4,stride=2,padding=1,bias=False)
        self.conv2D_128_256=torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4,stride=2,padding=1,bias=False)
        self.conv2D_256_512=torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4,stride=2,padding=1,bias=False)
        self.conv2D_512_512=torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4,stride=2,padding=1,bias=False)

        #self.conv2D_kk=torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4,stride=1,padding=1,bias=False)
        #torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', device=None, dtype=None)

        self.conv2DTrans_512_512=torch.nn.ConvTranspose2d(in_channels=512,out_channels=512,kernel_size=4,stride=2,padding=1,bias=False)
        self.conv2DTrans_1024_512=torch.nn.ConvTranspose2d(in_channels=1024,out_channels=512,kernel_size=4,stride=2,padding=1,bias=False)
        #self.conv2DTrans_512_256=torch.nn.ConvTranspose2d(in_channels=512,out_channels=256,kernel_size=4,stride=2,padding=1,bias=False)
        self.conv2DTrans_1024_256=torch.nn.ConvTranspose2d(in_channels=1024,out_channels=256,kernel_size=4,stride=2,padding=1,bias=False)
        #self.conv2DTrans_256_128=torch.nn.ConvTranspose2d(in_channels=256,out_channels=128,kernel_size=4,stride=2,padding=1,bias=False)
        self.conv2DTrans_512_128=torch.nn.ConvTranspose2d(in_channels=512,out_channels=128,kernel_size=4,stride=2,padding=1,bias=False)
        #self.conv2DTrans_128_64=torch.nn.ConvTranspose2d(in_channels=128,out_channels=64,kernel_size=4,stride=2,padding=1,bias=False)
        self.conv2DTrans_256_64=torch.nn.ConvTranspose2d(in_channels=256,out_channels=64,kernel_size=4,stride=2,padding=1,bias=False)
        #self.conv2DTrans_64_output=torch.nn.ConvTranspose2d(in_channels=64,out_channels=output_channels,kernel_size=4,stride=2,padding=1,bias=False)
        self.conv2DTrans_128_output=torch.nn.ConvTranspose2d(in_channels=128,out_channels=self._num_target_features,kernel_size=4,stride=2,padding=1,bias=False)

        self.LeakyReLU=torch.nn.LeakyReLU(0.3,False) #keras implementation?
        #self.LeakyReLU=torch.nn.LeakyReLU(0.2,False) #pytorch implementation
        self.ReLU=torch.nn.ReLU()
        #self.Tanh = torch.nn.Tanh()
        self.Sigmoid=torch.nn.Sigmoid()

        self.Dropout=torch.nn.Dropout2d(0.5)

        #self.BatchNorm=torch.nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None)
        self.BatchNorm_128=torch.nn.BatchNorm2d(128, eps=1e-03, momentum=0.99, affine=True, track_running_stats=True, device=None, dtype=None)
        self.BatchNorm_256=torch.nn.BatchNorm2d(256, eps=1e-03, momentum=0.99, affine=True, track_running_stats=True, device=None, dtype=None)
        self.BatchNorm_512=torch.nn.BatchNorm2d(512, eps=1e-03, momentum=0.99, affine=True, track_running_stats=True, device=None, dtype=None)

        self.reset_parameters(self.modules()) #self.modules returns an interable to the idfferent layers in the model class.


    def reset_parameters(self, m) -> None:
        
        #print("INITIALIZING WEIGHTS")
        #el pix2pix fa servir el random normal initialization:
        #initializer = tf.random_normal_initializer(0., 0.02)

        for layer in m:
            #print("model layers=",type(layer))
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight,mean=0.0,std=0.02)
                #print("initializing conv2d")
            elif isinstance(layer, nn.ConvTranspose2d):
                nn.init.normal_(layer.weight,mean=0.0,std=0.02)
                #print("initializing convTrans2d")
                #nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            #elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            #    nn.init.constant_(m.weight, 1)
            #    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass of the model.
        Note that the first layer has the number of input features as channel numbers.
        """

        skip_cons=[]
        #downsample
        x = self.conv2D_input_64(x)
        x = self.LeakyReLU(x)
        skip_cons.append(x)
        x = self.conv2D_64_128(x)
        x = self.LeakyReLU(self.BatchNorm_128(x))
        skip_cons.append(x)
        x = self.conv2D_128_256(x)
        x = self.LeakyReLU(self.BatchNorm_256(x))
        skip_cons.append(x)
        x = self.conv2D_256_512(x)
        x = self.LeakyReLU(self.BatchNorm_512(x))
        skip_cons.append(x)
        x = self.conv2D_512_512(x)
        x = self.LeakyReLU(self.BatchNorm_512(x))
        skip_cons.append(x)
        x = self.conv2D_512_512(x)
        x = self.LeakyReLU(self.BatchNorm_512(x))
        skip_cons.append(x)
        x = self.conv2D_512_512(x)
        x = self.LeakyReLU(self.BatchNorm_512(x))
        skip_cons.append(x)
        x = self.conv2D_512_512(x)
        #x = self.LeakyReLU(self.BatchNorm_512(x))
        x = self.LeakyReLU(x)
        #BOTTLENECK
        #upsample
        x = self.conv2DTrans_512_512(x)
        x = self.ReLU(self.Dropout(x))
        x=torch.cat((x,skip_cons[6]),axis=1)
        
        x = self.conv2DTrans_1024_512(x)
        x = self.ReLU(self.Dropout(x))
        x=torch.cat((x,skip_cons[5]),axis=1)
        
        x = self.conv2DTrans_1024_512(x)
        x = self.ReLU(self.Dropout(x))
        x=torch.cat((x,skip_cons[4]),axis=1)

        x = self.conv2DTrans_1024_512(x)
        x = self.ReLU(x)
        x=torch.cat((x,skip_cons[3]),axis=1)

        x = self.conv2DTrans_1024_256(x)
        x = self.ReLU(x)
        x=torch.cat((x,skip_cons[2]),axis=1)

        x = self.conv2DTrans_512_128(x)
        x = self.ReLU(x)
        x=torch.cat((x,skip_cons[1]),axis=1)

        x = self.conv2DTrans_256_64(x)
        x = self.ReLU(x)
        x=torch.cat((x,skip_cons[0]),axis=1)

        x = self.conv2DTrans_128_output(x)
        #x = self.ReLU(x)
        #x = self.Tanh(x)
        x = self.Sigmoid(x)

        return x


class UNet_wind_old(torch.nn.Module):

    def __init__(self, args):

        super(UNet_wind, self).__init__()
        self.args = args
        self._input_feature_keys=args.x_features
        self._target_feature_keys=args.y_features
        self._num_input_features=len(args.x_features)
        self._num_target_features=len(args.y_features)
        self._input_feature_xdim=args.input_xdim
        self._input_feature_ydim=args.input_ydim
        self._target_feature_xdim=args.target_xdim
        self._target_feature_ydim=args.target_ydim

        if self.args.verbose > 0: print('\nCreating Unet_wind model')

        #torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)

        self.conv2D_input_64=torch.nn.Conv2d(in_channels=self._num_input_features, out_channels=64, kernel_size=4,stride=2,padding=1,bias=False)
        self.conv2D_64_128=torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4,stride=2,padding=1,bias=False)
        self.conv2D_128_256=torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4,stride=2,padding=1,bias=False)
        self.conv2D_256_512=torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4,stride=2,padding=1,bias=False)
        self.conv2D_512_512=torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4,stride=2,padding=1,bias=False)

        #self.conv2D_kk=torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4,stride=1,padding=1,bias=False)
        #torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', device=None, dtype=None)

        self.conv2DTrans_512_512=torch.nn.ConvTranspose2d(in_channels=512,out_channels=512,kernel_size=4,stride=2,padding=1,bias=False)
        self.conv2DTrans_512_256=torch.nn.ConvTranspose2d(in_channels=512,out_channels=256,kernel_size=4,stride=2,padding=1,bias=False)
        self.conv2DTrans_256_128=torch.nn.ConvTranspose2d(in_channels=256,out_channels=128,kernel_size=4,stride=2,padding=1,bias=False)
        self.conv2DTrans_128_64=torch.nn.ConvTranspose2d(in_channels=128,out_channels=64,kernel_size=4,stride=2,padding=1,bias=False)
        self.conv2DTrans_64_output=torch.nn.ConvTranspose2d(in_channels=64,out_channels=self._num_target_features,kernel_size=4,stride=2,padding=1,bias=False)

        self.LeakyReLU=torch.nn.LeakyReLU(0.3,False) #keras implementation?
        #self.LeakyReLU=torch.nn.LeakyReLU(0.2,False) #pytorch implementation
        self.ReLU=torch.nn.ReLU()

        self.Dropout=torch.nn.Dropout2d(0.5)

        #self.BatchNorm=torch.nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None)
        self.BatchNorm_128=torch.nn.BatchNorm2d(128, eps=1e-03, momentum=0.99, affine=True, track_running_stats=True, device=None, dtype=None)
        self.BatchNorm_256=torch.nn.BatchNorm2d(256, eps=1e-03, momentum=0.99, affine=True, track_running_stats=True, device=None, dtype=None)
        self.BatchNorm_512=torch.nn.BatchNorm2d(512, eps=1e-03, momentum=0.99, affine=True, track_running_stats=True, device=None, dtype=None)

        self.reset_parameters(self.modules()) #self.modules returns an interable to the idfferent layers in the model class.

        '''
        #torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)

        self.conv2D_input_64=torch.nn.Conv2d(in_channels=self._num_input_features, out_channels=64, kernel_size=4,stride=2,padding=1,bias=False)
        self.conv2D_64_128=torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4,stride=2,padding=1,bias=False)
        #self.conv2D_128_256=torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4,stride=2,padding=1,bias=False)
        #self.conv2D_256_512=torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4,stride=2,padding=1,bias=False)
        #self.conv2D_512_512=torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4,stride=2,padding=1,bias=False)

        #self.conv2D_kk=torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4,stride=1,padding=1,bias=False)
        #torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', device=None, dtype=None)

        #self.conv2DTrans_512_512=torch.nn.ConvTranspose2d(in_channels=512,out_channels=512,kernel_size=4,stride=2,padding=1,bias=False)
        #self.conv2DTrans_512_256=torch.nn.ConvTranspose2d(in_channels=512,out_channels=256,kernel_size=4,stride=2,padding=1,bias=False)
        #self.conv2DTrans_256_128=torch.nn.ConvTranspose2d(in_channels=256,out_channels=128,kernel_size=4,stride=2,padding=1,bias=False)
        self.conv2DTrans_128_64=torch.nn.ConvTranspose2d(in_channels=128,out_channels=64,kernel_size=4,stride=2,padding=1,bias=False)
        self.conv2DTrans_64_output=torch.nn.ConvTranspose2d(in_channels=64,out_channels=self._num_target_features,kernel_size=4,stride=2,padding=1,bias=False)

        self.LeakyReLU=torch.nn.LeakyReLU(0.3,False) #keras implementation?
        #self.LeakyReLU=torch.nn.LeakyReLU(0.2,False) #pytorch implementation
        self.ReLU=torch.nn.ReLU()

        #self.Dropout=torch.nn.Dropout2d(0.5)

        #self.BatchNorm=torch.nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None)
        #self.BatchNorm_128=torch.nn.BatchNorm2d(128, eps=1e-03, momentum=0.99, affine=True, track_running_stats=True, device=None, dtype=None)
        #self.BatchNorm_256=torch.nn.BatchNorm2d(256, eps=1e-03, momentum=0.99, affine=True, track_running_stats=True, device=None, dtype=None)
        #self.BatchNorm_512=torch.nn.BatchNorm2d(512, eps=1e-03, momentum=0.99, affine=True, track_running_stats=True, device=None, dtype=None)

        self.reset_parameters(self.modules()) #self.modules returns an interable to the idfferent layers in the model class.
        '''

    '''
    def reset_parameters(self):
        """
        Initializes/resets network trainable parameters.
        """
        if self.args.verbose > 0: print('\nResetting module parameters')
        for name, module in self.named_children():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()
    '''

    '''
    def reset_parameters(self, m) -> None:

        #el pix2pix fa servir el random normal initialization:
        #initializer = tf.random_normal_initializer(0., 0.02)

        for layer in m:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    '''

    def reset_parameters(self, m) -> None:
        #print("INITIALIZING WEIGHTS")

        #el pix2pix fa servir el random normal initialization:
        #initializer = tf.random_normal_initializer(0., 0.02)

        for layer in m:
            #print("model layers=",type(layer))
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight,mean=0.0,std=0.02)
                #print("initializing conv2d")
            elif isinstance(layer, nn.ConvTranspose2d):
                nn.init.normal_(layer.weight,mean=0.0,std=0.02)
                #print("initializing convTrans2d")
                #nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            #elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            #    nn.init.constant_(m.weight, 1)
            #    nn.init.constant_(m.bias, 0)

    '''
    def forward(self, x):
        """
        Forward pass of the model.
        Note that the first layer has the number of input features as channel numbers.
        """
        #downsample
        x = self.conv2D_input_64(x)
        x = self.LeakyReLU(x)
        x = self.conv2D_64_128(x)
        #x = self.LeakyReLU(self.BatchNorm_128(x))
        #x = self.conv2D_128_256(x)
        #x = self.LeakyReLU(self.BatchNorm_256(x))
        #x = self.conv2D_256_512(x)
        #x = self.LeakyReLU(self.BatchNorm_512(x))
        #x = self.conv2D_512_512(x)
        #x = self.LeakyReLU(self.BatchNorm_512(x))
        #x = self.conv2D_512_512(x)
        #x = self.LeakyReLU(self.BatchNorm_512(x))
        #x = self.conv2D_512_512(x)
        #x = self.LeakyReLU(self.BatchNorm_512(x))
        #x = self.conv2D_512_512(x)
        #x = self.LeakyReLU(self.BatchNorm_512(x))
        #BOTTLENECK
        #upsample
        #x = self.conv2DTrans_512_512(x)
        #x = self.ReLU(self.Dropout(x))
        #x = self.conv2DTrans_512_512(x)
        #x = self.ReLU(self.Dropout(x))
        #x = self.conv2DTrans_512_512(x)
        #x = self.ReLU(self.Dropout(x))
        #x = self.conv2DTrans_512_512(x)
        #x = self.ReLU(x)
        #x = self.conv2DTrans_512_256(x)
        #x = self.ReLU(x)
        #x = self.conv2DTrans_256_128(x)
        #x = self.ReLU(x)
        x = self.conv2DTrans_128_64(x)
        x = self.ReLU(x)
        x = self.conv2DTrans_64_output(x)
        x = self.ReLU(x)

        return x
    
    ''' 

    def forward(self, x):
        """
        Forward pass of the model.
        Note that the first layer has the number of input features as channel numbers.
        """
        #downsample
        x = self.conv2D_input_64(x)
        x = self.LeakyReLU(x)
        x = self.conv2D_64_128(x)
        x = self.LeakyReLU(self.BatchNorm_128(x))
        x = self.conv2D_128_256(x)
        x = self.LeakyReLU(self.BatchNorm_256(x))
        x = self.conv2D_256_512(x)
        x = self.LeakyReLU(self.BatchNorm_512(x))
        x = self.conv2D_512_512(x)
        x = self.LeakyReLU(self.BatchNorm_512(x))
        x = self.conv2D_512_512(x)
        x = self.LeakyReLU(self.BatchNorm_512(x))
        x = self.conv2D_512_512(x)
        x = self.LeakyReLU(self.BatchNorm_512(x))
        x = self.conv2D_512_512(x)
        x = self.LeakyReLU(self.BatchNorm_512(x))
        #BOTTLENECK
        #upsample
        x = self.conv2DTrans_512_512(x)
        x = self.ReLU(self.Dropout(x))
        x = self.conv2DTrans_512_512(x)
        x = self.ReLU(self.Dropout(x))
        x = self.conv2DTrans_512_512(x)
        x = self.ReLU(self.Dropout(x))
        x = self.conv2DTrans_512_512(x)
        x = self.ReLU(x)
        x = self.conv2DTrans_512_256(x)
        x = self.ReLU(x)
        x = self.conv2DTrans_256_128(x)
        x = self.ReLU(x)
        x = self.conv2DTrans_128_64(x)
        x = self.ReLU(x)
        x = self.conv2DTrans_64_output(x)
        x = self.ReLU(x)

        return x
