#!/bin/env python
#
# model inference script

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch import nn
import numpy as np
import argparse
import string
import matplotlib.pyplot as plt
import os, h5py,glob
# from Models import UNet_wind
from Models import Generator2D 
# from Models import ResidualBlock2D,Discriminator2D
import sys
sys.path.append('../../pre-process/')
import STL2GeoTool_loop


BASE_FOLDER='./'


DATASET_BASE_PATH='../../pre-process/output/'
OUTPUT_PATH = '../output/'
DATA_SAMPLE_BASENAME='campusnord_1280'
MODEL_BASENAME='generator'
MODEL_LOADING_PATH='./'
INPUT_FEAT=['MASK','HEGT'] #having a MASK distinguishing solid and fluid regions at the first position is mandatory.
TARGET_FEAT=['U','V']
EXTRA_FEAT=[]
INPUT_XDIM=1280
INPUT_YDIM=1280
TARGET_XDIM=256
TARGET_YDIM=256
SPACING_X=1.0
SPACING_Y=1.0
SCALING_X=120. #whole dataset input max value
SCALING_Y=16.0 #whole dataset output max value


def plot_field(field, image_name, mask=None, vmin_cost=0.0, vmax_cost=0.0):
    if torch.cuda.is_available(): 
        plot_matrix = torch.clone(field)
        if plot_matrix.requires_grad:
            plot_matrix = plot_matrix.detach()
        plot_matrix = plot_matrix.cpu()

        if mask is not None:
            mask = mask.cpu()
            plot_matrix = np.ma.masked_where(mask < 1.0, plot_matrix.numpy())
        else:
            plot_matrix = plot_matrix.numpy()

    else:
        # Aquí field es un tensor, conviértelo a numpy
        if field.requires_grad:
            field = field.detach()
        plot_matrix = field.cpu().numpy()

        if mask is not None:
            if isinstance(mask, torch.Tensor):
                mask = mask.cpu().numpy()
            plot_matrix = np.ma.masked_where(mask < 1.0, plot_matrix)

    # Plotting
    if vmin_cost == 0.0 and vmax_cost == 0.0:
        plt.imshow(plot_matrix, cmap='Spectral')
    else:
        plt.imshow(plot_matrix, cmap='Spectral', vmin=vmin_cost, vmax=vmax_cost)

    plt.colorbar()
    plt.savefig(image_name)
    plt.clf()

def load_model(model,args):

    model_path=f"{args.model_loading_path}{args.model_basename}.pt"

    if not os.path.exists(model_path):
        print(f"The {model_path} file does not exist. Check file path and name.")
        exit(0)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print(f"Loading model weights from {model_path}")
    checkpoint = torch.load(model_path,map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model=model.float()
    model.eval()
    return model

def load_input_sample(args,file_path):
    #if not os.path.exists(f"{args.dataset_path}{args.data_sample_basename}-{idx}.h5"):
    #    print(f"The file {args.dataset_path}{args.data_sample_basename}-{idx}.h5 does not exist")
    #    exit(0)

    #file_path=f"{args.dataset_path}{args.data_sample_basename}-{idx}.h5"
    print(f"Opening file {file_path}")

    data=h5py.File(file_path)        

    X_features=np.empty((len(args.x_features),args.input_xdim,args.input_ydim),dtype=float)

    for idx,key in enumerate(args.x_features):
        #print(f'reading {key} var')
        scaling=1.0
        if key=='WDST' or key=='HEGT':
            scaling=1.0/args.scaling_x

        input_data=scaling*np.array(data[f"/FIELD/VARIABLES/{key}"],dtype = float) 
        input_data=np.nan_to_num(input_data,nan=0.0)

        new_feature=np.reshape(input_data,(args.input_xdim,args.input_ydim), order='C')
        new_feature=np.flip(new_feature,axis=0)

        X_features[idx]=new_feature

    Y_features=np.empty((len(args.y_features),args.target_xdim,args.target_ydim),dtype=float)

    for idx,key in enumerate(args.y_features):
        #print(f'reading {key} var')
        scaling=1.0
        if key=='U' or key=='V':
            scaling=1.0/args.scaling_y
        
        target_data=(scaling*np.array(data[f"/FIELD/VARIABLES/{key}"],dtype = float))+0.5
        target_data=np.nan_to_num(target_data,nan=0.0)
        
        new_feature=np.reshape(target_data,(args.target_xdim,args.target_ydim), order='C')
        new_feature=np.flip(new_feature,axis=0)

        Y_features[idx]=new_feature

    E_features=np.empty((len(args.e_features),args.target_xdim,args.target_ydim),dtype=float)

    for idx,key in enumerate(args.e_features):
        #print(f'reading {key} var')UPCNord_geometry],dtype = float) 
        
        new_feature=np.reshape(target_data,(args.target_xdim,args.target_ydim), order='C')
        new_feature=np.flip(new_feature,axis=0)
        
        if key=='GRDUX' or key=='GRDVY' or key=='GRDWZ':
            new_feature[:,0]=0.0
            new_feature[:,args.target_xdim-1]=0.0
            new_feature[0,:]=0.0
            new_feature[args.target_ydim-1,:]=0.0

        E_features[idx]=new_feature

    X_features=torch.unsqueeze(torch.from_numpy(X_features),0)
    Y_features=torch.unsqueeze(torch.from_numpy(Y_features),0)
    E_features=torch.unsqueeze(torch.from_numpy(E_features),0)

    return X_features, {'y':Y_features,'extra':E_features}

def get_args():
    """
    These arguments can be passed with "$ python train.py -<arg_name> <arg_value>",
    otherwise it used their CONSTANTS value by default.
    """
    parser = argparse.ArgumentParser(description='args for 2D H5 data samples training')
    ####DATASET PARAMETERS-------------------------------------------------------------------------------------------------------------------
    parser.add_argument('-dataset_base_path', default=DATASET_BASE_PATH, help='dataset folder name.')
    parser.add_argument('-output_path', default=OUTPUT_PATH)
    parser.add_argument('-data_sample_basename', default=DATA_SAMPLE_BASENAME, help='input dataset files base name')
    parser.add_argument('-model_basename', default=MODEL_BASENAME, help='model saving file base name')
    parser.add_argument('-model_loading_path', default=MODEL_LOADING_PATH, help='model weights loading path')
    parser.add_argument('-x_features', default=INPUT_FEAT, help='input feature names in an array')
    parser.add_argument('-y_features', default=TARGET_FEAT, help='target feature names in an array')
    parser.add_argument('-e_features', default=EXTRA_FEAT, help='extra feature names in an array')
    parser.add_argument('-input_xdim', type=int, default=INPUT_XDIM, help='input dataset sample size in X direction')
    parser.add_argument('-input_ydim', type=int, default=INPUT_YDIM, help='input dataset sample size in Y direction')
    parser.add_argument('-target_xdim', type=int, default=TARGET_XDIM ,help='target dataset sample size in X direction')
    parser.add_argument('-target_ydim', type=int, default=TARGET_YDIM ,help='target dataset sample size in Y direction')
    parser.add_argument('-spacing_x', type=float, default=SPACING_X ,help='datasample grid spacing in the X direction')
    parser.add_argument('-spacing_y', type=float, default=SPACING_Y ,help='datasample grid spacing in the Y direction')
    parser.add_argument('-scaling_x', type=float, default=SCALING_X ,help='whole dataset input max value to normalize input data into a [-1,1] range')
    parser.add_argument('-scaling_y', type=float, default=SCALING_Y ,help='whole dataset target max value to normalize output data into a [-1,1] range')
    parser.add_argument('-num_res_blocks', type=int, default=32, help='number of residual blocks in the generator model')
    args, _ = parser.parse_known_args()
    return args

if __name__ == '__main__':

    args = get_args() 

    device='null'
    if device not in ['cpu','cuda:0']: device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'used device={device}')   

    for wind_angle in STL2GeoTool_loop.WIND_DIRECTION:
        args.data_sample_basename = DATA_SAMPLE_BASENAME
        dataset_folder = f"{args.dataset_base_path}/output{wind_angle}-{DATA_SAMPLE_BASENAME}/"
        output_folder = f"{args.output_path}/output{wind_angle}-{DATA_SAMPLE_BASENAME}/" 

        os.makedirs(output_folder, exist_ok=True)
        print(f"Output folder created: {output_folder}")

        # Model initialization
        # model = UNet_wind(args)
        model=Generator2D(args) 
        # model = Discriminator2D(args)
        model = load_model(model, args).to(device)

        h5_files = [f for f in os.listdir(dataset_folder) if f.endswith('.h5')]

        for file in h5_files:
            index = file.split('-')[-2]  # Extract sample index from file name
            name_prefix = f"{args.data_sample_basename}-{index}"
            file_path = os.path.join(dataset_folder, file)


            # file=f'{args.dataset_base_path}{args.data_sample_basename}-{GEO}-{ANGLE}-{SAMPLE_ID}.h5'
            # print(f'Loading file {file}')

            model.train()
            x,output=load_input_sample(args,file_path)

            x=x.to(device)

            y=output['y']

            y=y.to(device)

            with torch.no_grad():
                ypred=model(x.float())

            # plot_field(y[0][0],f'{SAVING_FOLDER}/refU-{GEO}-{ANGLE}-{SAMPLE_ID}.png',x[0][0],vmin_cost=0.3,vmax_cost=0.8)
            # plot_field(y[0][1],f'{SAVING_FOLDER}/refV-{GEO}-{ANGLE}-{SAMPLE_ID}.png',x[0][0],vmin_cost=0.3,vmax_cost=0.7)
            # plot_field(ypred[0][0],f'{SAVING_FOLDER}/predU-{GEO}-{ANGLE}-{SAMPLE_ID}.png',x[0][0],vmin_cost=0.3,vmax_cost=0.8)
            # plot_field(ypred[0][1],f'{SAVING_FOLDER}/predV-{GEO}-{ANGLE}-{SAMPLE_ID}.png',x[0][0],vmin_cost=0.3,vmax_cost=0.7)
            print(f"Saving prediction and ground truth for {file}")
            # np.savetxt(os.path.join(output_folder, f"{name_prefix}-UGT.csv"), ypred[0][0].cpu().numpy(), delimiter=',')
            # np.savetxt(os.path.join(output_folder, f"{name_prefix}-VGT.csv"), ypred[0][1].cpu().numpy(), delimiter=',')
            # plot_field(ypred[0][0], os.path.join(output_folder, f"{name_prefix}-UGT"), x[0][0], vmin_cost=0.3, vmax_cost=0.8)
            # plot_field(ypred[0][1], os.path.join(output_folder, f"{name_prefix}-VGT"), x[0][0], vmin_cost=0.3, vmax_cost=0.7)

            # U_mag = np.sqrt(ypred[0][0].cpu()**2 + ypred[0][1].cpu()**2)
            # np.savetxt(os.path.join(output_folder, f"{name_prefix}-UMAG.csv"), U_mag, delimiter=',')
            # plot_field(U_mag, os.path.join(output_folder, f"{name_prefix}-UMAG"),x[0][0],vmin_cost=0.3,vmax_cost=0.8)
            # Create the mask where x[0][0] < 1.0
            mask = (x[0][0] >= 1.0).cpu().numpy()  # shape: (256, 256) or similar

            # Apply the mask to each output channel (Ux and Uy)
            Ux_masked = ypred[0][0].cpu().numpy() * mask
            Uy_masked = ypred[0][1].cpu().numpy() * mask

            # Dimensions
            print("Saving shape:", Ux_masked.shape)
            print("Saving shape:", Uy_masked.shape)

            # Save masked UGT and VGT
            np.savetxt(os.path.join(output_folder, f"{name_prefix}-UGT.csv"), Ux_masked, delimiter=',')
            np.savetxt(os.path.join(output_folder, f"{name_prefix}-VGT.csv"), Uy_masked, delimiter=',')

            

            # Plot masked Ux and Uy
            plot_field(ypred[0][0], os.path.join(output_folder, f"{name_prefix}-UGT"), x[0][0], vmin_cost=0.3, vmax_cost=0.8)
            plot_field(ypred[0][1], os.path.join(output_folder, f"{name_prefix}-VGT"), x[0][0], vmin_cost=0.3, vmax_cost=0.7)

            # Compute and save masked U magnitude
            np_umag = np.sqrt(Ux_masked**2 + Uy_masked**2)
            U_mag = np.sqrt(ypred[0][0].cpu()**2 + ypred[0][1].cpu()**2)
            np.savetxt(os.path.join(output_folder, f"{name_prefix}-UMAG.csv"), np_umag, delimiter=',')
            plot_field(U_mag, os.path.join(output_folder, f"{name_prefix}-UMAG"), x[0][0], vmin_cost=0.3, vmax_cost=0.8)


