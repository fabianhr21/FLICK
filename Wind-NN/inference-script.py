# inference-script.py
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch import nn
import numpy as np
import argparse
import matplotlib.pyplot as plt
import os, h5py
from Unet_model import UNet_wind
import sys
sys.path.append('../Pre-Process/')
import STL2GeoTool_loop

wind_angle = 0
DATA_SAMPLE_BASENAME='grid_of_cubes'
DATASET_PATH='../Pre-Process/output/'
OUTPUT_PATH='./output/'
MODEL_BASENAME= 'model' #'Wind-NN-2D-normalized-sigmoid'
MODEL_LOADING_PATH='/gpfs/scratch/bsc21/bsc084826/WRF-NN/inference-script/Models/Wind-NN-2D-normalized-sigmoid/'
# MODEL_LOADING_PATH='/gpfs/scratch/bsc21/bsc021742/NEURAL_NETWORKS/wind-NN/vel-magnitude/Wind-NN-2D-scaled-ReLu/checkpoints/' #VMAG
INPUT_FEAT=['MASK','HEGT','WDST'] #having a MASK distinguishing solid and fluid regions at the first position is mandatory.
TARGET_FEAT=['U','V']
EXTRA_FEAT=['GRDUX','GRDVY','GRDWZ']
# TARGET_FEAT=['VMAG']
INPUT_XDIM=256
INPUT_YDIM=256
TARGET_XDIM=256
TARGET_YDIM=256
SCALING_X=120. #whole dataset input max value
SCALING_Y=16.0 #whole dataset output max value
VERBOSE = 1

def get_args():
    """
    These arguments can be passed with "$ python train.py -<arg_name> <arg_value>",
    otherwise it used their CONSTANTS value by default.
    """
    parser = argparse.ArgumentParser(description='args for 2D H5 data samples training')
    ####DATASET PARAMETERS-------------------------------------------------------------------------------------------------------------------
    parser.add_argument('-dataset_path', default=DATASET_PATH, help='dataset folder name.')
    parser.add_argument('-output_path', default=OUTPUT_PATH, help='output folder name')
    parser.add_argument('-data_sample_basename', default=DATA_SAMPLE_BASENAME, help='input dataset files base name')
    parser.add_argument('-model_basename', default=MODEL_BASENAME, help='model saving file base name')
    parser.add_argument('-model_loading_path', default=MODEL_LOADING_PATH, help='model weights loading path and name for inference and training restart')
    parser.add_argument('-x_features', default=INPUT_FEAT, help='input feature names in an array')
    parser.add_argument('-y_features', default=TARGET_FEAT, help='target feature names in an array')
    parser.add_argument('-e_features', default=EXTRA_FEAT, help='extra features names in an array')
    parser.add_argument('-input_xdim', type=int, default=INPUT_XDIM, help='input dataset sample size in X direction')
    parser.add_argument('-input_ydim', type=int, default=INPUT_YDIM, help='input dataset sample size in Y direction')
    parser.add_argument('-target_xdim', type=int, default=TARGET_XDIM ,help='target dataset sample size in X direction')
    parser.add_argument('-target_ydim', type=int, default=TARGET_YDIM ,help='target dataset sample size in Y direction')
    parser.add_argument('-scaling_x', type=float, default=SCALING_X ,help='whole dataset input max value to normalize input data into a [-1,1] range')
    parser.add_argument('-scaling_y', type=float, default=SCALING_Y ,help='whole dataset target max value to normalize output data into a [-1,1] range')
    parser.add_argument('-verbose', type=int, default=VERBOSE, help='verbose level (0,1,2). 0: none. 1: text. 2: text and plots')
    args, _ = parser.parse_known_args()
    return args

def plot_field(field,name):
    if torch.cuda.is_available(): 
        plot_matrix=torch.clone(field)
        if plot_matrix.requires_grad:
            plot_matrix=plot_matrix.detach()
        plot_matrix=plot_matrix.cpu()

        plt.imshow(plot_matrix,cmap='magma')
        plt.colorbar()
        plt.show()
    else:
        plt.imshow(field,cmap='magma')
        plt.colorbar()
        plt.show()
        plt.savefig(f'image-{name}.png')
        plt.close()

def load_model(model,args):

    if not os.path.exists(f"{args.model_loading_path}{args.model_basename}.pt"):
        print(f"The {args.model_loading_path}{args.model_basename}.pt file does not exist. Check file path and name.")
        exit(0)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print(f"Loading model weights from {args.model_loading_path}{args.model_basename}.pt")
    checkpoint = torch.load(f"{args.model_loading_path}{args.model_basename}.pt",map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model=model.float()
    model.eval()
    return model

def load_input_sample(args,idx):
    if not os.path.exists(f"{args.dataset_path}{args.data_sample_basename}-{idx}-geodata.h5"):
       print(f"The file {args.dataset_path}{args.data_sample_basename}-{idx}-geodata.h5 does not exist")
       exit(0)

    file_path=f"{args.dataset_path}{args.data_sample_basename}-{idx}-geodata.h5"
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
        #print(f'reading {key} var')
        scaling=1.0
        if key=='GRDUX' or key=='GRDVY' or key=='GRDWZ':
            dist_scaling=args.scaling_x
            velo_scaling=args.scaling_y

            scaling=dist_scaling/velo_scaling

        target_data=scaling*np.array(data[f"/FIELD/VARIABLES/{key}"],dtype = float) 
        
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

if __name__ == '__main__':
    
    for wind_angle in STL2GeoTool_loop.WIND_DIRECTION:    
        args = get_args()
        DATA_SAMPLE_BASENAME = args.data_sample_basename
        DATASET_PATH= f'../Pre-Process/output/output{wind_angle}-{DATA_SAMPLE_BASENAME}/'
        OUTPUT_PATH=f'./output/output{wind_angle}-{DATA_SAMPLE_BASENAME}/'
        args = get_args()
        print(f"DATASET_PATH: {DATASET_PATH}, OUTPUT_PATH: {OUTPUT_PATH}, DATA_SAMPLE_BASENAME: {DATA_SAMPLE_BASENAME},")

 
        
        if not os.path.exists(OUTPUT_PATH):
            os.makedirs(OUTPUT_PATH)
    
        #model creation
        model=UNet_wind(args)

        #trained model weights loading
        model=load_model(model,args)

        #data sample indices to load examples from the dataset
        files = os.listdir(DATASET_PATH)
        h5_files = [file for file in files if file.endswith('.h5')]
        sample_indices = [i for i in range(0,len(h5_files))]

        for idx in sample_indices:
            #loading data sample. x corresponds to the model input fields (MASK,HEGT,WDST) while y corresponds to the output groundtruth fields (U,V). 
            # x,y=load_input_sample(args,idx)
            # file_path = f"{path}/{h5_files[idx]}"
            # file_path = '/gpfs/scratch/bsc21/bsc084826/WRF-NN/UPC_nord/output/'
            # base_name = f'UPC_small_origin-{idx}-geodata.h5'
            x,y=load_input_sample(args,idx)
            name = f"{DATA_SAMPLE_BASENAME}-{idx}-"

            print("Printing model input fields")
            # plot_field(x[0][0],name + 'MASK')  #Field 0 corresponds to MASK
            np.savetxt(OUTPUT_PATH + name + 'MASK_matrix.csv', x[0][0].numpy(), delimiter=',')
            # plot_field(x[0][1],name+'HEGT')  #Field 1 corresponds to building height HEGT
            # plot_field(x[0][2],'WDST')  #Field 2 corresponds to distance to nearest wall WDST

            print("Evaluating model")
            with torch.no_grad():
                ypred=model(x.float())

            print("Printing model output groundtruth and model predictions")    
            # y is groundtruth and ypred is the model prediciton.
            # plot_field(y[0][0],'UGT')      #Field 0 corresponds to U velocity component.

            # plot_field(ypred[0][0], name +'VMAG')
            np.savetxt(OUTPUT_PATH + name + 'UGT_matrix.csv', ypred[0][0].numpy(), delimiter=',')
            # plot_field(y[0][1],'VGT')      #Field 1 corresponds to V veloctiy component.
            np.savetxt(OUTPUT_PATH + name + 'VGT_matrix.csv', ypred[0][1].numpy(), delimiter=',') 
            # plot_field(ypred[0][1],name+'VPRED')