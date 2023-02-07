#!/usr/bin/env python

#SBATCH --time=3-00:00:00
#SBATCH --partition ccn
#SBATCH --nodes=1
#SBATCH -C skylake
import sys
import os 
sys.path.append(os.getcwd()) 


import numpy as np
import os
import re
import gc
from glob import glob

import torch
from torchvision import models, datasets, transforms

import capacity.replica_correlations as rep
import capacity.manifold_simcap_analysis as num
import capacity.low_rank_capacity as ori
import capacity.utils.make_manifold_data as md 
import argparse

raw_outdir = os.getcwd() + '/results/simclr/layers/raw'  # where to save raw representations
proj_outdir =  os.getcwd() + '/results/simclr/layers/proj_8k' # where to save 8k dimensional projections
outdir = os.getcwd() + '/results/simclr/capacities' # where to save capacity data

parser = argparse.ArgumentParser() 
parser.add_argument("--samp", help = "iteration number", type=int)
parser.add_argument("--imagenetpath", help = "path to ILSVRC training dataset (e.g., /.../CLS-LOC/train")
args = parser.parse_args()
samp = args.samp
imagenetpath = args.imagenetpath

np.random.seed(435348957)
# load the weights from the VISSL model zoo 
simclr_url ='https://dl.fbaipublicfiles.com/vissl/model_zoo/simclr_rn50_1000ep_simclr_8node_resnet_16_07_20.afe428c7/model_final_checkpoint_phase999.torch'
weights = torch.utils.model_zoo.load_url(simclr_url, map_location=torch.device('cpu'))
trunk = weights['classy_state_dict']['base_model']['model']['trunk']
trunk = {re.sub('_feature_blocks\.', '', key) : val for key, val in trunk.items()}
dummy_weight = torch.rand((1000, 2048))
dummy_bias = torch.rand((1000, ))
trunk['fc.weight'] = dummy_weight
trunk['fc.bias'] = dummy_bias
mod = models.resnet50()
mod.load_state_dict(trunk)
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
test_trnsfrm = transforms.Compose([transforms.Resize(256),
                                  transforms.CenterCrop(224),
                                  transforms.ToTensor(), 
                                  transforms.Normalize(mean, std)
                                 ])

dl = datasets.ImageFolder(imagenetpath, transform = test_trnsfrm)
mod.eval()

sampled_classes = 80
num_per_class = 45 #65
num_samp_reps = 5

proj_seeds = np.random.randint(low=0, high=10000, size=(2,num_samp_reps))
print('Seeds: ', proj_seeds)

# Choose the layers we use 
# node_names = ['x', 'layer1.1.conv1', 'layer1.2.bn3', 'layer2.1.relu', 'layer2.2.relu_2', 'layer3.0.conv3', 'layer3.2.bn1', 'layer3.3.add', 'layer3.5.relu_1', 'layer4.1.conv1', 'layer4.2.bn3', 'layer4.2.relu_1', 'layer4.2.relu_2', 'layer4.2.conv2', 'layer4.2.bn1']
node_names = [f'layer1.{i}.relu' for i in range(3)] + \
                      [f'layer2.{i}.relu' for i in range(4)] + \
                      [f'layer3.{i}.relu' for i in range(6)] + \
                      [f'layer4.{i}.relu' for i in range(3)]
# just grab the convolutional layers: 
#node_names = ['x', 'layer1.1.conv1', 'layer1.2.conv2', 'layer2.0.conv3', 'layer2.2.conv1', 'layer2.3.conv2', 'layer3.0.conv3', 'layer3.2.conv1', 'layer3.3.conv2', 'layer3.4.conv3', 'layer4.0.conv1', 'layer4.1.conv2', 'layer4.2.conv3']
# node_names = ['x', 'layer1.0.conv1', 'layer1.1.conv2', 'layer1.2.conv3', 'layer2.1.conv1', 'layer2.2.conv2', 'layer2.3.conv3', 'layer3.1.conv1', 'layer3.2.conv2', 'layer3.3.conv3', 'layer3.5.conv1', 'layer4.0.conv2', 'layer4.1.conv3', 'layer4.2.conv3']

# make directories for this draw 
os.makedirs(raw_outdir + f'/rep_{samp}', exist_ok=False)
os.makedirs(proj_outdir + f'/rep_{samp}', exist_ok=False)
os.makedirs(outdir + f'/rep_{samp}', exist_ok=False)

## sample the classes
dat = md.make_manifold_data(dl, sampled_classes, num_per_class, seed = proj_seeds[0, samp], max_class=1_000) #sample dict keys are the torch ids 
print(len(dat))
print([d.shape for d in dat], flush=True) 
activations = md.extract_features(dat, mod, node_names=node_names)
# save the raw activations
for key, val in activations.items(): 
   print(key, flush=True)
   print(type(val)) 
   print('Number of classes and shapes are: ', len(val), val[0].shape)
   if len(set([v.shape for v in val]))!=1: print( 'Uneven shapes in raw activations!')
   np.save(raw_outdir + f'/rep_{samp}/' + key + '.npy', np.array({key : val}))
del activations
gc.collect()

'''
project down to an 8,000 dimensional ambient dimension
''' 
np.random.seed(proj_seeds[1,samp])
dim=8_000
layers = glob(raw_outdir + f'/rep_{samp}/'+ '/*.npy')
layer_sizes = {} 
for layer in layers: 
    name = os.path.relpath(layer, raw_outdir + f'/rep_{samp}')
    data = np.load(layer, allow_pickle=True).item()
    data = list(data.values())[0]
    print(f'{name} data shape is {data[0].shape} with {len(data)} classes', flush=True)
    layer_sizes[name] = data[0].shape # save the layer widths for supplementary table 1
    X = [d.detach().to('cpu').numpy() for d in data] 
    X = [d.reshape(d.shape[0], -1).T for d in X]
    # Get the number of features in the flattened data
    N = X[0].shape[0]
    # If N is greater than dim, do the random projection to 10,000 features
    if N > dim:
        print("Projecting {}".format(name), flush=True)
        M = np.random.randn(dim, N)
        M /= np.sqrt(np.sum(M*M, axis=1, keepdims=True))
        X = [np.matmul(M, d) for d in X]
    np.save(f'{proj_outdir}/rep_{samp}/{name}', np.array(X))
    del data, X, M
    gc.collect()
np.save(f'{proj_outdir}/layer_sizes.npy', np.array(layer_sizes))

''' 
run the analyses
'''
# get all layers
layers = glob(proj_outdir + f'/rep_{samp}/*.npy')
np.random.seed(887788668997)
seeds = np.random.randint(low=1, high=1_000_000,size=(num_samp_rep, len(layers), 2))
print(seeds)
for i, layer in enumerate(layers): 
    np.random.seed(seeds[samp, i, 0])
    name = os.path.relpath(layer, proj_outdir + f'/rep_{samp}')
    name = name.replace('.npy', '')
    print(f'On layer {name}', flush = True)
    mfs = np.load(layer, allow_pickle=True)
    mfs = [m for m in mfs]
    print(f'Shape check: {mfs[0].shape}')
    alpha_ncom, *_ = ori.manifold_analysis_corr(mfs, 0, 150)
    print(f'n. comms: {1/np.mean(1/alpha_ncom)}', flush = True)
    res_dct = {}
    res_dct['alpha_ncom'] = alpha_ncom
    alpha_c, *_ = rep.manifold_analysis_corr(mfs, 0, 150)
    res_dct['alpha_c'] = alpha_c
    print(f'theory: {alpha_c}', flush=True)
    alpha_sim, P, Nc0, N_vec, p_vec = num.manifold_simcap_analysis(mfs, n_rep=20, seed=seeds[samp, i, 1])
    res_dct['alpha_sim'] = alpha_sim
    print(f'simulation: {alpha_sim}', flush = True)
    np.save(outdir + f'/rep_{samp}/{name}_results.npy', np.array(res_dct))
