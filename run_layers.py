#!/usr/bin/env python

#SBATCH --time=1-00:00:00
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
import capacity.mean_field_cap as mf
import argparse

raw_outdir = os.getcwd() + '/results/simclr/layers/raw'  # where to save raw representations
proj_outdir =  os.getcwd() + '/results/simclr/layers/proj_8k' # where to save 8k dimensional projections
outdir = os.getcwd() + '/results/simclr/capacities/sanity_simcap' # where to save capacity data

parser = argparse.ArgumentParser() 
parser.add_argument("--samp", help = "iteration number", type=int)
args = parser.parse_args()
samp = args.samp

sampled_classes = 70
num_per_class = 45 #65
num_samp_reps = 5

''' 
run the analyses
'''
# get all layers
layers = glob(proj_outdir + f'/rep_{samp}/*.npy')
np.random.seed(878668397)
seeds = np.random.randint(low=1, high=1_000_000,size=(num_samp_reps, len(layers), 2))
print(seeds)
n_t, kappa, n_rep = 150, 0, 20 # number of gaussian vectors for alphas, the margin, and the number of label configs to try in simcap 
for i, layer in enumerate(layers): 
    np.random.seed(seeds[samp, i, 0])
    name = os.path.relpath(layer, proj_outdir + f'/rep_{samp}')
    name = name.replace('.npy', '')
    print(f'On layer {name}', flush = True)
    mfs = np.load(layer, allow_pickle=True)
    mfs = [m for m in mfs]
    print(f'Shape check: {mfs[0].shape}')
    # alpha_ncom, *_ = ori.manifold_analysis_corr(mfs, kappa, n_t)
    # print(f'n. comms: {1/np.mean(1/alpha_ncom)}', flush = True)
    res_dct = {}
    # res_dct['alpha_ncom'] = alpha_ncom
    # alpha_c, *_ = rep.manifold_analysis_corr(mfs, kappa, n_t)
    # res_dct['alpha_c'] = alpha_c
    # print(f'theory: {alpha_c}', flush=True)
    alpha_sim, P, Nc0, N_vec, p_vec = num.manifold_simcap_analysis(mfs, n_rep=n_rep, seed=seeds[samp, i, 1])
    res_dct['alpha_sim'] = alpha_sim
    print(f'simulation: {alpha_sim}', flush = True)
    # alpha_mf, *_ = mf.manifold_analysis_corr(mfs, n_t=n_t, kappa=kappa)
    # alpha_mf = 1/np.mean(1/alpha_mf)
    # res_dct['alpha_m'] = alpha_mf 
    # print(f'mean field: {alpha_mf}', flush=True)
    np.save(outdir + f'/rep_{samp}/{name}_results.npy', np.array(res_dct))
