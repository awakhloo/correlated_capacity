#!/usr/bin/env python

#SBATCH --time=0-16:00:00
#SBATCH --partition ccn
#SBATCH --nodes=1
import sys
import os 
sys.path.append(os.getcwd()) 


import os
import numpy as np
import capacity.mean_field_cap as mf
import capacity.replica_correlations as rep
import capacity.low_rank_capacity as ori
import capacity.manifold_simcap_analysis as num 
import capacity.utils.sim_block_diag as sim
import pandas as pd

P, N, samples = 80, 3800, 40
df=pd.DataFrame(columns=['intensity', 'alpha_mf', 'alpha_rep', 'alpha_sim', 'alpha_nc', 'num_run', 'seed'])
intensities = np.linspace(0, 1, 20) 
level_sizes_cent = [2, 2, 2]
corrs_cent = np.array([0, 0.3, 0.5, 0.7])
level_sizes_pt = [2, 4]
corrs_pt = np.array([0.0, 0.5, 0.9])
kappa=0
n_t=100
n_rep=30 
num_runs = 5
seeds = np.random.randint(0, high=4294967295, size=(num_runs, len(intensities)))
for nrun in range(num_runs): 
    for n, i in enumerate(intensities): 
        np.random.seed(seeds[nrun, n])
        cloud = sim.gaussian_clouds(P, N, samples, level_sizes_cent, 
                                  i*corrs_cent, level_sizes_pt, i*corrs_pt, radius=1.,
                                    center_radius=5.)
        cloud = [c.T for c in cloud]
        print('dimension is ', cloud[0].shape)
        alpha_rep, *_ = rep.manifold_analysis_corr(cloud, n_t=n_t, kappa=kappa)
        alpha_nc, *_ = ori.manifold_analysis_corr(cloud, n_t=n_t, kappa=kappa) 
        alpha_nc = 1/np.mean(1/alpha_nc)
        print('Replica and NC: ', alpha_rep, alpha_nc)
        alpha_mf, *_ = mf.manifold_analysis_corr(cloud, n_t=n_t, kappa=kappa)
        alpha_mf = 1/np.mean(1/alpha_mf)
        alpha_sim, *_ = num.manifold_simcap_analysis(cloud, n_rep=n_rep)
        print('meanfield and Sim: ', alpha_mf, alpha_sim)
        dct = {'intensity' : i, 'alpha_mf' : alpha_mf, 'alpha_sim' : alpha_sim, 'alpha_nc' : alpha_nc, 
            'alpha_rep' : alpha_rep, 'num_run' : nrun, 'seed' : seeds[nrun, n]}
        df = df.append(pd.DataFrame(dct, index=[n]))
        df.to_csv(os.getcwd() + '/results/clouds/gauss_cloud.csv')
