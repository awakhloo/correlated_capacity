#!/usr/bin/env python

#SBATCH --time=0-4:00:00
#SBATCH --partition ccn
#SBATCH --nodes=1
#SBATCH --constraint=broadwell

import sys
sys.path.append('/mnt/home/awakhloo/ceph/block_projection') 
import numpy as np
import sphere_sim_capacity as sim
import matplotlib.pyplot as plt
import pandas as pd

P, N, K = 200, 1500, 7
nrep=5
df=pd.DataFrame(columns=['psi1', 'lam1', 'r', 'r0', 'alpha'])
np.random.seed(4353456)
ratios = np.linspace(0.25, 1, 10)
seedos = np.random.randint(6999999, size=(nrep, len(ratios), 4, 2))
free_variables = ['r', 'r0', 'lam', 'psi']
psi, lam, r, r0 = 0.85, 0.7, 0.5, 3.
base_vars = {'r' : r, 'r0' : r0, 'psi' : psi, 'lam' : lam}
adjust = {
          'r' : lambda c : c*r0*np.sqrt(1-psi)/(np.sqrt(K*(1-lam))),
          'r0' : lambda c : r*np.sqrt(K*(1-lam))/(c*np.sqrt(1-psi)),
          'lam' : lambda c : 1-(c*r0*np.sqrt(1-psi)/(r*np.sqrt(K)))**2,
          'psi' : lambda c: 1- (r*np.sqrt(K*(1-lam))/(r0*c))**2,
          }
print('cap_max_lam  ', r*np.sqrt(K)/(r0 * np.sqrt(1-psi)))
print('cap_min_psi ', r*np.sqrt(K*(1-lam))/r0) 
for i in range(nrep):
    for j, ratio in enumerate(ratios): 
        for k, var in enumerate(free_variables): 
            np.random.seed(seedos[i,j,k,0])
            base = base_vars.copy()
            base[var] = adjust[var](ratio) 
            print(ratio, base) 
            lam0, lam1 = 1 * base['r']**2, base['lam'] * base['r']**2 
            psi0, psi1 = 1 * base['r0']**2, base['psi'] * base['r0']**2
            sphere_axes = sim.get_axes(N, P, K, lam0, lam1, psi0, psi1)
            out = sim.sphere_simcap(sphere_axes, n_rep=1, seed=seedos[i,j,k,1])
            alpha = out[0]
            dat = pd.DataFrame({'r' : base['r'],
                                'r0':base['r0'],
                                'psi1':base['psi'],
                                'lam1':base['lam'],
                                'alpha' : alpha, 
                                'rep' : i}, index=[0])
            df = df.append(dat).reset_index(drop=True)
            df.to_csv(f'/mnt/home/awakhloo/ceph/slurm-codes/ratio_sphere_simcap_P{P}_N{N}_K{K}_nrep{nrep}.csv')




