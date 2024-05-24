
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
figure1.py

Creates figure 1: Memory Kernels, Spectral Densities, and density of states of 
single unit cell Pt(111) lattice with EMT
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# Library functions for project
sys.path.append("../lib/") if '../lib/' not in sys.path else False
import project_lib as lib

lib.init_niceplots()

#%%
############ Load  Data

# Path to data for single unit 4x4x4 unit ell
path1   = "../calc_memory/results/CO_Pt_EMT_fcc111_cell=4x4x8/w=%scm-1.npz"%(100)
path2   = "../calc_memory/results/CO_Pt_EMT_fcc111_cell=4x4x8/w=%scm-1.npz"%(300)
path3   = "../calc_memory/results/CO_Pt_EMT_fcc111_cell=4x4x8/w=%scm-1.npz"%(500)

paths = [path1,path2,path3]

# Load Theory Data
t_arr1 = []
Kt_arr1 = []
Kw_arr1 = []
freqmode_arr1 = []
for i in range(len(paths)):
    path = paths[i]
    npz = np.load(path)
    
    t = npz['t_arr']
    dt = t[1]-t[0]
    Nt = len(t)

    Kt = npz['Kt_arr'].mean(axis=0)
    Kw = npz['Kw_arr'].mean(axis=0)
    freqmode = npz['freq_arr'].mean(axis=0).flatten()

    t_arr1.append(t)
    Kt_arr1.append(Kt)
    Kw_arr1.append(Kw)
    freqmode_arr1.append(freqmode)

# Path to k-integrated data
path1   = "../calc_memory/results/CO_Pt_EMT_fcc111_cell=4x4x8_supercell=6x6/w=%scm-1.npz"%(100)
path2   = "../calc_memory/results/CO_Pt_EMT_fcc111_cell=4x4x8_supercell=6x6/w=%scm-1.npz"%(300)
path3   = "../calc_memory/results/CO_Pt_EMT_fcc111_cell=4x4x8_supercell=6x6/w=%scm-1.npz"%(500)
paths = [path1,path2,path3]

# Load Theory Data
t_arr2 = []
Kt_arr2 = []
Kw_arr2 = []
freqmode_arr2 = []
for i in range(len(paths)):
    path = paths[i]
    npz = np.load(path)
    
    t = npz['t_arr']
    dt = t[1]-t[0]
    Nt = len(t)    
    
    Kt = npz['Kt_arr'].mean(axis=0)
    Kw = npz['Kw_arr'].mean(axis=0)
    freqmode = npz['freq_arr'].mean(axis=0).flatten()
    
    t_arr2.append(t)
    Kt_arr2.append(Kt)
    Kw_arr2.append(Kw)
    freqmode_arr2.append(freqmode)

#%%
############ Make Plots

omegaTHz_to_nucm = lib.freq_converter(1,'radians/ps','cm-1')
nu_grid = np.linspace(0,freqmode_arr1[-1].max()*1.1*omegaTHz_to_nucm,1000)
fig, axs = plt.subplots(nrows=1,ncols=3,figsize=[13,4.66])

# amplitude and width for Lorentzian plotting of dirac comb
mult = 12.0
width = 2.0
    
# Plot Kw vs freq
for j in range(len(paths)):
    ax = axs[j]
    
    Kw = Kw_arr1[j]
    freq_arr = lib.freq_converter(freqmode_arr1[j],'radians/ps','cm-1')
    Kw_plot1 = lib.smear_spectral_lorentz(mult*Kw,freq_arr,nu_grid,width)
    
    Kw = Kw_arr2[j]
    freq_arr = lib.freq_converter(freqmode_arr2[j],'radians/ps','cm-1')
    Kw_plot2 =  lib.smear_spectral_lorentz(mult*Kw,freq_arr,nu_grid,width)
    
    ax.plot(nu_grid, Kw_plot1, color="black", linestyle="-", alpha=0.75, linewidth=2.0, label="4x4")
    ax.plot(nu_grid, Kw_plot2, color="blue", linestyle="--", alpha=0.75, linewidth=2.0, label="infinite")
    ax.set_xlim(-0.5, 250)
    ax.tick_params(axis='both', which='major', labelsize=20)
    
# Set Axes
axs[0].set_ylabel(r" $\bar{K}(\omega)$ ",fontsize=28)
axs[1].set_xlabel(r"$\omega$ (cm$^{-1})$",fontsize=28)
axs[0].legend(frameon=False,fontsize=20)
axs[0].set_title(r"$\omega_{\mathrm{as}} = 100$ cm$^{-1}$",fontsize=26,y=1.05)
axs[1].set_title(r"$\omega_{\mathrm{as}} = 300$ cm$^{-1}$",fontsize=26,y=1.05)
axs[2].set_title(r"$\omega_{\mathrm{as}} = 500$ cm$^{-1}$",fontsize=26,y=1.05)

fig.tight_layout()
plt.savefig("./figure4/figure4.pdf",bbox_inches="tight")
plt.show()

