
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
figureS1.py

Creates figure S1: Memory Kernels, Spectral Densities, and density of states of 
single unit cell Pt(111) lattice with LJ
"""

import os
import sys
import numpy as np
import scipy.fft as spfft
import matplotlib.pyplot as plt
from matplotlib import cm

# Library functions for project
sys.path.append("../lib/") if '../lib/' not in sys.path else False
import project_lib as lib

lib.init_niceplots()
omegaTHz_to_nucm = lib.freq_converter(1,'radians/ps','cm-1')

#%%
############ Load  Data

# Path to data 4x4x4 lattice
path1   = "../calc_memory/results/CO_Pt_EMT_fcc111_cell=4x4x4/w=%scm-1.npz"%(100)
path2   = "../calc_memory/results/CO_Pt_EMT_fcc111_cell=4x4x4/w=%scm-1.npz"%(300)
path3   = "../calc_memory/results/CO_Pt_EMT_fcc111_cell=4x4x4/w=%scm-1.npz"%(500)
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
    freqmode = npz['freq_arr'].mean(axis=0)
    print(npz['freq_arr'].shape)
    t_arr1.append(t)
    Kt_arr1.append(Kt)
    Kw_arr1.append(Kw)
    freqmode_arr1.append(freqmode)
    
path1   = "../calc_memory/results/CO_Pt_EMT_fcc100_cell=4x4x4/w=%scm-1.npz"%(100)
path2   = "../calc_memory/results/CO_Pt_EMT_fcc100_cell=4x4x4/w=%scm-1.npz"%(300)
path3   = "../calc_memory/results/CO_Pt_EMT_fcc100_cell=4x4x4/w=%scm-1.npz"%(500)
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
    freqmode = npz['freq_arr'].mean(axis=0)
    print(npz['freq_arr'].shape)
    t_arr2.append(t)
    Kt_arr2.append(Kt)
    Kw_arr2.append(Kw)
    freqmode_arr2.append(freqmode)


#%%
tulims = [-1,-0.5,-0.25]
tolims = [18,10,5]
plt_indx = [0,1,2]

nu_grid = np.linspace(0,np.array(freqmode_arr1).max()*2.0*omegaTHz_to_nucm,1000)


fig, axs = plt.subplots(nrows=2,ncols=3,figsize=[13,8.6])

mult = 12.0
width = 3.0

# Plot Kt vs time
for i in range(len(plt_indx)):
    ax = axs[0][i]
    j = plt_indx[i]
    
    ax.plot(t_arr1[j], Kt_arr1[i], color="black", linestyle="-", alpha=0.75, linewidth=2.0, label="EMT")
    ax.plot(t_arr2[j], Kt_arr2[i], color="blue", linestyle="--", alpha=0.75, linewidth=2.0, label="LJ")

    ax.set_xlim(tulims[i],tolims[i])
    ax.tick_params(axis='both', which='major', labelsize=20)
    #ax.set_title(r" $\omega_{as} = %d\ \mathrm{cm}^{-1}$"%freq_as_arr[j],fontsize=20)
    
# Plot Kw vs freq
for i in range(len(plt_indx)):
    ax = axs[1][i]
    j = plt_indx[i]
    
    Kw = Kw_arr1[j]
    freq_arr = freqmode_arr1[j] * omegaTHz_to_nucm
    Kw_plot1 = lib.smear_spectral_lorentz(mult*Kw,freq_arr,nu_grid,width)
    
    Kw = Kw_arr2[j]
    freq_arr = freqmode_arr2[j] * omegaTHz_to_nucm
    Kw_plot2 = lib.smear_spectral_lorentz(mult*Kw,freq_arr,nu_grid,width)
    
    
    ax.plot(nu_grid, Kw_plot1, color="black", linestyle="-", alpha=0.75, linewidth=2.0, label="EMT")
    ax.plot(nu_grid, Kw_plot2, color="blue", linestyle="--", alpha=0.75, linewidth=2.0, label="LJ")
    ax.set_xlim(-2, 300)
    ax.tick_params(axis='both', which='major', labelsize=20)
    

# Set Axes
axs[0][0].set_ylabel(r" $K(t)$ ",fontsize=28)
axs[0][1].set_xlabel("$t$ (ps)",fontsize=28)
axs[1][0].set_ylabel(r" $\bar{K}(\omega)$ ",fontsize=28)
axs[1][1].set_xlabel(r"$\omega$ (cm$^{-1})$",fontsize=28)
axs[0][0].legend(frameon=False,fontsize=20)
axs[0][0].set_title(r"$\omega_{as} = 100$ cm$^{-1}$",fontsize=26,y=1.05)
axs[0][1].set_title(r"$\omega_{as} = 300$ cm$^{-1}$",fontsize=26,y=1.05)
axs[0][2].set_title(r"$\omega_{as} = 600$ cm$^{-1}$",fontsize=26,y=1.05)

fig.tight_layout()
plt.savefig("./figureS2/figureS2.pdf",bbox_inches="tight")
plt.show()

