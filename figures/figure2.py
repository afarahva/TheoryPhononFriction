
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

# Path to data 4x4x4 lattice
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
    freqmode = npz['freq_arr'].mean(axis=0)
    print(npz['freq_arr'].shape)
    t_arr1.append(t)
    Kt_arr1.append(Kt)
    Kw_arr1.append(Kw)
    freqmode_arr1.append(freqmode)
    
    
# Path to Density of States Data
path1   = "../calc_memory/results/CO_Pt_EMT_fcc111_cell=4x4x8/w=%scm-1.npz"%(0)
npz = np.load(path)
freqmode_0 = npz['freq_arr'].mean(axis=0)


# Path to data 8x8x8 lattice
path1   = "../calc_memory/results/CO_Pt_EMT_fcc111_cell=8x8x8/w=%scm-1.npz"%(100)
path2   = "../calc_memory/results/CO_Pt_EMT_fcc111_cell=8x8x8/w=%scm-1.npz"%(300)
path3   = "../calc_memory/results/CO_Pt_EMT_fcc111_cell=8x8x8/w=%scm-1.npz"%(500)
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
############ Make Plots

# time, x-axis limits
tulims = [-1,-0.5,-0.25]
tolims = [18,10,5]

# indices to plot
plt_indx = [0,1,2]

# frequency gride
omegaTHz_to_nucm = lib.freq_converter(1,'radians/ps','cm-1')
nu_grid = np.linspace(0,np.array(freqmode_arr1).max()*1.2*omegaTHz_to_nucm,1000)


# amplitude and width for Lorentzian plotting of dirac comb
mult = 12.0
width = 2.0

fig, axs = plt.subplots(nrows=3,ncols=3,figsize=[13,13])


# Plot Kt vs time
for i in range(len(plt_indx)):
    ax = axs[0][i]
    j = plt_indx[i]
    
    ax.plot(t_arr1[j], Kt_arr1[i], color="black", linestyle="-", alpha=0.75, linewidth=2.0, label="4x4x8")
    ax.plot(t_arr2[j], Kt_arr2[i], color="blue", linestyle="--", alpha=0.75, linewidth=2.0, label="8x8x8")
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
    
    ax.plot(nu_grid, Kw_plot1, color="black", linestyle="-", alpha=0.75, linewidth=2.0, label="4x4x4")
    ax.plot(nu_grid, Kw_plot2, color="blue", linestyle="--", alpha=0.75, linewidth=2.0, label="8x8x8")
    ax.set_xlim(-2, 250)
    ax.tick_params(axis='both', which='major', labelsize=20)
    
# Plot rho_w vs freq
width = 1.0
dens_0 = lib.smear_spectral_lorentz( np.ones(len(freqmode_0)), freqmode_0*omegaTHz_to_nucm, nu_grid, width)
for i in range(len(plt_indx)):
    ax = axs[2][i]
    j = plt_indx[i]

    freqmode = freqmode_arr1[j] * omegaTHz_to_nucm
    dens = lib.smear_spectral_lorentz( np.ones(len(freqmode)), freqmode, nu_grid, width)
    
    ax.plot(nu_grid, dens_0, linestyle="-", color="black", alpha=0.75, linewidth=2.0, label="Bare Pt(111)")
    ax.plot(nu_grid, dens, linestyle="--", color="blue", alpha=0.75, linewidth=2.0, label="Pt(111) + CO")
    ax.set_xlim(-2, 250)
    ax.set_ylim(-0.1,9)
    ax.tick_params(axis='both', which='major', labelsize=20)

# Set Axes
axs[0][0].set_ylabel(r" $K(t)$ ",fontsize=28)
axs[0][1].set_xlabel("$t$ (ps)",fontsize=28)
axs[1][0].set_ylabel(r" $\bar{K}(\omega)$ ",fontsize=28)
axs[2][0].set_ylabel(r" $\rho(\omega)$ ",fontsize=28)
axs[2][1].set_xlabel(r"$\omega$ (cm$^{-1})$",fontsize=28)
axs[0][0].legend(frameon=False,fontsize=20)
axs[2][0].legend(frameon=False,fontsize=14,loc=2)
axs[0][0].set_title(r"$\omega_{\mathrm{as}} = 100$ cm$^{-1}$",fontsize=26,y=1.05)
axs[0][1].set_title(r"$\omega_{\mathrm{as}} = 300$ cm$^{-1}$",fontsize=26,y=1.05)
axs[0][2].set_title(r"$\omega_{\mathrm{as}} = 500$ cm$^{-1}$",fontsize=26,y=1.05)


fig.tight_layout()
plt.savefig("./figure2/figure2_raw.pdf",bbox_inches="tight")
plt.show()

