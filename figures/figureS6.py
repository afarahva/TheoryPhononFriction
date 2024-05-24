
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
figureS6.py

Creates figure S6: Spectral densities for strong coupling perturbation theory
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
path1   = "../calc_memory/results/CO_Pt_EMT_fcc111_cell=4x4x4_pert/w=%scm-1.npz"%(100)
path2   = "../calc_memory/results/CO_Pt_EMT_fcc111_cell=4x4x4_pert/w=%scm-1.npz"%(150)
path3   = "../calc_memory/results/CO_Pt_EMT_fcc111_cell=4x4x4_pert/w=%scm-1.npz"%(200)
paths = [path1,path2,path3]

# Load Theory Data
freq_1 = []
freq_2 = []
freq_exact = []
Kw_strong1 = []
Kw_strong2 = []
Kw_exact = []
for i in range(len(paths)):
    path = paths[i]
    npz = np.load(path)
    
    freq = npz['freqs_exact']
    Kw = npz["Kw_ave_exact"]
    freq_exact.append(freq)
    Kw_exact.append(Kw)
    
    freq = npz['freqs_strong_1']
    Kw = npz["Kw_ave_strong1"]
    freq_1.append(freq)
    Kw_strong1.append(Kw)
    
    freq = npz['freqs_strong_2']
    Kw = npz["Kw_ave_strong2"]
    freq_2.append(freq)
    Kw_strong2.append(Kw)


#%%
tulims = [-1,-0.5,-0.25]
tolims = [18,10,5]

nu_grid = np.linspace(0,np.array(freq_exact).max()*1.5*omegaTHz_to_nucm,1000)


fig, axs = plt.subplots(nrows=1,ncols=3,figsize=[13,4.3])

mult = 12.0
width = 3.0

# Plot Kw vs freq
for i in range(len(paths)):
    ax = axs[i]
    
    Kw = Kw_exact[i]
    freq_arr = freq_exact[i] * omegaTHz_to_nucm
    Kw_plot1 = lib.smear_spectral_lorentz(mult*Kw,freq_arr,nu_grid,width)
    
    Kw = Kw_strong1[i]
    freq_arr = freq_1[i] * omegaTHz_to_nucm
    Kw_plot2 = lib.smear_spectral_lorentz(mult*Kw,freq_arr,nu_grid,width)
    
    Kw = Kw_strong2[i]
    freq_arr = freq_2[i] * omegaTHz_to_nucm
    Kw_plot3 = lib.smear_spectral_lorentz(mult*Kw,freq_arr,nu_grid,width)
    
    ax.plot(nu_grid, Kw_plot1, color="black", linestyle="-", alpha=0.5, linewidth=3.0, label="exact")
    ax.plot(nu_grid, Kw_plot2, color="blue", linestyle="--", alpha=0.75, linewidth=2.0, label="1st order")
    ax.plot(nu_grid, Kw_plot3, color="red", linestyle="-.", alpha=0.75, linewidth=2.0, label="2nd order")

    ax.set_xlim(-2, 250)
    ax.tick_params(axis='both', which='major', labelsize=20)
    

# Set Axes
axs[0].set_ylabel(r" $\bar{K}(\omega)$ ",fontsize=28)
axs[1].set_xlabel(r"$\omega$ (cm$^{-1})$",fontsize=28)
axs[0].set_title(r"""$\omega_{as} = 350$ cm$^{-1}$
                     $\tilde{\omega}_{s} = 159$ cm$^{-1}$""",fontsize=26,y=1.05)
axs[1].set_title(r"""$\omega_{as} = 400$ cm$^{-1}$
                     $\tilde{\omega}_{s} = 173$ cm$^{-1}$""",fontsize=26,y=1.05)
axs[2].set_title(r"""$\omega_{as} = 450$ cm$^{-1}$
                     $\tilde{\omega}_{s} = 188$ cm$^{-1}$""",fontsize=26,y=1.05)
axs[2].legend(frameon=False,fontsize=18)
fig.tight_layout()
plt.savefig("./figureS6/figureS6.pdf",bbox_inches="tight")
plt.show()

