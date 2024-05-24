#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
figure3.py

Creates figure 3: phonon dispersion curves
"""

import os
import sys
import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

from ase import units
from ase.build import bulk, surface, fcc111, fcc110, fcc100, bcc111, bcc110, bcc100, hcp0001
from asap3 import EMT
from ase.phonons import Phonons
from ase.visualize import view
from ase.io import Trajectory, read, write, animation, xyz
from ase.optimize import BFGS
from ase.spectrum.band_structure import BandStructurePlot

# Library functions for project
sys.path.append("../lib/") if '../lib/' not in sys.path else False
import project_lib as lib

lib.init_niceplots()

def plot_ylines(ax,xcoords,ymax,ymin,**kwargs):
    for i in range(len(xcoords)):
        x = np.ones(2)*xcoords[i]
        y = [ymin,ymax]
        ax.plot(x,y,**kwargs)

#%%
############ 1) Bulk Dispersion

# Lattice Parameters 
Nx, Ny, Nz = 10,10,10   # Size of supercell
Nk = 100  # Number of k-points

lattice_ele  = "Pt"
lattice_sym = 'fcc'
calc = EMT()

# Setup crystal and EMT calculator
atoms = bulk(lattice_ele, lattice_sym)
atoms.calc = calc
cell = copy.deepcopy(atoms.cell)
lat = cell.get_bravais_lattice()
print("Symmetry of unit cell before optimization ",lat)
print("Special Points ", list(lat.get_special_points()))
BFGS(atoms).run(fmax=0.005)
print("Symmetry of unit cell after optimization ",atoms.cell.get_bravais_lattice())

ph = Phonons(atoms, calc, supercell=(Nx, Ny, Nz), delta=0.05)
ph.run()

# Read forces and assemble the dynamical matrix
ph.read(acoustic=False)
ph.clean()

# Plot the band structure
path = atoms.cell.bandpath('GXULG', npoints=Nk)
bs = ph.get_band_structure(path)
bs_cm = copy.deepcopy(bs)
energies_eV = bs._energies
energies_cm = 8065.540106923572 * bs._energies.copy()
bs_cm._energies = energies_cm
kcoords_bulk, special_pts_bulk, special_labels_bulk = bs_cm.get_labels()
fig,ax = plt.subplots(nrows=1,ncols=1,figsize=[7,4])
bs_cm.plot(ax=ax, ylabel=r"$\omega(k)$ (cm$^{-1})$", colors=["b"], emin=energies_cm.min()-5, emax=energies_cm.max()*1.1)

special_labels_bulk[0] = r"$\Gamma$"
special_labels_bulk[-1] = r"$\Gamma$"

#%%
############ 2) Surface Dispersion
path1   = "./dispersion/Pt111_cell=%dx%dx%d_supercell=6x6/w=%scm-1.npz"%(4,4,8,0)
path2   = "./dispersion/Pt111_cell=%dx%dx%d_supercell=6x6/w=%scm-1.npz"%(4,4,8,480)
paths = [path1,path2]

# Load Data
kcoords_arr = []
freqs_arr  = []
for i in range(len(paths)):
    path = paths[i]
    npz = np.load(path)
    
    kcoords = npz['kcoords']
    freqs = npz['freqs']
    special_pts = npz['special_pts']
    special_labels = npz['special_labels'].tolist()
    
    kcoords_arr.append(kcoords)
    freqs_arr.append(freqs)
    
special_labels[0] = r"$\Gamma$"
special_labels[-1] = r"$\Gamma$"


# Sort by slope
slopes_arr = []
for i in range(len(paths)):
    freqs = freqs_arr[i][0]
    kcoords = kcoords_arr[i]
    special_indx = np.argmin( np.abs(kcoords-special_pts[1]) )
    slopes = (freqs[special_indx,:]-freqs[0,:])/(kcoords[special_indx])
    slopes_arr.append(slopes)
    
#%%
############ 3) Plots
from mpl_toolkits.axes_grid1 import make_axes_locatable
cmap = cm.get_cmap('viridis',30).reversed()

fig, axs = plt.subplots(nrows=1,ncols=3,figsize=[13,5.0], sharey=True)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.4)

### Bulk Dispersion
axs[0].plot(kcoords_bulk, energies_cm[0], color=cmap(1.0), linestyle="-", linewidth=2.0)

#ylim = copy.deepcopy(axs[0].get_ylim())
ylim = (-10.10050925825251, 212.09534044011312)

# x-line
axs[0].plot(kcoords_bulk,np.zeros(len(kcoords_bulk)), color="black", linestyle=":", alpha=0.75, linewidth=1.0)
# y-line
plot_ylines(axs[0],special_pts_bulk, 0, ylim[1], color='black',alpha=0.5,linewidth=1.0)

# format axis
axs[0].set_ylim(ylim)
axs[0].set_xticks(special_pts_bulk)
axs[0].set_xticklabels(special_labels_bulk,fontsize=22)
axs[0].set_ylabel(r"$\omega(k)$ (cm$^{-1})$",fontsize=26)

### Surface Dispersion, 1

# Select Data
freqs = freqs_arr[0][0]
kcoords = kcoords_arr[0]
slopes = slopes_arr[0]

# Color scale
vmin = -3
vmax = np.max(slopes)
norm = Normalize(vmin=vmin, vmax=vmax)

P = norm(np.abs(slopes))
colors = cmap(P)

# Plot Lines
for i in range(len(freqs[0,:])):
    slope_i = slopes[i]
    if slope_i > 0:
        axs[1].plot(kcoords, freqs[:,i], color=colors[i], linestyle="-", linewidth=2.0)

# x-line
axs[1].plot(kcoords,np.zeros(len(kcoords)), color="black", linestyle=":", alpha=0.75, linewidth=1.0)

# y-line
plot_ylines(axs[1], special_pts, 0, ylim[1], color='black',alpha=0.5,linewidth=1.0)

# format axis
axs[1].set_ylim(ylim)
axs[1].set_xticks(special_pts)
axs[1].set_xticklabels(special_labels,fontsize=22)

### Surface Dispersion, 2

# Select Data
freqs = freqs_arr[1][0]
kcoords = kcoords_arr[1]
slopes = slopes_arr[1]

# Color scale
vmin = -3
vmax = np.max(slopes)
norm = Normalize(vmin=vmin, vmax=vmax)

P = norm(np.abs(slopes))
colors = cmap(P)

# Plot Lines
for i in range(len(freqs[0,:])):
    slope_i = slopes[i]
    if slope_i > -0.5:
        axs[2].plot(kcoords, freqs[:,i], color=colors[i], linestyle="-", linewidth=2.0)
ylim = copy.deepcopy(axs[2].get_ylim())

# x-line
axs[2].plot(kcoords,np.zeros(len(kcoords)), color="black", linestyle=":", alpha=0.75, linewidth=1.0)

# y-line
plot_ylines(axs[2], special_pts, 0, ylim[1], color='black',alpha=0.5,linewidth=1.0)

# format axis
axs[2].set_ylim(ylim)
axs[2].set_xticks(special_pts)
axs[2].set_xticklabels(special_labels,fontsize=22)


axs[0].set_title(r"Bulk Pt",fontsize=22,y=1.0)
axs[1].set_title(r"Bare surface Pt",fontsize=22,y=1.0)
axs[2].set_title(r"Surface Pt with adsorbed CO",fontsize=20,y=1.0)

#fig.tight_layout()
fig.tight_layout()
plt.savefig("./figure3/dispersions.pdf",bbox_inches="tight")
plt.show()


#%%
# Make colorbar
fig, ax = plt.subplots(figsize=(0.25, 9.0))
fig.subplots_adjust(bottom=0.5)
fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),
             cax=ax, orientation='vertical',label="$\Gamma$-$K$ slope")
fig.tight_layout()
plt.savefig("./figure3/colorbar.pdf",bbox_inches="tight")
plt.show()