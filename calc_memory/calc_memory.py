#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calc_memory.py

Calculates memory kernel for an adsorbate on top of a surface slab site by 
diagonalizing mass-weighted Hessian. 
"""

# basic python imports, numpy and matplotlib
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# ASE imports
from ase import units, Atoms
from ase.build import bulk, surface, fcc111, fcc110, fcc100, bcc111, bcc110, bcc100, hcp0001
from ase.calculators.kim import KIM
from ase.calculators.lj import LennardJones

from ase.vibrations import Vibrations
from ase.optimize import BFGS

# ASAP 3 EMT calculator
from asap3 import EMT

# Library functions for project
sys.path.append("../lib/") if '../lib/' not in sys.path else False
import project_lib as lib


#%%
############ Script Parameters

##### Slab parameters 

Na, Nb, Nc = 4,4,8 # size of periodic surface slab
surface_ele  = "Pt" # element
surface_gen = fcc111 # slab generator function
orthogonal=False # make unit cell orthogonal


##### Surface forcefield parameters 

calc = EMT()
calc_name = "EMT"

## EAM forcefields

# calc = KIM('EAM_Dynamo_BonnyCastinTerentyev_2013_FeNiCr__MO_763197941039_000') #Fe, Cr
# calc = KIM('EAM_Dynamo_BonnyBakaevTerentyev_2017_WRe__MO_234187151804_000') #W
# calc = KIM('EAM_Dynamo_FortiniMendelevBuldyrev_2008_Ru__MO_114077951467_005') #Ru
# calc_name = "EAM"

## Lennard-Jones
# Source: https://doi-org.libproxy.mit.edu/10.1021/jp801931d
# sigma = 2.845 / (2**(1/6)) # Angstrom
# eps   = 0.33824            # eV
# calc = LennardJones(sigma=sigma, epsilon=eps, cutoff=15.0, smooth=False)
# calc_name = "LJ"

##### Adsorbate  parameters 

## CO
adsorb_ele  = "CO"
adsorb_mass = 28.01
freq_as_cm  = 500 # frequency of z motion in cm^-1

## Xe
# adsorb_ele  = "Xe"
# adsorb_mass = 131.293
# freq_as_cm  = 28.3 # frequency of z motion in cm^-1

## frequency of z motion in ASE freq units
freq_as  = lib.freq_converter(freq_as_cm,
                            'cm-1','radians/ps') / lib.ase_units_ps

## Output directory
output_dir = "./results/%s_%s_%s_%s_cell=%dx%dx%d/"%(adsorb_ele,surface_ele,calc_name,surface_gen.__name__,Na,Nb,Nc)

## Time Axes
dt = 0.020
Nt = 2500
#%%
############ Construct Surface

# Helpful variables
Natom = Na*Nb*Nc
indices_surface = np.arange(Na*Nb*(Nc-1), Na*Nb*Nc, 1) # surface sites indxs
Nsurface = len(indices_surface)
indices_fixed = [0]
indices_free = np.delete(np.arange(Natom),indices_fixed)
Nfree = len(indices_free)
Ndof = Nfree*3

surfatom_mass = Atoms(surface_ele).get_masses()[0]

# Reduced Ads-Surface mass
mu = adsorb_mass*surfatom_mass/(adsorb_mass+surfatom_mass) 

# Create surface slab
atoms = surface_gen(surface_ele, orthogonal=orthogonal, size=(Na, Nb, Nc))
atoms.pbc=(True, True, False)
cell = atoms.cell

# Center and optimizer
atoms.center(axis=2, vacuum=0)
atoms.calc = calc
BFGS(atoms).run(fmax=0.005)

# convert ads-surf frequency to harmonic force constant
k_ads = mu*freq_as**2/surfatom_mass

#%%
############ Calculate Mass-Weighted Hessian

vib = Vibrations(atoms,indices=indices_free, delta=0.01)
vib.run()
vib.read()
vib.clean()
D2_v = vib.im[:, None] * vib.H.copy() * vib.im  # Mass-Weighted Hessian

#%%
############ Calculate Memory Kernel

t_convert = (1e-12) / ( 1e-10 * np.sqrt(units._amu/units._e) )

t_arr = np.arange(0,Nt*dt,dt)
freqs, modes = [], []
Kt_arr, Kw_arr, C_arr = [], [], []

# Loop through each surface atom and average over results
for i in range(Nsurface):
    indx = indices_surface[i] - len(indices_fixed)
    print("Calculating Results for Atom ",indx)
    dof_indx = indx*3+2
    D2_i = D2_v.copy()
    D2_i[dof_indx,dof_indx] =  D2_i[dof_indx,dof_indx] + k_ads
    
    # Calculate frequencies and modes
    freq_v2, modes_v = np.linalg.eigh(D2_i)
    freq_v = np.where(freq_v2 > 0, np.sqrt(freq_v2)*t_convert, 0)
    
    # Calculate coupling and memory kernel
    C_i = mu*(freq_as*lib.ase_units_ps)**2/np.sqrt(surfatom_mass * adsorb_mass) * modes_v[dof_indx,:]
    Kw_i = np.where(freq_v > 0, C_i * C_i.conj()/freq_v**2, 0)
    Kt_i = []
    for t in t_arr:
        Kt_i.append(np.sum( Kw_i * np.cos( freq_v*t) ))
    Kt_i=np.array(Kt_i)
    Kw_i = Kw_i.flatten()
    
    # append
    freqs.append(freq_v)
    modes.append(modes_v)
    C_arr.append(C_i)
    Kt_arr.append(Kt_i)
    Kw_arr.append(Kw_i)
    
freqs = np.array(freqs)
modes = np.array(modes)
Kt_arr, Kw_arr = np.array(Kt_arr).real, np.array(Kw_arr).real
Kt_ave = np.mean(Kt_arr,axis=0)
Kw_ave = np.mean(Kw_arr,axis=0)

# Plot 
plt.figure()
plt.plot(t_arr,Kt_ave,color="k")
plt.xlim(-0.5,18)

freq_grid = np.linspace(0,freqs.max()*1.2,10000)
spectral_dens = lib.smear_spectral_lorentz(Kw_ave, freqs.ravel(), freq_grid, 0.10)
plt.figure()
freq_grid = lib.freq_converter(freq_grid,'radians/ps','cm-1')
plt.plot(freq_grid,spectral_dens,color='k')

# Save Results
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

np.savez(output_dir+"/w=%dcm-1.npz"%(freq_as_cm),t_arr=t_arr,freq_arr=freqs,
         C_arr=C_arr,Kt_arr=Kt_arr,Kw_arr=Kw_arr)