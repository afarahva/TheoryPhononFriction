#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calc_memory_pert.py

Calculates memory kernel for an adsorbate on top of a surface slab site by 
diagonalizing mass-weighted Hessian. 

This version does further perturbative analysis.
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

# GLEaPy (https://github.com/afarahva/gleqpy)
from gleqpy.memory.proj import BathProjection


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
freq_as_cm  = 480 # frequency of z motion in cm^-1

## Xe
# adsorb_ele  = "Xe"
# adsorb_mass = 131.293
# freq_as_cm  = 28.3 # frequency of z motion in cm^-1

# frequency of z motion in ASE freq units
freq_as  = lib.freq_converter(freq_as_cm,
                            'cm-1','radians/ps') / lib.ase_units_ps

## Output Directory
output_dir = "./results/%s_%s_%s_%s_cell=%dx%dx%d_pert/"%(adsorb_ele,
                surface_ele,calc_name,surface_gen.__name__,Na,Nb,Nc)

## Time Axes
dt = 0.020
Nt = 2500
#%%
############ Construct Lattice

# Helpful variables
Natom = Na*Nb*Nc
indices_surface = np.arange(Na*Nb*(Nc-1), Na*Nb*Nc, 1) # surface sites indxs
Nsurface = len(indices_surface)
indices_fixed = [0]
indices_free = np.delete(np.arange(Natom),indices_fixed)
Nfree = len(indices_free)
Ndof = Nfree*3

surfatom_mass = Atoms(surface_ele).get_masses()[0]

# Reduced Mass
mu = adsorb_mass*surfatom_mass/(adsorb_mass+surfatom_mass)

# Create surface slab
atoms = surface_gen(surface_ele, orthogonal=orthogonal, size=(Na, Nb, Nc))
atoms.pbc=(True, True, False)

# Center and Apply PBC
atoms.center(axis=2, vacuum=0)
atoms.calc = calc
BFGS(atoms).run(fmax=0.005)

# Adsorbate-Surface Frequency
k_ads = mu*freq_as**2/surfatom_mass
#%%
############ Calculate Mass-Weighted Hessian
t_convert = (1e-12) / ( 1e-10 * np.sqrt(units._amu/units._e) )

# Vibrations Calculator
vib = Vibrations(atoms,indices=indices_free, delta=0.01)
vib.run()
vib.read()
vib.clean()
D2_v = vib.im[:, None] * vib.H.copy() * vib.im 
#%%
############ Calculate memories kernels using exact diagonalization

t_arr = np.arange(0,Nt*dt,dt)
freqs_exact, modes_exact = [], []
Kt_exact, Kw_exact, C_exact = [], [], []

for i in range(Nsurface):
    indx = indices_surface[i] - len(indices_fixed)
    print("Calculating Exact Results for Atom ",indx)
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
    
    
    # Append
    freqs_exact.append(freq_v)
    modes_exact.append(modes_v)
    C_exact.append(C_i)
    Kt_exact.append(Kt_i)
    Kw_exact.append(Kw_i)
    
freqs_exact = np.array(freqs_exact)
modes_exact = np.array(modes_exact)
Kt_exact, Kw_exact = np.array(Kt_exact).real, np.array(Kw_exact).real
Kt_ave_exact = np.mean(Kt_exact,axis=0)
Kw_ave_exact = np.mean(Kw_exact,axis=0)

#%%
############ Calculate memories kernels using perturbation theory in the weak coupling limit

# 0th order results
freqs_weak_0, modes_weak_0 = [], []
Kt_weak_0, Kw_weak_0, C_weak_0 = [], [], []

# 1st order results
freqs_weak_1, modes_weak_1 = [], []
Kt_weak_1, Kw_weak_1, C_weak_1 = [], [], []

# 2nd order results
freqs_weak_2, modes_weak_2 = [], []
Kt_weak_2, Kw_weak_2, C_weak_2 = [], [], []

for i in range(Nsurface):
    indx = indices_surface[i] - len(indices_fixed)
    dof_indx = indx*3+2
    print("Calculating weak coupling perturbation results for Atom ",indx)
    
    # unperturbed Hessian 
    H_0 = D2_v.copy()
    
    # perturbation
    H_1  = np.zeros(H_0.shape)
    H_1[dof_indx,dof_indx] =  H_1[dof_indx,dof_indx] + k_ads
    
    # 0th order evecs/evals
    freq2_0, modes_0 = np.linalg.eigh(H_0)
    freq_0 = np.where(freq2_0 > 0, np.sqrt(freq2_0)*t_convert, 0)
    
    # 1st order evecs/evals
    freq2_1 = freq2_0.copy()
    for j in range(Ndof):
        freq2_1[j] += modes_0[:,j:j+1].T @ H_1 @ modes_0[:,j:j+1]
    freq_1 =  np.where(freq2_1 > 0, np.sqrt(freq2_1)*t_convert, 0)
    
    modes_1 = modes_0.copy()
    for j in range(Ndof):
        for k in range(Ndof):
            if j==k:
                continue
            else:
                coeff = (modes_0[:,k].T @ H_1 @ modes_0[:,j]) / (freq2_0[j] - freq2_0[k])
                modes_1[:,j] += coeff*modes_0[:,k]
        modes_1[:,j] = modes_1[:,j]/np.linalg.norm(modes_1[:,j])
        
    # 2nd order evals
    freq2_2 = freq2_1.copy()
    for j in range(Ndof):
        for k in range(Ndof):
            if j==k:
                continue
            else:
                freq2_2[j] += np.abs(modes_0[:,k].T @ H_1 @ modes_0[:,j])**2 / (freq2_0[j] - freq2_0[k])
    freq_2 =  np.where(freq2_2 > 0, np.sqrt(freq2_2)*t_convert, 0)
                
    # 0th order C/Kt/Kw
    C_i0 = mu*(freq_as*lib.ase_units_ps)**2/np.sqrt(surfatom_mass * adsorb_mass) * modes_0[dof_indx,:]
    Kw_i0 = np.where(freq_0 > 0, C_i0 * C_i0.conj()/freq_0**2, 0)
    Kt_i0 = []
    for t in t_arr:
        Kt_i0.append(np.sum( Kw_i0 * np.cos( freq_0*t) ))
    Kt_i0 = np.array(Kt_i0)
    Kw_i0 = Kw_i0.flatten()
    
    freqs_weak_0.append(freq_0)
    modes_weak_0.append(modes_0)
    C_weak_0.append(C_i0)
    Kt_weak_0.append(Kt_i0.real)
    Kw_weak_0.append(Kw_i0.real)
    
    # 1st order C/Kt/Kw
    C_i1 = mu*(freq_as*lib.ase_units_ps)**2/np.sqrt(surfatom_mass * adsorb_mass) * modes_1[dof_indx,:]
    Kw_i1 = np.where(freq_1 > 0, C_i1 * C_i1.conj()/freq_1**2, 0)
    Kt_i1 = []
    for t in t_arr:
        Kt_i1.append(np.sum( Kw_i1 * np.cos( freq_1*t) ))
    Kt_i1 = np.array(Kt_i1)
    Kw_i1 = Kw_i1.flatten()
    
    freqs_weak_1.append(freq_1)
    modes_weak_1.append(modes_1)
    C_weak_1.append(C_i1)
    Kt_weak_1.append(Kt_i1.real)
    Kw_weak_1.append(Kw_i1.real)

    # 2nd order C/Kt/Kw
    C_i2 = C_i1.copy()
    Kw_i2 = np.where(freq_2 > 0, C_i2 * C_i2.conj()/freq_2**2, 0)
    Kt_i2 = []
    for t in t_arr:
        Kt_i2.append(np.sum( Kw_i2 * np.cos( freq_2*t) ))
    Kt_i2 = np.array(Kt_i2)
    Kw_i2 = Kw_i2.flatten()
    
    freqs_weak_2.append(freq_2)
    modes_weak_2.append(modes_1)
    C_weak_2.append(C_i2)
    Kt_weak_2.append(Kt_i2.real)
    Kw_weak_2.append(Kw_i2.real)

# average over arrays
freqs_weak_0,freqs_weak_1,freqs_weak_2 = map(np.array,[freqs_weak_0,freqs_weak_1,freqs_weak_2])
modes_weak_0,modes_weak_1,modes_weak_2 = map(np.array,[modes_weak_0,modes_weak_1,modes_weak_2])
Kt_weak_0,Kt_weak_1,Kt_weak_2 = map(np.array,[Kt_weak_0,Kt_weak_1,Kt_weak_2])
Kw_weak_0,Kw_weak_1,Kw_weak_2 = map(np.array,[Kw_weak_0,Kw_weak_1,Kw_weak_2])

Kt_ave_weak0,Kt_ave_weak1,Kt_ave_weak2 = np.mean(Kt_weak_0,axis=0), np.mean(Kt_weak_1,axis=0), np.mean(Kt_weak_2,axis=0)
Kw_ave_weak0,Kw_ave_weak1,Kw_ave_weak2 = np.mean(Kw_weak_0,axis=0), np.mean(Kw_weak_1,axis=0), np.mean(Kw_weak_2,axis=0)
#%%
############ Calculate memories kernels using perturbation theory in the strong coupling limit

# 0th order results
freqs_strong_0, modes_strong_0 = [], []
Kt_strong_0, Kw_strong_0, C_strong_0 = [], [], []

# 1st order results
freqs_strong_1, modes_strong_1 = [], []
Kt_strong_1, Kw_strong_1, C_strong_1 = [], [], []

# 2nd order results
freqs_strong_2, modes_strong_2 = [], []
Kt_strong_2, Kw_strong_2, C_strong_2 = [], [], []

for i in range(Nsurface):
    indx = indices_surface[i] - len(indices_fixed)
    dof_indx = indx*3+2
    print("Calculating strong coupling perturbation results for Atom ",indx)
    
    # Diagonalize bath Hamiltonian
    bp = BathProjection(D2_v,1.0,[dof_indx])
    bp.diagonalize_symm()
    freq2_b, C = bp.freq2, bp.iC
    
    # unperturbed Hamiltonian - site and environment modes are uncoupled
    H_0  = np.zeros(D2_v.shape)
    H_0[0,0] =  k_ads + D2_v[dof_indx,dof_indx]
    H_0[1:,1:] = np.diag(freq2_b)
    
    # perturbed Hamiltonian - coupling between site and other modes
    H_1        = np.zeros(D2_v.shape)
    H_1[0,1:]   = C
    H_1[1:,0]   = C
    
    # 0th order evecs/evals
    freq2_0, modes_0 = np.diag(H_0), np.eye(Ndof)
    freq_0 = np.where(freq2_0 > 0, np.sqrt(freq2_0)*t_convert, 0)
    
    # 1st order evecs/evals
    freq2_1 = freq2_0.copy()
    for j in range(Ndof):
        freq2_1[j] += modes_0[:,j:j+1].T @ H_1 @ modes_0[:,j:j+1]
    freq_1 =  np.where(freq2_1 > 0, np.sqrt(freq2_1)*t_convert, 0)
    
    modes_1 = modes_0.copy()
    for j in range(Ndof):
        for k in range(Ndof):
            if j==k:
                continue
            else:
                coeff = (modes_0[:,k].T @ H_1 @ modes_0[:,j]) / (freq2_0[j] - freq2_0[k])
                modes_1[:,j] += coeff*modes_0[:,k]
        modes_1[:,j] = modes_1[:,j]/np.linalg.norm(modes_1[:,j])
        
    # 2nd order evals
    freq2_2 = freq2_1.copy()
    for j in range(Ndof):
        for k in range(Ndof):
            if j==k:
                continue
            else:
                freq2_2[j] += np.abs(modes_0[:,k].T @ H_1 @ modes_0[:,j])**2 / (freq2_0[j] - freq2_0[k])
    freq_2 =  np.where(freq2_2 > 0, np.sqrt(freq2_2)*t_convert, 0)
        
    # 0th order C/Kt/Kw
    C_i0 = mu*(freq_as*lib.ase_units_ps)**2/np.sqrt(surfatom_mass * adsorb_mass) * modes_0[0,:]
    Kw_i0 = np.where(freq_0 > 0, C_i0 * C_i0.conj()/freq_1**2, 0)
    Kt_i0 = []
    for t in t_arr:
        Kt_i0.append(np.sum( Kw_i0 * np.cos( freq_0*t) ))
    Kt_i0 = np.array(Kt_i0)
    Kw_i0 = Kw_i0.flatten()
    
    freqs_strong_0.append(freq_0)
    modes_strong_0.append(modes_0)
    C_strong_0.append(C_i0)
    Kt_strong_0.append(Kt_i0.real)
    Kw_strong_0.append(Kw_i0.real)

    # 1st order C/Kt/Kw
    C_i1 = mu*(freq_as*lib.ase_units_ps)**2/np.sqrt(surfatom_mass * adsorb_mass) * modes_1[0,:]
    Kw_i1 = np.where(freq_1 > 0, C_i1 * C_i1.conj()/freq_1**2, 0)
    Kt_i1 = []
    for t in t_arr:
        Kt_i1.append(np.sum( Kw_i1 * np.cos( freq_1*t) ))
    Kt_i1 = np.array(Kt_i1)
    Kw_i1 = Kw_i1.flatten()
    
    freqs_strong_1.append(freq_1)
    modes_strong_1.append(modes_1)
    C_strong_1.append(C_i1)
    Kt_strong_1.append(Kt_i1.real)
    Kw_strong_1.append(Kw_i1.real)
    
    # 2nd order C/Kt/Kw
    C_i2 = C_i1.copy()
    Kw_i2 = np.where(freq_2 > 0, C_i2 * C_i2.conj()/freq_2**2, 0)
    Kt_i2 = []
    for t in t_arr:
        Kt_i2.append(np.sum( Kw_i2 * np.cos( freq_2*t) ))
    Kt_i2 = np.array(Kt_i2)
    Kw_i2 = Kw_i2.flatten()
    
    freqs_strong_2.append(freq_2)
    modes_strong_2.append(modes_1)
    C_strong_2.append(C_i2)
    Kt_strong_2.append(Kt_i2.real)
    Kw_strong_2.append(Kw_i2.real)

freqs_strong_0,freqs_strong_1,freqs_strong_2 = map(np.array,[freqs_strong_0,freqs_strong_1,freqs_strong_2])
modes_strong_0,modes_strong_1,modes_strong_2 = map(np.array,[modes_strong_0,modes_strong_1,modes_strong_2])
Kt_strong_0,Kt_strong_1,Kt_strong_2 = map(np.array,[Kt_strong_0,Kt_strong_1,Kt_strong_2])
Kw_strong_0,Kw_strong_1,Kw_strong_2 = map(np.array,[Kw_strong_0,Kw_strong_1,Kw_strong_2])

Kt_ave_strong0,Kt_ave_strong1,Kt_ave_strong2 = np.mean(Kt_strong_0,axis=0), np.mean(Kt_strong_1,axis=0), np.mean(Kt_strong_2,axis=0)
Kw_ave_strong0,Kw_ave_strong1,Kw_ave_strong2 = np.mean(Kw_strong_0,axis=0), np.mean(Kw_strong_1,axis=0), np.mean(Kw_strong_2,axis=0)

#%%
############ Make plots and save

### Plot 

# Exact vs weak coupling
plt.figure()
plt.plot(t_arr,Kt_ave_exact,color="k",linewidth=4.0,alpha=0.5)
plt.plot(t_arr,Kt_ave_weak0,color="b",linewidth=2.0,alpha=0.5)
plt.plot(t_arr,Kt_ave_weak1,color="g",linewidth=2.0,alpha=0.5)
plt.plot(t_arr,Kt_ave_weak2,color="r",linewidth=2.0,alpha=0.5)
plt.xlim(-0.5,18)

freq_grid = np.linspace(0,freqs_exact.max()*1.2,10000)
spectral_densE = lib.smear_spectral_lorentz(Kw_ave_exact, freqs_exact.ravel(), freq_grid, 0.10)
spectral_dens0 = lib.smear_spectral_lorentz(Kw_ave_weak0, freqs_weak_0.ravel(), freq_grid, 0.10)
spectral_dens1 = lib.smear_spectral_lorentz(Kw_ave_weak1, freqs_weak_1.ravel(), freq_grid, 0.10)
spectral_dens2 = lib.smear_spectral_lorentz(Kw_ave_weak2, freqs_weak_2.ravel(), freq_grid, 0.10)
freq_grid = lib.freq_converter(freq_grid,'radians/ps','cm-1')

plt.figure()
plt.plot(freq_grid,spectral_densE,color='k',linewidth=4.0,alpha=0.5)
plt.plot(freq_grid,spectral_dens0,color='b',linewidth=2.0,alpha=0.5)
plt.plot(freq_grid,spectral_dens1,color='g',linewidth=2.0,alpha=0.5)
plt.plot(freq_grid,spectral_dens2,color='r',linewidth=2.0,alpha=0.5)

# Exact vs strong coupling
plt.figure()
plt.plot(t_arr,Kt_ave_exact,color="k",linewidth=4.0,alpha=0.5)
plt.plot(t_arr,Kt_ave_strong0,color="b",linewidth=2.0,alpha=0.5)
plt.plot(t_arr,Kt_ave_strong1,color="g",linewidth=2.0,alpha=0.5)
plt.plot(t_arr,Kt_ave_strong2,color="r",linewidth=2.0,alpha=0.5)
plt.xlim(-0.5,18)

freq_grid = np.linspace(0,freqs_exact.max()*1.2,10000)
spectral_densE = lib.smear_spectral_lorentz(Kw_ave_exact, freqs_exact.ravel(), freq_grid, 0.10)
spectral_dens0 = lib.smear_spectral_lorentz(Kw_ave_strong0, freqs_strong_0.ravel(), freq_grid, 0.10)
spectral_dens1 = lib.smear_spectral_lorentz(Kw_ave_strong1, freqs_strong_1.ravel(), freq_grid, 0.10)
spectral_dens2 = lib.smear_spectral_lorentz(Kw_ave_strong2, freqs_strong_2.ravel(), freq_grid, 0.10)
freq_grid = lib.freq_converter(freq_grid,'radians/ps','cm-1')

plt.figure()
plt.plot(freq_grid,spectral_densE,color='k',linewidth=4.0,alpha=0.5)
plt.plot(freq_grid,spectral_dens0,color='b',linewidth=2.0,alpha=0.5)
plt.plot(freq_grid,spectral_dens1,color='g',linewidth=2.0,alpha=0.5)
plt.plot(freq_grid,spectral_dens2,color='r',linewidth=2.0,alpha=0.5)

# Save Results
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

np.savez(output_dir+"/w=%dcm-1.npz"%(freq_as_cm),t_arr=t_arr,
         Kt_ave_exact=Kt_ave_exact,Kw_ave_exact=Kw_ave_exact,freqs_exact=freqs_exact.ravel(),
         Kt_ave_strong0=Kt_ave_strong0,Kt_ave_strong1=Kt_ave_strong1,Kt_ave_strong2=Kt_ave_strong2,
         Kw_ave_strong0=Kw_ave_strong0,Kw_ave_strong1=Kw_ave_strong1,Kw_ave_strong2=Kw_ave_strong2,
         freqs_strong_0=freqs_strong_0.ravel(),
         freqs_strong_1=freqs_strong_1.ravel(),
         freqs_strong_2=freqs_strong_2.ravel(),
         Kt_ave_weak0=Kt_ave_weak0,Kt_ave_weak1=Kt_ave_weak1,Kt_ave_weak2=Kt_ave_weak2,
         Kw_ave_weak0=Kw_ave_weak0,Kw_ave_weak1=Kw_ave_weak1,Kw_ave_weak2=Kw_ave_weak2,
         freqs_weak_0=freqs_weak_0.ravel(),
         freqs_weak_1=freqs_weak_1.ravel(),
         freqs_weak_2=freqs_weak_2.ravel())
