#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calc_memory_kint.py

Calculates memory kernel for an adsorbate on top of a surface slab site by 
diagonalizing mass-weighted Hessian. Averages results across a periodically 
replicated surface along the 'a' and 'b' lattice vectors. 
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

from ase.phonons import Phonons
from ase.dft import kpoints, monkhorst_pack
from ase.spectrum.band_structure import BandStructure
from ase.spectrum.dosdata import RawDOSData

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
super_a, super_b = 6, 6 # supercell size
nk_a, nk_b = 10,10 # number of k-points along reciprocal lattice

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
output_dir = "./results/%s_%s_%s_%s_cell=%dx%dx%d_supercell=%dx%d/"%(adsorb_ele,
                surface_ele,calc_name,surface_gen.__name__,Na,Nb,Nc,super_a,super_b)

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
Nsupercell = Natom * super_a * super_b

# Reduced Mass
surfatom_mass = Atoms(surface_ele).get_masses()[0]
mu = adsorb_mass*surfatom_mass/(adsorb_mass+surfatom_mass)

# Create surface slab
atoms = surface_gen(surface_ele, size=(Na, Nb, Nc))
atoms.pbc=(True, True, False)
cell = atoms.cell
lat = cell.get_bravais_lattice()
print("Symmetry of surface ",lat)
print("Special Points ", list(lat.get_special_points()))

# Center and Apply PBC
atoms.center(axis=2, vacuum=0)
atoms.calc = calc
BFGS(atoms).run(fmax=0.005)
#print("Symmetry of surface after optimization ",atoms.cell.get_bravais_lattice())
#print("Special Points ", list(atoms.cell.get_bravais_lattice().get_special_points()))

# Adsorbate-Surface Frequency
k_ads = mu*freq_as**2/surfatom_mass

#%%
############ Calculate Extended Mass-Weighted Hessian

# Phonon calculator
ph = Phonons(atoms, calc, supercell=(super_a, super_b, 1), delta=0.01)
ph.indices = indices_free
ph.run()
ph.read(method='standard',acoustic=False)
ph.clean()
D_N = ph.D_N.copy()
atoms.set_cell(cell)

#%%
############ Calculate Memory Kernel

t_arr = np.arange(0,Nt*dt,dt)
freqs, modes = [], []
Kt_arr, Kw_arr, C_arr = [], [], []

# Loop through each surface atom and average over results
for i in range(len(indices_surface)):
    indx = indices_surface[i] - len(indices_fixed)
    print("Calculating Results for Atom ",indx)
    dof_indx = indx*3+2
    ph.D_N = D_N.copy()
    ph.D_N[0,dof_indx,dof_indx] = ph.D_N[0,dof_indx,dof_indx] + k_ads

    # Calculate modes/freqs over Monkhorst-Pack first brillouin zone
    k_grid  = monkhorst_pack((nk_a, nk_b, 1))
    freq_k, modes_k = ph.band_structure(k_grid, modes=True, verbose=True)
    modes_k = modes_k.reshape(modes_k.shape[0],modes_k.shape[1],-1)
    modes_k = modes_k/np.linalg.norm(modes_k,axis=2)[:,:,None]
    freq_k = lib.freq_converter(freq_k,'eV','radians/ps')

    # Calculate coupling C_i, spectral density Kw_i, and memory kernel Kt_i
    C_i = mu*(freq_as*lib.ase_units_ps)**2/np.sqrt(surfatom_mass * adsorb_mass) * modes_k[:,:,dof_indx]
    C_i = C_i/np.sqrt(np.size(modes_k,axis=0))
    Kw_i = np.where(freq_k > 0, C_i * C_i.conj()/freq_k**2, 0)
    Kt_i = []
    for t in t_arr:
        Kt_i.append(np.sum( Kw_i * np.cos( freq_k*t) ))
    Kt_i=np.array(Kt_i)
    Kw_i = Kw_i.flatten()
    
    freqs.append(freq_k)
    modes.append(modes_k)
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
spectral_dens = lib.smear_spectral_lorentz(Kw_ave, freqs.ravel(), freq_grid, 0.05)
plt.figure()
freq_grid = lib.freq_converter(freq_grid,'radians/ps','cm-1')
plt.plot(freq_grid,spectral_dens,color='k')

# Save Results
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

np.savez(output_dir+"/w=%dcm-1.npz"%(freq_as_cm),t_arr=t_arr,freq_arr=freqs,
         C_arr=C_arr,Kt_arr=Kt_arr,Kw_arr=Kw_arr)

