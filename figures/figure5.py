#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
figure5.py

Creates figure 5: Desorption Rates for CO and Xe
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import sys

# Library functions for project
sys.path.append("../lib/") if '../lib/' not in sys.path else False
import project_lib as lib
lib.init_niceplots()

kB = 0.008314
h = 0.399031
hbar = h/(2*np.pi)

#%%
############ Load Data From CSV File (Experimental Data)

data_CO = np.loadtxt("/Users/ardavan/PhD_projects/gle_surface/project_2/rate_theory/CO_des_data.csv",delimiter=",",skiprows=3,dtype='str')

data_CO_single = data_CO[0:11,0:2].astype(float)
data_CO_fast = data_CO[0:12,2:4].astype(float)
data_CO_slow = data_CO[0:12,4:6].astype(float)

data = np.loadtxt("/Users/ardavan/PhD_projects/gle_surface/project_2/rate_theory/Xe_des_data.csv",delimiter=",",skiprows=2,dtype='str').astype(float)
exp_T_Xe = 1000/data[:,0]
exp_k_Xe = data[:,1]

#%%
############ Load Memory Kernel and Set Parameters of Physical System

### Parameters of physical system
m_C = 12.01 # mass of C
m_O = 16.0  # mass of O
mu_CO = m_C * m_O / (m_C + m_O) # reduced mass of CO
m_tot = m_C + m_O # total mass of CO
r_CO = 0.1128 # length of CO bond
I_CO = mu_CO*r_CO**2 # moment of inertia
m_Pt = 195.084 #mass of Pt
mu_PtCO = m_tot * m_Pt / (m_tot + m_Pt) #reduced mass of PtCO

### Load Memory Kernel (phononic corrections)
Nx, Ny, Nz = 4,4,8 # Size of lattices
# path   = "/Users/ardavan/PhD_projects/gle_surface/project_2/ads_memory_theory/results/Pt111_cell=%dx%dx%d_unitcell/w=%scm-1.npz"%(Nx,Ny,Nz,480)
path   = "/Users/ardavan/PhD_projects/gle_surface/project_2/ads_memory_theory/results/Pt111_cell=%dx%dx%d_supercell=6x6/w=%scm-1.npz"%(Nx,Ny,Nz,480)
#path   = "../calc_memory/results/CO_Pt_EMT_fcc111_cell=4x4x8_supercell=6x6/w=%scm-1.npz"%(480)

npz = np.load(path)
Kt = npz['Kt_arr'].mean(axis=0)
Kw = npz['Kw_arr'].mean(axis=0)
freq = npz['freq_arr'].mean(axis=0)

freq_eff = np.sqrt(mu_PtCO/m_tot * lib.freq_converter(480.0,'cm-1','radians/ps')**2 - Kt[0])
print("Effective Surface-Adsorbate Frequency",freq_eff,"rad/ps^",lib.freq_converter(freq_eff,'radians/ps','cm-1'),"cm-1")
freq_max = freq.max()
print("Maximum phonon frequency",freq_max,"rad/ps",lib.freq_converter(freq_max,'radians/ps','cm-1'),"cm-1")

# Pt-Xe angular vibrational frequency, cm-1 to rad/ps
omega_Pt    =  lib.freq_converter(480.0,'cm-1','radians/ps')    # static frequency
omega_PtCO  =  freq_eff  # corrected frequency
omega_debye = lib.freq_converter(156.0,'cm-1','radians/ps') # bare debye frequency
omega_max   = freq_max # True maximum Frequency

# Barrier Energy, kJ/mol
E_bar = 1.47 * 96.485

#%%
############ Calculate TST rate constants for CO

### Set Temperature Ranges
temp_max = 730
temp_min = 600
temp_arr_CO = np.linspace(temp_min,temp_max,100)
Etemp_arr_CO = kB*temp_arr_CO
beta_arr_CO = 1/Etemp_arr_CO

### Calculate Partition Functions with various approximations

# Rotational/2D translational partition function
Q_trans  = np.sqrt(2*np.pi*m_tot*Etemp_arr_CO/(h**2))
Q_rot    = 8* np.pi**2 * I_CO * Etemp_arr_CO/(h**2)

# Vibrational Parition Functions using uncorrected frequency
Q_Pt_qm = 1/(1-np.exp(-beta_arr_CO*hbar*omega_Pt))
Q_Pt_cl = 1/(beta_arr_CO*hbar*omega_Pt)

# Vibrational Parition Functions using corrected frequency
Q_PtCO_qm = 1/(1-np.exp(-beta_arr_CO*hbar*omega_PtCO))
Q_PtCO_cl = 1/(beta_arr_CO*hbar*omega_PtCO)

cl_ratio = omega_max/omega_debye
qm_ratio = (1-np.exp(-beta_arr_CO*hbar*omega_max))/(1-np.exp(-beta_arr_CO*hbar*omega_debye))

### Calculate rate constants with various approximations

# Quantum TST rate constants
k_qm_ph_CO = 1/(beta_arr_CO * h) * Q_rot/Q_PtCO_qm * qm_ratio * np.exp(-beta_arr_CO*E_bar) 
k_qm_st_CO = 1/(beta_arr_CO * h) * Q_rot/Q_Pt_qm * np.exp(-beta_arr_CO*E_bar) 

# Classical TST rate constants
k_cl_ph_CO = 1/(beta_arr_CO * h) * Q_rot/Q_PtCO_cl * cl_ratio * np.exp(-beta_arr_CO*E_bar)
k_cl_st_CO = 1/(beta_arr_CO * h) * Q_rot/Q_Pt_cl * np.exp(-beta_arr_CO*E_bar) 

#%%
############ Load Memory Kernel and Set Parameters of Physical System

### Parameters of physical system
m_Pt = 195.084
m_Xe = 131.293
mu_PtXe = m_Xe * m_Pt / (m_Xe + m_Pt) #reduced mass of PtCO

### Load Memory Kernel (phononic corrections)
path   = "../calc_memory/results/Xe_Pt_EMT_fcc111_cell=4x4x8_supercell=6x6/w=%scm-1.npz"%(28)

npz = np.load(path)
Kt = npz['Kt_arr'].mean(axis=0)
Kw = npz['Kw_arr'].mean(axis=0)
freq = npz['freq_arr'].mean(axis=0)

freq_eff = np.sqrt(mu_PtXe/m_Xe * lib.freq_converter(28.3,'cm-1','radians/ps')**2 - Kt[0])
print("Effective Surface-Adsorbate Frequency",freq_eff,"rad/ps^",lib.freq_converter(freq_eff,'radians/ps','cm-1'),"cm-1")
freq_max = freq.max()
print("Maximum phonon frequency",freq_max,"rad/ps",lib.freq_converter(freq_max,'radians/ps','cm-1'),"cm-1")

# Pt-Xe angular vibrational frequency, cm-1 to rad/ps
omega_Pt   =  lib.freq_converter(28.3,'cm-1','radians/ps')  # static frequency
omega_PtXe =  freq_eff                                      # corrected frequency

omega_debye = lib.freq_converter(153.3,'cm-1','radians/ps') # bare debye frequency
omega_max   = freq_max # True maximum Frequency


# Barrier Energy, kJ/mol
E_bar = 0.245 * 96.485

#%%
############ Calculate TST rate constants for Xe

### Set Temperature Ranges
temp_max = exp_T_Xe.max()*1.25
temp_min = exp_T_Xe.min()*0.9
temp_arr_Xe = np.linspace(temp_min,temp_max,100)
Etemp_arr_Xe = kB*temp_arr_Xe
beta_arr_Xe = 1/Etemp_arr_Xe

### Parameters of Physical System
m_Pt = 195.084
m_Xe = 131.293

# Pt-Xe angular vibrational frequency, cm-1 to rad/ps
omega_Pt = lib.freq_converter(28.3,'cm-1','radians/ps')
omega_PtXe = lib.freq_converter(20.8,'cm-1','radians/ps')

# Barrier Energy, kJ/mol
E_bar = 0.245 * 96.485

### Calculate Partition Functions with various approximations

# Vibrational Parition Functions using uncorrected frequency
Q_Pt_qm = 1/(1-np.exp(-beta_arr_Xe*hbar*omega_Pt))
Q_Pt_cl = 1/(beta_arr_Xe*hbar*omega_Pt)

# Vibrational Parition Functions using corrected frequency
Q_PtXe_qm = 1/(1-np.exp(-beta_arr_Xe*hbar*omega_PtXe))
Q_PtXe_cl = 1/(beta_arr_Xe*hbar*omega_PtXe)

### Calculate rate constants with various approximations

# Quantum TST rate constants
k_qm_ph_Xe = 1/(beta_arr_Xe * h) * 1/Q_PtXe_qm  * np.exp(-beta_arr_Xe*E_bar)
k_qm_st_Xe = 1/(beta_arr_Xe * h) * 1/Q_Pt_qm * np.exp(-beta_arr_Xe*E_bar) 

# Classical TST rate constants
k_cl_ph_Xe =  1/(beta_arr_Xe * h) * 1/Q_PtXe_qm  * np.exp(-beta_arr_Xe*E_bar) 
k_cl_st_Xe =  1/(beta_arr_Xe * h) * 1/Q_Pt_qm * np.exp(-beta_arr_Xe*E_bar)

#%%
############ Plot Data 
from matplotlib.ticker import FormatStrFormatter

##### Plot CO Data and Format Axes
fig, ax = plt.subplots(nrows=1,ncols=1,figsize=[6,4.5])
s1 = ax.scatter(1000/data_CO_fast[:,0],     data_CO_fast[:,1]*np.log10(np.e),   color="black")
s2 = ax.scatter(1000/data_CO_single[:,0], data_CO_single[:,1]*np.log10(np.e),   color="grey", marker="s")
l1,= ax.plot(1000/temp_arr_CO,np.log10(k_cl_st_CO*10**12),linestyle="-",color="blue",linewidth=2.0, alpha=0.7)
l2,= ax.plot(1000/temp_arr_CO,np.log10(k_cl_ph_CO*10**12),linestyle="--",color="blue",linewidth=2.0, alpha=0.7)
l3,= ax.plot(1000/temp_arr_CO,np.log10(k_qm_st_CO*10**12),linestyle="-",color="red",linewidth=2.0, alpha=0.7)
l4,= ax.plot(1000/temp_arr_CO,np.log10(k_qm_ph_CO*10**12),linestyle="--",color="red",linewidth=2.0, alpha=0.7)


# format lower/left x/y axis
itemp_x = np.arange(1.39,1.70,0.09).round(2)
ax.set_xticks(itemp_x)
ax.set_xticklabels(itemp_x)
ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax.set_xlim(1.35,1.68)
ax.set_ylim(2.5,6)
ax.set_xlabel(" 1000/T (1/K)")
ax.set_ylabel(" $\log_{10}[ k\ (\mathrm{s}^{-1}) ]$ ")

leg = ax.legend((s1,s2),("Exp (terrace)","Exp (terrace + step)"),frameon=False,fontsize=15,loc=(0.42,0.75))
ax.add_artist(leg)
ax.legend((l1,l2,l3,l4),("$k_{d1}$","$k_{d2}$","$k_{d3}$","$k_{d4}$"),frameon=False,fontsize=18,ncol=2,loc=(0.44,0.54),columnspacing=0.8)

# create upper x-axis
ax1u = ax.twiny()  
itemp_x = np.arange(1.38,1.70,0.09).round(2)
temp_x = np.round(1000/itemp_x,-1).astype(int)
itemp_x = 1000/temp_x
ax1u.set_xticks(itemp_x)
ax1u.set_xticklabels(temp_x)
ax1u.set_xlim(1.35,1.68)
ax1u.set_xlabel(" T (K)")

# create right y-axis
ax1r = ax.twinx()  
ax1r.plot(1000/temp_arr_CO,k_qm_ph_CO*10**12,linestyle="--",color="red",linewidth=2.0,alpha=0.0)
ax1r.set_yscale('log')
ax1r.set_ylabel(" $k\ (\mathrm{s}^{-1}) $ ")
plt.savefig("./figure5/desorption_rates_CO.pdf",bbox_inches="tight")

##### Plot Xe Data and Format Axes
fig, ax = plt.subplots(nrows=1,ncols=1,figsize=[6,4.5])
ax.scatter(1000/(exp_T_Xe),exp_k_Xe,color="k",label="Exp")
ax.plot(1000/temp_arr_Xe,np.log10(k_cl_st_Xe*10**12),linestyle="-",color="blue",linewidth=2.0,alpha=0.4)
ax.plot(1000/temp_arr_Xe,np.log10(k_cl_ph_Xe*10**12),linestyle="--",color="blue",linewidth=2.0,alpha=0.4)
ax.plot(1000/temp_arr_Xe,np.log10(k_qm_st_Xe*10**12),linestyle="-",color="red",linewidth=2.0,alpha=0.4)
ax.plot(1000/temp_arr_Xe,np.log10(k_qm_ph_Xe*10**12),linestyle="--",color="red",linewidth=2.0,alpha=0.4)


# format lower/left x/y axis
itemp_x = np.arange(5.5,14.0,2.5).round(1)
ax.set_xticks(itemp_x)
ax.set_xticklabels(itemp_x)
ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.set_xlim(5.0,13.5)
ax.set_xlabel(" 1000/T (1/K)")
ax.set_ylabel(" $\log_{10}[ k\ (\mathrm{s}^{-1}) ]$ ")
ax.legend(frameon=False,fontsize=18)

# # create upper x-axis
ax2u = ax.twiny()  
itemp_x = np.arange(5.5,14.0,2.5).round(1)
temp_x = np.array([180,125,95,80])#np.round(1000/itemp_x,-1).astype(int)
itemp_x = 1000/temp_x
ax2u.set_xticks(itemp_x)
ax2u.set_xticklabels(temp_x)
ax2u.set_xlim(5.0,13.5)
ax2u.set_xlabel(" T (K)")

# create right y-axis
ax2r = ax.twinx()  
ax2r.plot(1000/temp_arr_Xe,np.log10(k_qm_ph_Xe*10**12),linestyle="--",color="red",linewidth=2.0,alpha=0.0)
ax2r.set_yscale('log')
ax2r.set_ylabel(" $k\ (\mathrm{s}^{-1}) $ ")

plt.savefig("./figure5/desorption_rates_Xe.pdf",bbox_inches="tight")