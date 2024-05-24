import matplotlib.pyplot as plt
import numpy as np
from ase import units

##### Nice Plots
def init_niceplots(figsize=(8,6)):
    plt.rcParams["figure.figsize"] = figsize
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.sans-serif'] = ['Computer-Modern']
    plt.rcParams['axes.labelsize'] = 28
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['xtick.labelsize'] =  22
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.labelsize'] = 22
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['ytick.right'] = True
    plt.rcParams['legend.fontsize'] = 22
    plt.rcParams['figure.titlesize'] = 28
    plt.rcParams['lines.linewidth'] = 1.5
    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.preamble'] = r"\usepackage{amsmath} \usepackage{braket}"
    pass

##### Unit Conversions
kbT_eV = 0.02585  # 300K in eV
ase_units_ps = units.fs * 1e3 # 1 ps in ASE time units

def freq_converter(values, from_unit, to_unit):
    """
    Converts between common frequency units

    Possible unit names 'Hz' 's-1' 'THz' 'ps-1' 'radian/s' 'radian/ps' 'cm-1' 'eV'
    
    Parameters
    ----------
    values : Array/Int/Float
        Values to convert.
    from_unit : String
        name of units of input values.
    to_unit : String
        name of units of desired output.

    """
    
    unit_conversions = {
        'Hz': {
            'Hz': 1.0,
            's-1': 1.0,
            'THz': 1e-12,
            'ps-1': 1e-12,
            'radians/s': 2*np.pi,
            'radians/ps': 2*np.pi*1e-12,
            'cm-1': 3.3356e-11,
            'eV': 4.13567e-15
        },
        's-1': {
            'Hz': 1.0,
            's-1': 1.0,
            'THz': 1e-12,
            'ps-1': 1e-12,
            'radians/s': 2*np.pi,
            'radians/ps': 2*np.pi*1e-12,
            'cm-1': 3.3356e-11,
            'eV': 4.13567e-15
        },
        'THz': {
            'Hz': 1e12,
            's-1': 1e12,
            'THz': 1,
            'ps-1': 1,
            'radians/s': 2*np.pi*1e12,
            'radians/ps': 2*np.pi,
            'cm-1': 3.3356e1,
            'eV': 4.13567e-3
        },
        'ps-1': {
            'Hz': 1e12,
            's-1': 1e12,
            'THz': 1,
            'ps-1': 1,
            'radians/s': 2*np.pi*1e12,
            'radians/ps': 2*np.pi,
            'cm-1': 3.3356e1,
            'eV': 4.13567e-3
        },        
        'radians/s': {
            'Hz': 1.0/(2*np.pi),
            's-1': 1.0/(2*np.pi),
            'THz': 1e-12/(2*np.pi),
            'ps-1': 1e-12/(2*np.pi),
            'radians/s': 1,
            'radians/ps': 1e-12,
            'cm-1': 3.3356e-11/(2*np.pi),
            'eV': 4.13567e-15/(2*np.pi)
        },
        'radians/ps': {
            'Hz': 1e12/(2*np.pi),
            's-1': 1e12/(2*np.pi),
            'THz': 1/(2*np.pi),
            'ps-1': 1/(2*np.pi),
            'radians/s': 1e-12,
            'radians/ps': 1,
            'cm-1': 3.3356e1/(2*np.pi),
            'eV': 4.13567e-3/(2*np.pi)
        },
        'cm-1': {
            'Hz': 29979613862,
            's-1': 29979613862,
            'THz': 0.0299796138,
            'ps-1': 0.0299796138,
            'radians/s': 2*np.pi*29979613862,
            'radians/ps': 2*np.pi*0.0299796138,
            'cm-1': 1,
            'eV': 0.0001239842
        },
        'eV': {
            'Hz': 1/(4.13567e-15),
            's-1': 1/(4.13567e-15),
            'THz': 1/(4.13567e-3),
            'ps-1': 1/(4.13567e-3),
            'radians/s': 2*np.pi/(4.13567e-15),
            'radians/ps': 2*np.pi/(4.13567e-3),
            'cm-1': 8065.54,
            'eV': 1
        }
    }

    if from_unit == to_unit:
        return values

    if from_unit not in unit_conversions or to_unit not in unit_conversions[from_unit]:
        raise ValueError('Invalid conversion')

    conversion_factor = unit_conversions[from_unit][to_unit]
    converted_values = values * conversion_factor

    return converted_values

##### Helpful Functions for Computing/Visualizing Spectral Density 

def calc_spectral(signals, scale):
    """
    Calculate positive fast-fourier transform of a real, even signal.
    Scales output to the appropriate height. 
    """
    output = []
    for i in range(len(signals)):
        f_i = np.abs( np.real( np.fft.fft(signals[i]) ) ) /scale
        output.append( f_i[0:len(f_i)//2-1])
    return output

    
def smear_spectral_lorentz(weights, freqs, grid, width):
    """
    Smears spectral density across a set of Lorentzian's on a given frequency grid
    """
    output = 0
    for i in range(len(weights)):
        output = output + (1/np.pi) * weights[i] * ( width/(width**2 + (grid - freqs[i])**2) )
    return output

def smear_spectral_gaussian(weights, freqs, grid, width):
    """
    Smears spectral density across a set of Gaussians's on a given frequency grid
    """
    output = 0
    for i in range(len(weights)):
        output = output + (1/np.sqrt(2*np.pi*width**2)) * weights[i] * np.exp(-(freqs[i]-grid)**2/(2*width**2))
    return output

def plot_dirac_comb( freq, weight, color="k", cutoff=1.0, scaling=1.0,
                  linewidth=4.0, alpha=0.5, s=40, label="",
                  figsize=plt.rcParams["figure.figsize"]):
    """
    Plots a dirac comb (series of delta functions) as horizontal lines with points
    at the end.
    """
    first = True
    for j in range(len(freq)):
        freq_j = freq[j]
        x = [freq_j]*2
        y = [0, weight[j] * scaling]
        if y[1]> cutoff*scaling:
            if first:
                plt.plot(x,y, color=color, linewidth=linewidth, alpha=alpha, label=label )
                plt.scatter(freq_j, weight[j] * scaling, color=color, s=s)
                first = False
            else:
                plt.plot(x,y ,color=color, linewidth=linewidth, alpha=alpha)
                plt.scatter(freq_j, weight[j] * scaling, color=color, s=s)
        pass

##### Fitting Functions for Memory Kernels/Spectral Densities

def lorentz(coeff, decay, omega_0, w):
    """
    Lorentzian Function
    """
    f = coeff * ( decay/(decay**2 + (w - omega_0)**2) )
    return f

def damped_exp(coeff, decay, omega_0, t):
    """
    Damped Exponential Function
    """
    f = coeff * np.cos(omega_0*t) * np.exp(-decay*t)
    return f

def multiterm_lorentz(nterm):
    """
    Multiterm Lorentzian Function
    """
    
    def f_output(w_arr,*args):
        coeffs = args[0:nterm]
        decays = args[nterm:2*nterm]
        omegas = args[2*nterm:3*nterm]
        result = 0
        for i in range(nterm):
            result = result + lorentz(coeffs[i],decays[i],omegas[i],w_arr)
        return result
    
    return f_output
        
def multiterm_exp(nterm):
    """
    Multiterm Damped Exponential Function
    """
    
    def f_output(t_arr,*args):
        coeffs = args[0:nterm]
        decays = args[nterm:2*nterm]
        omegas = args[2*nterm:3*nterm]
        result = 0
        for i in range(nterm):
            result = result + damped_exp(coeffs[i],decays[i],omegas[i],t_arr)
        return result
    
    return f_output

def params_to_input(coeffs,decays,omegas):
    """
    Convert seperate parameters arrays to single input to function
    """
    input_args = []
    input_args.extend(coeffs)
    input_args.extend(decays)
    input_args.extend(omegas)
    return input_args

def input_to_params(input_args, nterm):
    """
    Convert single input to funciton to seperate parameters arrays
    """
    coeffs = input_args[0:nterm]
    decays = input_args[nterm:2*nterm]
    omegas = input_args[2*nterm:3*nterm]
    return coeffs,decays,omegas


if __name__=="__main__":
    pass
