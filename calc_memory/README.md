Calculating memory kernel on adsorbates from harmonic approach.
======

All scripts in this directory are written in python and organized in cells. 

The first cell contains only imports and defines helpful functions. The second cell contains the parameters necessary to run the script. The remaining cells perform different numerical calculations.

The **results** subdirectory is made up of numpy zip archives which contain the results of the calculations run using the scripts in this directory.

Scripts
-----------

**calc_memory.py** - Script that calculates memory kernel on adsorbates by diagonalizing Hessian for a surface slab.

**calc_memory_kint.py** - Script that calculates memory kernel on adsorbates by diagonalizing Hessian and further averaging over the first BZ.

**calc_memory_pert.py** - Script that calculates memory kernel on adsorbates by perturbatively diagonalizing the Hessian.


