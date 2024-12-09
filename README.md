# asymmetric_ternary_mixtures

This repository contains code used to set up simulations of asymmetric ternary lipid membranes, as well as various analysis routines that can be performed on the resulting trajectories. The simulation engine employed is ESPResSoMD [https://espressomd.org/wordpress/, https://github.com/espressomd/espresso].

# File Descriptions

1. Lipid.py - lipid class file
2. mbtools.py - membrane set up functions
3. ternarymix_allvars.py - simulation setup file for an NPT simulation with three components. Needs mbtools.py and Lipid.py to run.
4. stress_analysis.py - calculates the differential stress, net stress and first moment of stress (along with respective error bars) following the Irving-Kirkwood formalism.
5. new_hexatic.py - calculates hexatic order parameter, nearest neighbor distances, specific area (voronoi). Also outputs the indices of lipids in the upper leaflet and lower leaflet (separately) and the indices of the stray lipids
6. hmm_full.py - uses the data generated by new_hexatic.py to perform the HMM analysis.


