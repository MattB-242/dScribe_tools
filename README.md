# dScribe_tools

The various scripts in this toolset are intended to test the functionality of the dScribe descriptor comparisons, using elements of the T2 experimental and simulated dataset as a test. 

**soap_basic.py** is the fundamental script - given an input directory, output directory and integer as parameters it will find any CIF file in the input directory, read it in and compare the first n crystals in it pairwaise using the AVERAGE kernel. Parameters are hard-coded (note that the radial basis functions use the default Gaussian setting, so l_max can only be set up to nine)

**soap_param_test.py** outputs to the output directory plots of descriptor length and computation time across a range of the three basic input parameters (rcut = cut-off radius, lmax = degree of spherical harmonic expansion, nmax = number of radial basis functions) for a single input file which is required as an argument on running. 

[TBD] **soap_stability_test** outputs to the output directory a plot of the maximum *absolute difference* between coefficients in the descriptors calculated for the same structure at different basic input parameters

[TBD] **soap_compare_time.py** outputs to the output directory a plot of time taken for a pairwise comparison of n input files, using the average



