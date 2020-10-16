# dScribe_tools

The various scripts in this toolset are intended to test the functionality of the dScribe descriptor comparisons, using elements of the T2 experimental and simulated dataset as a test. 

**soap_basic.py** is the fundamental script - given an input directory, output directory and integer as parameters it will find any CIF file in the input directory, read it in and compare the SOAP descriptors for the first n crystals in it pairwaise using the AVERAGE kernel. Parameters are hard-coded (note that the radial basis functions use the default Gaussian setting, so l_max can only be set up to nine)

**soap_param_test.py** outputs to the output directory plots of SOAP descriptor length and computation time across a range of the three basic input parameters (rcut = cut-off radius, lmax = degree of spherical harmonic expansion, nmax = number of radial basis functions) for a single input file which is required as an argument on running. 

**soap_stability_test** takes two CIF files and an output directory as inputs. It runs comparisons of:
(1) SOAP Kernel computation time for the REMatch and Average Kernel methodologies (with fixed alpha, gamma, threshold) across a range of r_cut, nmax, lmax
(2) SOAP Kernel comparison value between the two different input files using the REMatch and Average Kernel methodologies (with fixed alpha, gamma, threshold) across a range of r_cut, nmax, lmax
(3) Difference between the first value in the first SOAP descriptor across a range of r_cut, nmax, lmax




