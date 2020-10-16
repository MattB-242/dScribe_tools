#---------------------------------------------------------------------
#PACKAGE IMPORTS
#---------------------------------------------------------------------

import sys
import os
import gemmi
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dscribe.descriptors import SOAP
from dscribe.kernels import AverageKernel
from ase import Atoms
from ase.io import read

#---------------------------------------------------------------------
#SYSTEM PARAMETERS
#---------------------------------------------------------------------

# Check parameters
if (len(sys.argv) != 3):
    print("I require an input file!")
    sys.exit()
inputfile = sys.argv[1]
outputdir = sys.argv[2]


# Check that input file is appropriate
if not(os.path.isfile(inputfile)) :
    print("First parameter must be a file")
    sys.exit()

# Check that output directory is appropriate
if not(os.path.isdir(outputdir)) :
    print("Second parameter must be a directory")
    sys.exit()


#---------------------------------------------------------------------
#EXTRACT LIST OF FILES FROM INPUTDIR, MAKE FILENAMES
#AND READ INTO LIST
#---------------------------------------------------------------------

#Generate structure name from file
filename_split = inputfile.split("/")
name = str(filename_split[len(filename_split)-1][:-4])

#ase reader from inputfile
structure = read(inputfile)
ns = len(structure)

#Determine atomic species
species = set(structure.get_chemical_symbols())

print(f"Atomic species = {species}")

rbf_type = ['gto', 'polynomial']

#---------------------------------------------------------------------------
#RUN SOAP ACROSS RANGE OF NMAX VALUES AND PLOT ROWS, COLS, COMPUTATION TIME
#---------------------------------------------------------------------------   
xax = [i for i in range(1,10)]
lmax = 1
rcut = 10.0

polycol = []
gtocol = []
polytime = []
gtotime = []

for nmax in range(1,10):
    for rbf in rbf_type:
        soapgen_rcut = SOAP(species = species, rcut = rcut, nmax = nmax, lmax = lmax, periodic= True, sparse = False, rbf = rbf)
    
        tic_1 = time.perf_counter()
        soap_rcut = soapgen_rcut.create(structure)
        toc_1 = time.perf_counter()
    
        if rbf == 'gto':
            gtocol.append(len(soap_rcut[0]))
            gtotime.append(toc_1 - tic_1)
        else:
            polycol.append(len(soap_rcut[0]))
            polytime.append(toc_1 - tic_1)

rs = len(soap_rcut)
print(f"rows = {rs}")

vl = len(soap_rcut[0])
print(f'maximum vector length: {vl}')

polymax = polytime[-1]
gtomax = gtotime[-1]

print(f'Maximum polynomial computation time over nmax: {polymax:.2}s')
print(f'Maximum gaussian computation time over nmax: {gtomax:.2}s')

plt.plot(xax, gtotime, label = 'Gaussian RBF')
plt.plot(xax, polytime, label = 'Polynomial RBF')
plt.xlabel('Number of radial basis functions')
plt.ylabel('Computation time (s)')
plt.title(f'lmax = {lmax}, rcut = {rcut}')
plt.legend()

plt.savefig(outputdir+f"/{name}_time_lmax={lmax}_rcut={rcut}.png")
plt.cla()

plt.plot(xax, gtocol, label = 'Gaussian RBF')
plt.plot(xax, polycol, label = 'Polynomial RBF')
plt.xlabel('Number of radial basis functions')
plt.ylabel('Length of local descriptors')
plt.title(f'lmax = {lmax}, rcut = {rcut}')
plt.legend()

plt.savefig(outputdir+f"/{name}_coeffs_lmax={lmax}_rcut={rcut}.png")
plt.cla()

#---------------------------------------------------------------------------
#RUN SOAP ACROSS RANGE OF LMAX VALUES AND PLOT ROWS, COLS, COMPUTATION TIME
#---------------------------------------------------------------------------   
xax = [i for i in range(1,9)]
nmax = 1
rcut = 10.0

polycol = []
gtocol = []
polytime = []
gtotime = []

for lmax in range(1,9):
    for rbf in rbf_type:
        soapgen_rcut = SOAP(species = species, rcut = rcut, nmax = nmax, lmax = lmax, periodic= True, sparse = False, rbf = rbf)
    
        tic_1 = time.perf_counter()
        soap_rcut = soapgen_rcut.create(structure)
        toc_1 = time.perf_counter()
    
        if rbf == 'gto':
            gtocol.append(len(soap_rcut[0]))
            gtotime.append(toc_1 - tic_1)
        else:
            polycol.append(len(soap_rcut[0]))
            polytime.append(toc_1 - tic_1)
            
polymax = polytime[-1]
gtomax = gtotime[-1]

print(f'Maximum polynomial computation time over lmax: {polymax:.2}s')
print(f'Maximum gaussian computation time over lmax: {gtomax:.2}s')

plt.plot(xax, gtotime, label = 'Gaussian RBF')
plt.plot(xax, polytime, label = 'Polynomial RBF')
plt.xlabel('Degree of Spherical Harmonics')
plt.ylabel('Computation time (s)')
plt.title(f'nmax = {nmax}, rcut = {rcut} A')
plt.legend()

plt.savefig(outputdir+f"/{name}_time_nmax={nmax}_rcut={rcut}.png")
plt.cla()

plt.plot(xax, gtocol, label = 'Gaussian RBF')
plt.plot(xax, polycol, label = 'Polynomial RBF')
plt.xlabel('Degree of Spherical Harmonics')
plt.ylabel('Length of Local Descriptor')
plt.title(f'nmax = {nmax}, rcut = {rcut} A')
plt.legend()

plt.savefig(outputdir+f"/{name}_coeffs_nmax={nmax}_rcut={rcut}.png") 
plt.cla()

#---------------------------------------------------------------------------
#RUN SOAP ACROSS RANGE OF RCUT VALUES AND PLOT COMPUTATION TIME
#---------------------------------------------------------------------------   
lmax = 4
nmax = 4

rcutax = []
polycol = []
gtocol = []
polytime = []
gtotime = []

for rcut in np.linspace(2,15,20):
    rcutax.append(rcut)
    for rbf in rbf_type:
        soapgen_rcut = SOAP(species = species, rcut = rcut, nmax = nmax, lmax = lmax, periodic= True, sparse = False, rbf = rbf)
    
        tic_1 = time.perf_counter()
        soap_rcut = soapgen_rcut.create(structure)
        toc_1 = time.perf_counter()
    
        if rbf == 'gto':
            gtocol.append(len(soap_rcut[0]))
            gtotime.append(toc_1 - tic_1)
        else:
            polycol.append(len(soap_rcut[0]))
            polytime.append(toc_1 - tic_1)

polymax = polytime[-1]
gtomax = gtotime[-1]

print(f'Maximum polynomial computation time over rcut: {polymax:.2}s')
print(f'Maximum gaussian computation time over rcut: {gtomax:.2}s')

plt.plot(rcutax, gtotime, label = 'Gaussian RBF')
plt.plot(rcutax, polytime, label = 'Polynomial RBF')
plt.xlabel('r_cut (Angstroms)')
plt.ylabel('Computation time (s)')
plt.title(f'nmax = {nmax}, lmax = {lmax}')
plt.legend()

plt.savefig(outputdir+f"/{name}_time_nmax={nmax}_lmax={lmax}.png")
plt.cla()

plt.plot(xax, gtocol, label = 'Gaussian RBF')
plt.plot(xax, polycol, label = 'Polynomial RBF')
plt.xlabel('r_cut (Angstroms)')
plt.ylabel('Length of Local Descriptor')
plt.title('nmax = {nmax}, lmax = {lmax} A')
plt.legend()

plt.savefig(outputdir+f"/{name}_coeffs_nmax={nmax}_lmax={lmax}.png")
plt.cla()


