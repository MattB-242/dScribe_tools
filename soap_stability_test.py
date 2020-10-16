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
from dscribe.kernels import REMatchKernel
from ase import Atoms
from ase.io import read
from sklearn.preprocessing import normalize

#---------------------------------------------------------------------
#SYSTEM PARAMETERS
#---------------------------------------------------------------------

# Check parameters
if (len(sys.argv) != 4):
    print("I require an input directory!")
    sys.exit()
testfile = sys.argv[1]
compfile = sys.argv[2]
outputdir = sys.argv[3]


# Check that input file is appropriate
if not(os.path.isfile(testfile) or os.path.isfile(compfile)) :
    print("First and second parameters must be two files to compare")
    sys.exit()

if not(os.path.isdir(outputdir)):
    print("Third parameter must be a directory!")
    sys.exit()

#---------------------------------------------------------------------
#EXTRACT LIST OF FILES FROM INPUTDIR, MAKE FILENAMES
#AND READ INTO LIST
#---------------------------------------------------------------------

#Generate structure name from file
test_filename_split = testfile.split("/")
comp_filename_split = compfile.split("/")
test_name = test_filename_split[-1][:-4]
comp_name = comp_filename_split[-1][:-4]

structures = [read(testfile), read(compfile)]

species = ['C', 'H', 'O', 'N']
#----------------------------------------------------------------------------------------
#RUN SOAP ACROSS RANGE OF NMAX VALUES AND CALCULATE DIFFERENCE BETWEEN FIRST TERM OF DESCRIPTOR
#----------------------------------------------------------------------------------------   
xax = [i for i in range(1,15)]
lmax = 4
rcut = 20.0

descdiffs = []
kerndiffs = []
remkerndiffs = []
ctime = []
rectime=[]
allkerndiffs = []

for nmax in range(1,15):
    soapgen_rcut = SOAP(species = species, rcut = rcut, nmax = nmax, lmax = lmax, periodic= True, sparse = False, rbf = 'gto')
    descriptors = [soapgen_rcut.create(i) for i in structures]
    descdiffs.append(descriptors[1][0][0] - descriptors[0][0][0])

    tic_1 = time.perf_counter()
    re = AverageKernel(metric = 'linear')
    kern = re.create(descriptors)
    toc_1 = time.perf_counter()
    ctime.append(toc_1-tic_1)
    kerndiffs.append(kern[0][1])

    tic_2 = time.perf_counter()
    normed = [normalize(i) for i in descriptors]
    rem = REMatchKernel(metric = 'rbf', gamma = 1,alpha = 1, threshold = 1e-6)
    remkern = rem.create(descriptors)
    toc_2 = time.perf_counter()
    rectime.append(toc_2-tic_2)
    remkerndiffs.append(remkern[0][1])

    allkerndiffs.append(abs(remkern[0][1] - kern[0][1]))
    
plt.plot(xax, ctime, label = 'Average Kernel')
plt.plot(xax, rectime, label = 'REMatch Kernel')
plt.xlabel('Number of radial basis functions')
plt.ylabel('Kernel comparison time')
plt.title(f'lmax = {lmax}, rcut = {rcut}')
plt.legend()

plt.savefig(outputdir+f"/{test_name}_{comp_name}_kerncomp_lmax={lmax}_rcut={rcut}.png")
plt.cla()

plt.plot(xax, descdiffs)
plt.xlabel('Number of radial basis functions')
plt.ylabel('Difference between first term of first descriptor')
plt.title(f'lmax = {lmax}, rcut = {rcut}')
plt.legend()

plt.savefig(outputdir+f"/{test_name}_{comp_name}_descdiff_lmax={lmax}_rcut={rcut}.png")
plt.cla()

plt.plot(xax, kerndiffs, label = 'Average Kernel')
plt.plot(xax, remkerndiffs, label = 'REMatch Kernel')
plt.xlabel('Number of radial basis functions')
plt.ylabel('Kernel match')
plt.title(f'lmax = {lmax}, rcut = {rcut}')
plt.legend()

plt.savefig(outputdir+f"/{test_name}_{comp_name}_kernmatch_lmax={lmax}_rcut={rcut}.png")
plt.cla()
    
#----------------------------------------------------------------------------------------
#RUN SOAP ACROSS RANGE OF NMAX VALUES AND CALCULATE DIFFERENCE BETWEEN FIRST TERM OF DESCRIPTOR
#----------------------------------------------------------------------------------------   
xax = [i for i in range(1,10)]
nmax = 4
rcut = 20.0

descdiffs = []
kerndiffs = []
remkerndiffs = []
ctime = []
rectime=[]
allkerndiffs = []

for lmax in range(1,10):
    soapgen_rcut = SOAP(species = species, rcut = rcut, nmax = nmax, lmax = lmax, periodic= True, sparse = False, rbf = 'gto')
    descriptors = [soapgen_rcut.create(i) for i in structures]
    descdiffs.append(descriptors[1][0][0] - descriptors[0][0][0])

    tic_1 = time.perf_counter()
    re = AverageKernel(metric = 'linear')
    kern = re.create(descriptors)
    toc_1 = time.perf_counter()
    ctime.append(toc_1-tic_1)
    kerndiffs.append(kern[0][1])

    tic_2 = time.perf_counter()
    normed = [normalize(i) for i in descriptors]
    rem = REMatchKernel(metric = 'rbf', gamma = 1,alpha = 1, threshold = 1e-6)
    remkern = rem.create(descriptors)
    toc_2 = time.perf_counter()
    rectime.append(toc_2-tic_2)
    remkerndiffs.append(remkern[0][1])

    allkerndiffs.append(abs(remkern[0][1] - kern[0][1]))
    
plt.plot(xax, ctime, label = 'Average Kernel')
plt.plot(xax, rectime, label = 'REMatch Kernel')
plt.xlabel('Number of radial basis functions')
plt.ylabel('Kernel comparison time')
plt.title(f'nmax = {nmax}, rcut = {rcut}')
plt.legend()

plt.savefig(outputdir+f"/{test_name}_{comp_name}_kerncomp_nmax={nmax}_rcut={rcut}.png")
plt.cla()

plt.plot(xax, descdiffs)
plt.xlabel('Degree of Spherical Harmonics')
plt.ylabel('Difference between first term of first descriptor')
plt.title(f'nmax = {nmax}, rcut = {rcut}')

plt.savefig(outputdir+f"/{test_name}_{comp_name}_descdiff_nmax={nmax}_rcut={rcut}.png")
plt.cla()

plt.plot(xax, kerndiffs, label = 'Average Kernel')
plt.plot(xax, remkerndiffs, label = 'REMatch Kernel')
plt.xlabel('Degree of Spherical Harmonics')
plt.ylabel('Kernel match')
plt.title(f'lmax = {lmax}, rcut = {rcut}')
plt.legend()

plt.savefig(outputdir+f"/{test_name}_{comp_name}_kernmatch_nmax={nmax}_rcut={rcut}.png")
plt.cla()

#----------------------------------------------------------------------------------------
#RUN SOAP ACROSS RANGE OF RCUT VALUES AND CALCULATE DIFFERENCE BETWEEN FIRST TERM OF DESCRIPTOR
#----------------------------------------------------------------------------------------   
xax = [i for i in range(1,10)]
nmax = 4
lmax = 4
xax = []

descdiffs = []
kerndiffs = []
remkerndiffs = []
ctime = []
rectime=[]
allkerndiffs = []

for rcut in np.linspace(2,15,20):
    xax.append(rcut)
    soapgen_rcut = SOAP(species = species, rcut = rcut, nmax = nmax, lmax = lmax, periodic= True, sparse = False, rbf = 'gto')
    descriptors = [soapgen_rcut.create(i) for i in structures]
    descdiffs.append(descriptors[1][0][0] - descriptors[0][0][0])

    tic_1 = time.perf_counter()
    re = AverageKernel(metric = 'linear')
    kern = re.create(descriptors)
    toc_1 = time.perf_counter()
    ctime.append(toc_1-tic_1)
    kerndiffs.append(kern[0][1])

    tic_2 = time.perf_counter()
    normed = [normalize(i) for i in descriptors]
    rem = REMatchKernel(metric = 'rbf', gamma = 1,alpha = 1, threshold = 1e-6)
    remkern = rem.create(descriptors)
    toc_2 = time.perf_counter()
    rectime.append(toc_2-tic_2)
    remkerndiffs.append(remkern[0][1])

    allkerndiffs.append(abs(remkern[0][1] - kern[0][1]))
    
plt.plot(xax, ctime, label = 'Average Kernel')
plt.plot(xax, rectime, label = 'REMatch Kernel')
plt.xlabel('r_cut (A)')
plt.ylabel('Kernel comparison time')
plt.title(f'nmax = {nmax}, lmax = {lmax}')
plt.legend()

plt.savefig(outputdir+f"/{test_name}_{comp_name}_kerncomp_nmax={nmax}_lmax = {lmax}.png")
plt.cla()

plt.plot(xax, descdiffs)
plt.xlabel('r_cut (A)')
plt.ylabel('Difference between first term of first descriptor')
plt.title(f'nmax = {nmax}, lmax = {lmax}')

plt.savefig(outputdir+f"/{test_name}_{comp_name}_descdiff_nmax={nmax}_lmax = {lmax}.png")
plt.cla()

plt.plot(xax, kerndiffs, label = 'Average Kernel')
plt.plot(xax, remkerndiffs, label = 'REMatch Kernel')
plt.xlabel('r_cut (A)')
plt.ylabel('Kernel match')
plt.title(f'nmax = {nmax}, lmax = {lmax}')
plt.legend()

plt.savefig(outputdir+f"/{test_name}_{comp_name}_kernmatch_nmax={nmax}_lmax = {lmax}.png")
plt.cla()
