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
if (len(sys.argv) != 3):
    print("I require an input directory!")
    sys.exit()
testfile = sys.argv[1]
outputdir = sys.argv[2]


# Check that input file is appropriate
if not(os.path.isfile(testfile)) :
    print("First parameter must be a file!")
    sys.exit()

if not(os.path.isdir(outputdir)):
    print("Second parameter must be a directory!")
    sys.exit()

#-----------------------------------------------------------------------------------------------
#FUNCTIONS TO RUN A COMPARISON BETWEEN EACH SOAP DESCRIPTOR IN A LIST AND ITS PREDECESSOR
#USING THE AVERAGE AND REMAX METHODS AND TO OUTPUT THE COMPARISON FIGURE INTO A NEW LIST
#-----------------------------------------------------------------------------------------------

#Average Kernel Method
def average_listcomp(desc_list):
    re = AverageKernel(metric = 'linear')
    av_comp_list = []
    loop_count = 0
    
    for i in range(0, len(desc_list)-1):
        comp_pair = [desc_list[i], desc_list[i+1][:,0:len(desc_list[i][0])]]
        print([len(comp_pair[0]), len(comp_pair[1])])
        print([len(comp_pair[0][0]), len(comp_pair[1][0])])
        kern = re.create(comp_pair)
        av_comp_list.append(kern[0][1])
        loop_count+=1
        print(f'done {loop_count} comparisons')

    return av_comp_list
        

#REMax Kernel Method       
def remax_listcomp(desc_list):
    re = REMatchKernel(metric = 'rbf', gamma = 1,alpha = 1, threshold = 1e-6)
    re_comp_list = []

    for i in range(0, len(desc_list)-1):
        comp_pair = [desc_list[i], desc_list[i+1][:,0:len(desc_list[i][0])]]
        norm_pair = [normalize(j) for j in comp_pair]
        kern = re.create(norm_pair)
        re_comp_list.append(kern[0][1])

    return re_comp_list      

#---------------------------------------------------------------------
#EXTRACT LIST OF FILES FROM INPUTDIR, MAKE FILENAMES
#AND READ INTO LIST
#---------------------------------------------------------------------

#Generate structure name from file
test_filename_split = testfile.split("/")
test_name = test_filename_split[-1][:-4]

#Read structure and species
structure = read(testfile)
species = ['C', 'H', 'O', 'N']


#-----------------------------------------------------------------------------------------------
#RUN STABILITY COMPARISON FOR RANGE OF RADIAL BASIS FUNCTIONS
#-----------------------------------------------------------------------------------------------   
print('starting RBF comparison')
#Set up plot axis and fixed parameters
xax = [i for i in range(1,9)]
lmax = 4
rcut = 20.0

#Make descriptor list
descriptors = []
for nmax in range(1,10):
    soapgen = SOAP(species = species, rcut = rcut, nmax = nmax, lmax = lmax, periodic= True, sparse = False, rbf = 'gto')
    descriptors.append(soapgen.create(structure))


#Make comparison list
kerndiffs = average_listcomp(descriptors)
remkerndiffs = remax_listcomp(descriptors)

#Plot kernel calculation by comparison
plt.plot(xax, kerndiffs, label = 'Average Kernel')
plt.plot(xax, remkerndiffs, label = 'REMatch Kernel')
plt.xlabel('Number of radial basis functions')
plt.ylabel('Kernel match')
plt.title(f'lmax = {lmax}, rcut = {rcut}')
plt.legend()

plt.savefig(outputdir+f"/{test_name}__kernstab_lmax={lmax}_rcut={rcut}.png")
plt.cla()
    
#-----------------------------------------------------------------------------------------------
#RUN SOAP ACROSS RANGE OF NMAX VALUES AND CALCULATE DIFFERENCE BETWEEN FIRST TERM OF DESCRIPTOR
#-----------------------------------------------------------------------------------------------   
print('starting spherical harmonic comparison')
#Set up plot axis and fixed parameters
xax = [i for i in range(1,9)]
nmax = 4
rcut = 20.0

#Make descriptor list
descriptors = []
for lmax in range(1,10):
    soapgen = SOAP(species = species, rcut = rcut, nmax = nmax, lmax = lmax, periodic= True, sparse = False, rbf = 'gto')
    descriptors.append(soapgen.create(structure))

#Make comparison list
kerndiffs = average_listcomp(descriptors)
remkerndiffs = remax_listcomp(descriptors)

#Plot kernel calculation by comparison
plt.plot(xax, kerndiffs, label = 'Average Kernel')
plt.plot(xax, remkerndiffs, label = 'REMatch Kernel')
plt.xlabel('Spherical Harmonic Degree')
plt.ylabel('Kernel match')
plt.title(f'nmax = {nmax}, rcut = {rcut}')
plt.legend()

plt.savefig(outputdir+f"/{test_name}_kernstab_nmax={nmax}_rcut={rcut}.png")
plt.cla()
#----------------------------------------------------------------------------------------
#RUN SOAP ACROSS RANGE OF RCUT VALUES AND CALCULATE DIFFERENCE BETWEEN FIRST TERM OF DESCRIPTOR
#----------------------------------------------------------------------------------------   
print('starting rcut comparison')
nmax = 4
lmax = 4
xax = []

descriptors = []
for rcut in np.linspace(2,20,30):
    xax.append(rcut)
    soapgen = SOAP(species = species, rcut = rcut, nmax = nmax, lmax = lmax, periodic= True, sparse = False, rbf = 'gto')
    descriptors.append(soapgen.create(structure))

#Make comparison list
clipax = xax[1:]
kerndiffs = average_listcomp(descriptors)
remkerndiffs = remax_listcomp(descriptors)
    
plt.plot(clipax, kerndiffs, label = 'Average Kernel')
plt.plot(clipax, remkerndiffs, label = 'REMatch Kernel')
plt.xlabel('r_cut (A)')
plt.ylabel('Kernel match')
plt.title(f'nmax = {nmax}, lmax = {lmax}')
plt.legend()

plt.savefig(outputdir+f"/{test_name}_kerstab_nmax={nmax}_lmax = {lmax}.png")
plt.cla()
