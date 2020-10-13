#---------------------------------------------------------------------
#PACKAGE IMPORTS
#---------------------------------------------------------------------

import sys
import os
import gemmi
import time
import pandas as pd
from dscribe.descriptors import SOAP
from dscribe.kernels import AverageKernel
from ase import Atoms
from ase.io import read

#---------------------------------------------------------------------
#SYSTEM PARAMETERS
#---------------------------------------------------------------------

# Check parameters
if (len(sys.argv) != 4):
    print("Must have 3 parameters: input dir, output dir, number of files")
    sys.exit()
inputdir = sys.argv[1]
outputdir = sys.argv[2]
n = sys.argv[3]

# Check they are directories
if not(os.path.isdir(inputdir) & os.path.isdir(outputdir)) :
    print("Parameters are not directories!")
    sys.exit()

#---------------------------------------------------------------------
#EXTRACT LIST OF FILES FROM INPUTDIR, MAKE FILENAMES
#AND READ INTO LIST
#---------------------------------------------------------------------
e = int(n)
files = list(gemmi.CifWalk(inputdir))[:e]

filename_split = [i.split("/") for i in files]
names = [str(i[len(i)-1][:-4]) for i in filename_split]

structures = [read(c) for c in files]

ns = len(structures)


#---------------------------------------------------------------------
#INITIATE PERIODIC SOAP DESCRIPTOR
#---------------------------------------------------------------------
species = ['C','H','O','N']
r_cut = 20.0
nmax = 16
lmax = 9

t2_per_soap = SOAP(species = species, rcut = r_cut, nmax = nmax, lmax = lmax, periodic= True, sparse = False)

#---------------------------------------------------------------------
#RUN SOAP ACROSS n FILES IN LIST AND OUTPUT COMPARISON KERNEL
#---------------------------------------------------------------------
tic_1 = time.perf_counter()
comparisons = [t2_per_soap.create(i) for i in structures]

metric = "linear"

re = AverageKernel(metric = metric)
kern = re.create(comparisons)

toc_1 = time.perf_counter()

comp_time = toc_1 - tic_1

print(f"Took {comp_time:.2} seconds to compare {ns} structures with r_cut = {r_cut:.2}, lmax = {lmax}, nmax = {nmax}")

#---------------------------------------------------------------------
#OUTPUT COMPARISON AS CSV FILE
#---------------------------------------------------------------------
soap_array = pd.DataFrame(kern, index = names, columns = names)
soap_array.to_csv(outputdir+"/soap_comparison_rcut = %s.csv" %r_cut, index = True, header = True, sep = ',')

