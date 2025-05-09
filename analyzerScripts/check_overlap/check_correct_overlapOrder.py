import numpy as np
import os
import util
import warnings
import random
from ttns2.state import loadTTNSFromHdf5

# -----------------   Description ------------------------------------
# This tests output file data in order to check correct_overlapOrder.py 
# -------------------------------------------------------------------
D = 70

# ----- unaltered reference data ------------------------------------
refPath = f"/data/larsson/Eigen/RUNS/tns_D{D}/"
sortedE = np.loadtxt(refPath+"energies.dat",skiprows=1,usecols=(1)) # in au
num = len(sortedE)

unsortedE = np.empty(num,dtype=float)
for itree in range(num):
    filename = refPath + f"states/tns_{itree:05d}.h5"
    print(filename)
    energy = loadTTNSFromHdf5(filename)[1]["energy"]
    unsortedE[itree] = energy
sorted_indices = np.argsort(unsortedE)

# --------- unsorted overlap data ----------------------------------
file_unsorted = "unsorted/Overlap_it1_vec0.out"
overlapData = np.loadtxt(file_unsorted,usecols=(0,1,2,3,4,5),skiprows=1)

# --------- Load reference indices ---------------------------------------
refOrderFile = f"/home/madhumitarano/data/PR39/Eigen/forPaper/EigenRefOrder/AccordanceREF{D}.dat"
correctOrder = list(np.genfromtxt(refOrderFile,skip_header=1,skip_footer=2,usecols=(2),dtype=int))
numRef = len(correctOrder)
assert num == numRef 

# ---------------    Check1: ordering in correct_overlapOrder.py ----------
overlapData_indep3 = overlapData[:,3][sorted_indices]
overlapData_indep4 = overlapData[:,4][sorted_indices]

numRef = len(overlapData[:,0]) # some are truncated
correctOrder = correctOrder[0:numRef]

           
overlapData[:,3] = overlapData[:,3][correctOrder]
overlapData[:,4] = overlapData[:,4][correctOrder]

assert(np.allclose(overlapData_indep3,overlapData[:,3],atol=1e-4,rtol=0.0))
assert(np.allclose(overlapData_indep4,overlapData[:,4],atol=1e-4,rtol=0.0))
print("Check of ordering is successful")

# ---------------    Check2: ordered and written data same ----------
# -----Load data that are sorted by correct_overlapOrder.py ---------
filename = "sorted/Overlap_it1_vec0.out"
overlap = np.loadtxt(filename,usecols=(3),skiprows=1,dtype=float)
overlap2 = np.loadtxt(filename,usecols=(4),skiprows=1,dtype=float)
total = np.loadtxt(filename,usecols=(5),skiprows=1,dtype=float)[-1]

# That is no mismatch in cloumn/ variable name while writting
assert(np.allclose(overlap,overlapData[:,3],atol=1e-4,rtol=0.0))
assert(np.allclose(overlap2,overlapData[:,4],atol=1e-4,rtol=0.0))
print("Check of file is successful")
# -------------------------------------------------------------------
