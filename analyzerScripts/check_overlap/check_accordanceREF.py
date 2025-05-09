import numpy as np
import os
import util
import warnings
from ttns2.state import loadTTNSFromHdf5

# -------------------------------------------------------------------
# This is to check overlap order obtained from accordanceREF.py 
# It is not a mainstream test module related to eigensolvers codes,
# that is, it it being kept at "analyzerScripts/"
# Manual check is okay for eigenvalue to closest eigenvalue in output 
#(run.out) of accordanceREF.py 


# ---------------- Reference TTNSs order-----------------------------
def order_accordanceREF(D):
    refOrderFile = f"/home/madhumitarano/data/PR39/Eigen/forPaper/EigenRefOrder/AccordanceREF{D}.dat"
    correctOrder = list(np.genfromtxt(refOrderFile,skip_header=1,skip_footer=2,usecols=(2),dtype=int))

    return correctOrder

# ---------------- Reference TTNSs ----------------------------------
def ref_savedData(D):
    refPath = f"/data/larsson/Eigen/RUNS/tns_D{D}/"
    sortedE = np.loadtxt(refPath+"energies.dat",skiprows=1,usecols=(1)) # in au
    num = len(sortedE)
    
    unsortedE = np.empty(num,dtype=float)
    for itree in range(num):
        filename = refPath + f"states/tns_{itree:05d}.h5"
        print(filename)
        energy = loadTTNSFromHdf5(filename)[1]["energy"]
        unsortedE[itree] = energy

    return sortedE, unsortedE

Dim_list = [70,150]
for D in Dim_list:
    accordanceOrder = order_accordanceREF(D)
    elem, counts = np.unique(accordanceOrder, return_counts=True)
    dup = elem[counts > 1];print(dup)
    assert(len(dup) == 0)# no duplicates
    numRef = len(accordanceOrder)
    sortedE, unsortedE = ref_savedData(D)
    numRef2 = len(sortedE)
    sorted_indices = np.argsort(unsortedE)

    assert(numRef == numRef2)
    assert((accordanceOrder == sorted_indices).all())
    assert(np.allclose(unsortedE[accordanceOrder],sortedE,atol=1e-10,rtol=0.0))
    print(f"Test is successful for D{D}")
# -------------------------------------------------------------------
