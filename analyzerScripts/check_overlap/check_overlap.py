import numpy as np
import os
import util
import warnings
from ttns2.state import loadTTNSFromHdf5

# -------------------------------------------------------------------
# This is to check overlap files obtained from overlap.py
# It is not a mainstream test module related to eigensolvers codes,
# that is, it it being kept at "analyzerScripts/"

states = [[1,0],[2,1],[3,1]]

# Function1 : check overlap data-------------------------------------
def overlap_check(file1,file2,atol):
    overlapData1 = abs(np.loadtxt(file1,usecols=(3,4),skiprows=1))
    overlapData2 = abs(np.loadtxt(file2,usecols=(3,4),skiprows=1))

    ncols = overlapData1.shape[1]
    for i in range(ncols):
        assert(np.allclose(overlapData1[:,i],overlapData2[:,i],atol=atol,rtol=0.0))


# ------- Check1: Nonorthogonal data --------------------------------
#    (file1) Output from overlap.py 
#    (file2) Not from overlap.py (Separate calculation)

for state in states:
    it = state[0]; vec = state[1]
    file1 = f"nonortho/Overlap_it{it}_vec{vec}.out"
    file2 = f"sorted/Overlap_it{it}_vec{vec}.out"
    overlap_check(file1,file2,atol=1e-3)

print("Check of nonorthogonal overlap is successful")

# ------- Check2: Orthogonal data ----------------------------------
#  (file1) Output from overlap.py without supplying nonortho overlap
#  (file2) Output from overlap.py with supplying nonortho overlap---

for state in states:
    it = state[0]; vec = state[1]
    file1 = f"ortho/Overlap_it{it}_vec{vec}.out"
    file2 = f"ortho_using_saved_overlap/Overlap_it{it}_vec{vec}.out"
    overlap_check(file1,file2,5e-2)

print("Check of orthogonal overlap is successful")
# -------------------------------------------------------------------
