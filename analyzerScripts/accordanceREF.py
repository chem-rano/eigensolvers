import numpy as np
import os
from ttns2.state import loadTTNSFromHdf5
import util
import warnings
from ttns2.driver import bracket

# -------------------------------------------------------------
# Decsription: Calculation of accordance/ correspondance of 
# eigenvalues from "energies.dat" to actual eigenvalues, saved
# to states/wavefunctions
# -------------------------------------------------------------

D = 70
# ---------------- Reference TTNSs -----------------------------
refPath = f"/data/larsson/Eigen/RUNS/tns_D{D}/"
allRefE = np.loadtxt(refPath+"energies.dat",skiprows=1,usecols=(1)) # in au
numRef = len(allRefE)
# --------------- get actual energies --------------------------
actualE = np.empty(numRef,dtype=float)
for itree in range(numRef):
    filename = refPath + f"states/tns_{itree:05d}.h5"
    energy = loadTTNSFromHdf5(filename)[1]["energy"]
    actualE[itree] = energy
# ------------------ get corresponding energy -----------------
print("Eigenvalue (cm-1)\tClosest 5 eigenvalues (cm-1)")
outfile = open(f"AccordanceREF{D}.dat","w")
outfile.write("{:>10} {:>24} {:>10} {:>24}".format("Index(file)","Eigenvalue(file)",\
        "Index(WF)","Eigenvalue(WF)")+"\n")

seqMatch = None
mismatch = 0
for num in range(numRef):
    refE = allRefE[num]
    sorted_indices = np.argsort(abs(actualE-refE)) # safer option
    
    refEIncm = util.au2unit(refE,"cm-1")
    actualEIncm = util.au2unit(actualE[sorted_indices[0]],"cm-1")
    
    lines = "{:>10}".format(num)
    lines += "{:>24}".format(f"{refEIncm}")
    lines += "{:>10}".format(sorted_indices[0])
    lines += "{:>24}".format(f"{actualEIncm}"+"\n")
    outfile.write(lines)
    
    if num != sorted_indices[0]:
        mismatch += 1
    if mismatch == 1 and seqMatch == None:
        seqMatch = num

    np.set_printoptions(formatter={'float_kind': lambda x: "{:.12f}".format(x)})
    closest5 = np.array(util.au2unit(actualE[sorted_indices[:4]],"cm-1"))
    print(refEIncm,f"{closest5}") # to monitor

outfile.write(f"\nTotal consequtive states matching: {seqMatch}")
outfile.write(f"\nTotal mismatch: {mismatch}")
outfile.close()
# -------------------------------------------------------------
