import numpy as np
import os
from ttns2.state import loadTTNSFromHdf5
import util
import warnings
from ttns2.driver import bracket

# -------------------- Parameters -----------------------------
startCum = 1
maxCum = 3 # input 

path = "demo/saveTNSs/" 
filename = path + f"tns_{startCum}_0.h5"
status = loadTTNSFromHdf5(filename)[1]["status"]
nBlock = status["nBlock"]
# ---------------- Reference TTNSs -----------------------------
refPath = "/data/larsson/Eigen/RUNS/tns_D70/"
allRefE = np.loadtxt(refPath+"energies.dat",skiprows=1,usecols=(1)) # in au
numRef = len(allRefE)
# ------------------ sum of overlap over all references------
for i in range(startCum,maxCum+1):
    
       
    Ylist = [] # emptied KS list at begining of each iteration 
    # Krylov vector at cum iteration
    for l in range(1000): # 1000 sufficiently large
        try:
            filename = path + "tns_"+str(i)+"_"+str(l)+".h5"
            Ylist.append(loadTTNSFromHdf5(filename)[0])
            coeff = loadTTNSFromHdf5(filename)[1]["eigencoefficients"]
            eigenvalues = loadTTNSFromHdf5(filename)[1]["eigenvalues"]
        except FileNotFoundError:
            break
    
    mvectors = coeff.shape[1]
    assert(mvectors == len(Ylist))
    print("length of Ylist",len(Ylist))
    # Krylov vector at cum iteration
    for vec in range(0,mvectors):
        
        evIncm = util.au2unit(eigenvalues[vec],"cm-1")
    
        #overlap_file = open(f"check_overlap/data/Overlap_it{i}_vec{vec}.out","w")
        overlap_file = open(f"check_overlap/data/Overlap_it{i}_vec{vec}_unsorted.out","w")
        lines = "{:>6} {:>16} {:>16} {:>16} {:>16} {:>16}".format("Index","RefE",\
                "eigenvalue","overlap","overlap-squared","Total")
        lines += "\n"
        overlap_file.write(lines)

        total = 0.0
        for num in range(numRef):
            overlap = 0.0
            overlap2 = 0.0

            refEIncm = util.au2unit(allRefE[num],"cm-1")
            filename = refPath + f"states/tns_{num:05d}.h5"
            print(refEIncm,filename)
            refTree = loadTTNSFromHdf5(filename)[0]
        
        
            lines = "{:>6}".format(num)
            lines += "{:>16}".format(f"{refEIncm:.6f}")
            lines += "{:>16}".format(f"{evIncm:.6f}")

            for k in range(len(Ylist)):
                overlap += bracket(refTree, Ylist[k])*coeff[k,vec]
            overlap2 = overlap*overlap
            total += overlap2
        
            lines += "{:>16}".format(f"{overlap:.4f}")
            lines += "{:>16}".format(f"{overlap2:.4f}")
            lines += "{:>16}".format(f"{total:.4f}")+"\n"
            overlap_file.write(lines)
   
    
overlap_file.close()
# -------------------------------------------------------
