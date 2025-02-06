import numpy as np
import util
import os

# -----------------------------------------------------------------------------
# In PR39 of eigensolvers, convergence had a bug of skiping residual calcuation
# in every restarts. This script scrutinizes if there is forced continuation.
# Run this script where the summary file is present.
# -----------------------------------------------------------------------------

# ......................... Input .............................................
zpve = 9837.4069
filename = "demo/summary.out"
# ........... Confirmation of complete Lanczos run to set skip_footer .........
f = open(filename,"r")
readlines = f.read().splitlines()
totalLines = len(readlines)
skipFooter = 0
for iline in range(totalLines):
    if readlines[iline] == "endingPoint":
        skipFooter = 5
        break
f.close()
nBlock = int(readlines[6].split(" ")[4])
eConv = float(readlines[11].split(" ")[17])
# ................... Read summary file .......................................
outerIt = np.genfromtxt(filename, skip_header=26, skip_footer=skipFooter,usecols=(0))
innerIt = np.genfromtxt(filename, skip_header=26, skip_footer=skipFooter,usecols=(1))
evcolumns = tuple((4+ib) for ib in range(nBlock))
eigenvalues = np.genfromtxt(filename, skip_header=26, skip_footer=skipFooter,usecols=evcolumns)
res = np.genfromtxt(filename, skip_header=26, skip_footer=skipFooter,usecols=(4+nBlock))

# .................... Evaluation and checking of residual ....................
ntotal = len(outerIt)
if ntotal == 1:
    os.exit("Only one iteration. Residual is not needed to calculate.")

eigenvalues = eigenvalues.reshape(ntotal,nBlock) 
for i in range(ntotal):
    if outerIt[i] > 0 and innerIt[i] == 1:
        actualRes = 0.0
        for ib in range(nBlock):
            diff = util.unit2au(abs(eigenvalues[i][ib] - eigenvalues[i-1][ib]),"cm-1")
            denom = util.unit2au(abs(eigenvalues[i][ib]+zpve),"cm-1")
            actualRes += diff/denom
        print(res[i-1],actualRes)
        if actualRes <= eConv: 
            print("Alert: Forced continuation after restart")
# ...................... EOF ..................................................
