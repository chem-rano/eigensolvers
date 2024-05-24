import util
from magic import *
import mctdh_stuff
import basis
import copy
import sys
import time
from mpiWrapper import MPI
MPI.activateMPI()
import operatornD
from ttns2.driver import eigenStateComputations
from ttns2.diagonalization import IterativeDiagonalizationOptions
from ttns2.parseInput import parseTree
from ttns2.contraction import TruncationEps
from ttns2.misc import mpsToTTNS, getVerbosePrinter
from inexact_Lanczos import inexactDiagonalization 
from ttnsVector import TTNSVector
from util_funcs import find_nearest
from datetime import datetime
from printUtils import *

timeStarting = time.time()
#######################################################
MAX_D = 100 
# 5e-9 ok
if len(sys.argv) > 1:
    EPS    = float(sys.argv[1]) # only used in between!
    if EPS < 0:
        EPS = None
else:
    EPS = None
convTol = 1e-5
N_STATES = 8
#######################################################
_print = getVerbosePrinter(True)
_print("# EPS=",EPS)

fOp = 'examples/ch3cn.op'  # this one is used for HRL's 2019 jcp work
Hop = mctdh_stuff.translateOperatorFile(fOp, verbose=False)
_print("# Hop: nSum=",Hop.nSum)

N = 42
DVRopts = [
    basis.Hermite.getOptions(N=N , HOx0=0, HOw=1, HOm=1),
    basis.Hermite.getOptions(N=N , HOx0=0, HOw=1, HOm=1),
    basis.Hermite.getOptions(N=N , HOx0=0, HOw=1, HOm=1),
    basis.Hermite.getOptions(N=N , HOx0=0, HOw=1, HOm=1),
    basis.Hermite.getOptions(N=N , HOx0=0, HOw=1, HOm=1),
    basis.Hermite.getOptions(N=N , HOx0=0, HOw=1, HOm=1),
    basis.Hermite.getOptions(N=N , HOx0=0, HOw=1, HOm=1),
    basis.Hermite.getOptions(N=N , HOx0=0, HOw=1, HOm=1),
    basis.Hermite.getOptions(N=N , HOx0=0, HOw=1, HOm=1),
    basis.Hermite.getOptions(N=N , HOx0=0, HOw=1, HOm=1),
    basis.Hermite.getOptions(N=N, HOx0=0, HOw=1, HOm=1),
    basis.Hermite.getOptions(N=N, HOx0=0, HOw=1, HOm=1),
]
treeString = """
0> 3 3 3
    1> 3 3
        2> [x1]
        2> 3 3
            3> [x5]
            3> [x6]
    1> 3 3
        2> 3 3
            3> [x7]
            3> [x8]
        2> 3 3
            3> [x9]
            3> [x10]
    1> 3 3
        2> 3 3 
            3> [x3]
            3> 3 3
               4> [x2]
               4> [x4]
        2> 3 3 
            3> [x11] 
            3> [x12]
    """
bases = [basis.basisFactory(o) for o in DVRopts]
nBas = [b.N for b in bases]
Hop.storeMatrices(bases)
Hop = operatornD.contractSoPOperatorSimpleUsage(Hop)
operatornD.absorbCoeff(Hop)
_print("# Hop contracted nSum",Hop.nSum)
Hop.obtainMultiplyOp(bases)
basisDict = {l:b for l,b in zip(Hop.DoFlabel, bases)}
# basisDict["stateAv"] = basis.electronic(2)
tns = parseTree(treeString, basisDict, returnType="TTNS")
np.random.seed(898989)
tns.setRandom()
tns.toPdf()
tns.label = "CH3CN using CSC PES"

if EPS is not None:
    bondDimensionAdaptions = [TruncationEps(min(EPS*1e3,1e-3), maxD=20, offset=1, truncateViaDiscardedSum=False)] * 4
    bondDimensionAdaptions.extend([TruncationEps(min(EPS*1e2,1e-3), maxD=40, offset=2, truncateViaDiscardedSum=False)] * 4)
    bondDimensionAdaptions.extend([TruncationEps(min(EPS*1e2,1e-3), maxD=MAX_D, offset=2, truncateViaDiscardedSum=False)] * 2)
    bondDimensionAdaptions.extend([TruncationEps(min(EPS*1e1,1e-3), maxD=MAX_D, offset=2, truncateViaDiscardedSum=False)] * 2)
    bondDimensionAdaptions.extend([TruncationEps(EPS, maxD=MAX_D, offset=2, truncateViaDiscardedSum=False)] * 1)
else:
    bondDimensionAdaptions = None
noises = [1e-6] * 4 + [1e-7] * 4 + [1e-8] * 6

'''
davidsonOptions = [IterativeDiagonalizationOptions(tol=1e-7, maxIter=500,maxSpaceFac=200)] * 8
davidsonOptions.append(IterativeDiagonalizationOptions(tol=1e-8, maxIter=500,maxSpaceFac=200))
tnsList, energies = eigenStateComputations(tns, Hop,
                                     nStates=1,
                                     nSweep=999,
                                     projectionShift=util.unit2au(9999,"cm-1"),
                                     iterativeDiagonalizationOptions=davidsonOptions,
                                     bondDimensionAdaptions= bondDimensionAdaptions,
                                     noises = noises,
                                     allowRestart=True,
                                     convTol=convTol)
'''
# ---------- USER INPUT -----------------------
zpve = 9837.4069  # cm-1
sigma = 1782; # excitation energy 
L = 20  # 
maxit = 20
nsweepOrtho = 800
orthoTol = 1e-08
optShift = 0.0
bondDimensionAdaptions = None

siteLinearTol = 1e-3
globalLinearTol = 1e-2
nsweepLinear = 1000
#raiseNonConvergenceException = True

fittingTol = 1e-9
nsweepFitting = 1000
eConv = 1e-4 # abs in cm-1
fout = open("iterations.out","a")
fplot = open("data2Plot.out","a")
files={"out":fout,"plot":fplot}
# ---------- USER INPUT -----------------------

optionsOrtho = {"nSweep":nsweepOrtho, "convTol":orthoTol, "optShift":optShift, "bondDimensionAdaptions":bondDimensionAdaptions}
optionsLinear = {"nSweep":nsweepLinear, "iterativeLinearSystemOptions":[siteLinearTol],"convTol":globalLinearTol}
optionsFitting = {"nSweep":nsweepFitting, "convTol":fittingTol}
options = {"orthogonalizationArgs":optionsOrtho, "linearSystemArgs":optionsLinear, "stateFittingArgs":optionsFitting}

dateTime = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
writeInputs(files["out"],dateTime,sigma,zpve,L, maxit, MAX_D,eConv,options,guess="Random",printInfo=True) # using _writeFile
fplotHeader(files["plot"],dateTime,sigma,zpve,L,maxit,MAX_D,eConv,options)
#tns = TTNSVector(tnsList[0],options)
tns = TTNSVector(tns,options)
startTime = time.time()
energies, tnsList = inexactDiagonalization(Hop,tns,sigma,L,maxit,eConv,eShift=zpve)[0:2] # main function
ev_nearest = find_nearest(energies,sigma)[1]
files["out"].write("\n\n"+"-"*20+"\tFINAL RESULTS\t"+"-"*20+"\n")
files["out"].write("{:30} :: {: <4}, {: <4}".format("Sigma, calculated nearest",sigma,round(ev_nearest),4)+"\n")

list_results = ""
for i in range(0,(len(energies)-1),1):
    list_results += str(round(energies[i],4))+", "
list_results +=  str(round(energies[-1],4))
files["out"].write("{:30} :: {: <4}".format("All subspace eigenvalues",list_results)+"\n")
printfooter(fout,printInfo=True)
fplotFooter(files["plot"])
files["out"].close()
files["plot"].close()
# -----------------   EOF  -----------------------
