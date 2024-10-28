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
from inexact_Lanczos import inexactLanczosDiagonalization
from ttnsVector import TTNSVector
from util_funcs import find_nearest
from ttns2.diagonalization import IterativeLinearSystemOptions


timeStarting = time.time()
#######################################################
MAX_D = 3 
# if EPS < 0: EPS = None
# 5e-9 ok
EPS = 5e-9
convTol = 1e-5
N_STATES = 8
#######################################################
_print = getVerbosePrinter(True)
_print("# EPS=",EPS)

fOp = 'ch3cn.op'  # this one is used for HRL's 2019 jcp work
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
tns = parseTree(treeString, basisDict, returnType="TTNS")
np.random.seed(898989)
tns.setRandom()
tns.toPdf()
tns.label = "CH3CN using CSC PES"

if EPS is not None:
    #bondDimensionAdaptions = [TruncationEps(min(EPS*1e3,1e-3), maxD=20, offset=1, truncateViaDiscardedSum=False)] * 4
    #bondDimensionAdaptions.extend([TruncationEps(min(EPS*1e2,1e-3), maxD=40, offset=2, truncateViaDiscardedSum=False)] * 4)
    #bondDimensionAdaptions.extend([TruncationEps(min(EPS*1e2,1e-3), maxD=MAX_D, offset=2, truncateViaDiscardedSum=False)] * 2)
    #bondDimensionAdaptions.extend([TruncationEps(min(EPS*1e1,1e-3), maxD=MAX_D, offset=2, truncateViaDiscardedSum=False)] * 2)
    #bondDimensionAdaptions.extend([TruncationEps(EPS, maxD=MAX_D, offset=2, truncateViaDiscardedSum=False)] * 1)
    # TODO for larger system than ch3cn, try to start with lower bond dimension
    bondDimensionAdaptions = [TruncationEps(EPS, maxD=MAX_D, offset=2, truncateViaDiscardedSum=False)]
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
##tns = TTNSVector(tnsList[0],options)
'''
# ---------- USER INPUT -----------------------
target = 722
maxit = 10 
L = 20 
eConv = 1e-6 
zpve = 9837.4069  
# ---------- USER INPUT -----------------------

optionsOrtho = {"nSweep":1000, "convTol":1e-2, "bondDimensionAdaptions":bondDimensionAdaptions}
optsCheck = IterativeLinearSystemOptions(solver="gcrotmk",tol=1e-4,maxIter=500) 
optionsLinear = {"nSweep":30, "iterativeLinearSystemOptions":optsCheck,"convTol":1e-4,"bondDimensionAdaptions":bondDimensionAdaptions}
optionsFitting = {"nSweep":1000, "convTol":1e-9,"bondDimensionAdaptions":bondDimensionAdaptions}
options = {"orthogonalizationArgs":optionsOrtho, "linearSystemArgs":optionsLinear, "stateFittingArgs":optionsFitting}

tns = TTNSVector(tns,options)
sigma = util.unit2au((target+zpve),unit="cm-1")
eConvAU = util.unit2au(eConv,unit="cm-1")
ev, tnsList = inexactLanczosDiagonalization(Hop,tns,sigma,L,maxit,eConvAU,eShift=zpve,convertUnit="cm-1")[0:2]
# -----------------   EOF  -----------------------
