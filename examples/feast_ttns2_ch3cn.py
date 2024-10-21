import util
from magic import *
import mctdh_stuff
import basis
import copy
import sys
import time
import operatornD
from ttns2.driver import eigenStateComputations
from ttns2.diagonalization import IterativeDiagonalizationOptions
from ttns2.parseInput import parseTree
from ttns2.contraction import TruncationEps
from ttns2.misc import mpsToTTNS, getVerbosePrinter
from feast import feastDiagonalization 
from ttnsVector import TTNSVector
from util_funcs import find_nearest
from printUtils import *
from ttns2.diagonalization import IterativeLinearSystemOptions
from ttns2.driver import orthogonalize 


#######################################################
MAX_D = 3
# 5e-9 ok
EPS = 5e-9
convTol = 1e-5
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
#np.random.seed(898989)
#tns.setRandom(dtype=complex)
#tns.toPdf()
#tns.label = "CH3CN using CSC PES"

if EPS is not None:
    bondDimensionAdaptions = [TruncationEps(EPS, maxD=MAX_D, offset=2, truncateViaDiscardedSum=False)]
else:
    bondDimensionAdaptions = None
'''    
noises = [1e-6] * 4 + [1e-7] * 4 + [1e-8] * 6
N_SUBSPACE = 3

davidsonOptions = [IterativeDiagonalizationOptions(tol=1e-7, maxIter=500,maxSpaceFac=200)] * 8
davidsonOptions.append(IterativeDiagonalizationOptions(tol=1e-8, maxIter=500,maxSpaceFac=200))
tnsList, energies = eigenStateComputations(tns, Hop,
                                     nStates=N_SUBSPACE,
                                     nSweep=999,
                                     projectionShift=util.unit2au(9999,"cm-1"),
                                     iterativeDiagonalizationOptions=davidsonOptions,
                                     bondDimensionAdaptions= bondDimensionAdaptions,
                                     noises = noises,
                                     allowRestart=False,
                                     convTol=convTol)
'''
# ---------- USER INPUT -----------------------
Emin = 720  # Lower limit of excitation energy for target interval
Emax = 730   # Upper limit of excitation energy for target interval
maxit = 100 
nc = 6
eps = 1e-6 
quad = "legendre"
zpve = 9837.4069

# ---------- USER INPUT -----------------------

bondAdaptFitting = [TruncationEps(EPS, maxD=10, offset=2, truncateViaDiscardedSum=False)]
optionsLinear = {"nSweep":1000, "iterativeLinearSystemOptions":IterativeLinearSystemOptions(solver="gcrotmk",tol=1e-3,maxIter=1000),"convTol":1e-3,"bondDimensionAdaptions":bondDimensionAdaptions}
optionsFitting = {"nSweep":1000, "convTol":1e-9,"bondDimensionAdaptions":bondAdaptFitting}
options = {"linearSystemArgs":optionsLinear, "stateFittingArgs":optionsFitting}
status = {"eShift":zpve, "convertUnit":"cm-1"}

m0 = 4 
# Random orthogonal tress
setTrees = []
for i in range(m0):
    tns = parseTree(treeString, basisDict, returnType="TTNS")
    np.random.seed(20+i);tns.setRandom(dtype=complex)
    setTrees.append(tns)
setTrees= orthogonalize(setTrees)
        
# Make a TTNSVector list from above orthogonal trees
guess = []
for i in range(m0):
    guess.append(TTNSVector(setTrees[i],options))

fileHeader("out",options,Emin,nc, maxit,eps,MAX_D)
fileHeader("plot",options,Emin,nc,maxit,eps,MAX_D,printInfo=False)

ev_min = util.unit2au((Emin+zpve),"cm-1")  # lower limit of eigenvalue in a.u.
ev_max = util.unit2au((Emax+zpve),"cm-1")  # upper limit of eigenvalue in a.u.
#Y = []
#for i in range(N_SUBSPACE):
#    Y.append(TTNSVector(tnsList[i],options))

ev, tnsList = feastDiagonalization(Hop,guess,nc,quad,ev_min,ev_max,eps,maxit)
print("Eigenvalues",util.au2unit(ev,"cm-1")-zpve)
fileFooter("out")
fileFooter("plot",printInfo=False)
# -----------------   EOF  -----------------------
