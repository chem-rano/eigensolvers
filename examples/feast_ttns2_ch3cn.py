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
from feast import feastDiagonalization 
from ttnsVector import TTNSVector
from util_funcs import find_nearest
from printUtils import *
from ttns2.diagonalization import IterativeLinearSystemOptions


timeStarting = time.time()
#######################################################
MAX_D = 10 
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
tns.setRandom(dtype=complex)
tns.toPdf()
tns.label = "CH3CN using CSC PES"

if EPS is not None:
    bondDimensionAdaptions = [TruncationEps(EPS, maxD=MAX_D, offset=2, truncateViaDiscardedSum=False)]
else:
    bondDimensionAdaptions = None
noises = [1e-6] * 4 + [1e-7] * 4 + [1e-8] * 6
m0 = 5

davidsonOptions = [IterativeDiagonalizationOptions(tol=1e-7, maxIter=500,maxSpaceFac=200)] * 8
davidsonOptions.append(IterativeDiagonalizationOptions(tol=1e-8, maxIter=500,maxSpaceFac=200))
tnsList, energies = eigenStateComputations(tns, Hop,
                                     nStates=m0,
                                     nSweep=999,
                                     projectionShift=util.unit2au(9999,"cm-1"),
                                     iterativeDiagonalizationOptions=davidsonOptions,
                                     bondDimensionAdaptions= bondDimensionAdaptions,
                                     noises = noises,
                                     allowRestart=True,
                                     convTol=convTol)
##tns = TTNSVector(tnsList[0],options)
# ---------- USER INPUT -----------------------
rmin = 722
rmax = 724
maxit = 5 
nc = 10
eps = 1e-6 
quad = "legendre"
zpve = 9837.4069
rmin += 9837.4069
rmax += 9837.4069
rmin = util.unit2au(rmin,"cm-1")
rmax = util.unit2au(rmax,"cm-1")
nsweepOrtho = 800
orthoTol = 1e-08
optShift = 0.0

siteLinearTol = 1e-3
globalLinearTol = 1e-2
nsweepLinear = 1000

fittingTol = 1e-9
nsweepFitting = 1000
# ---------- USER INPUT -----------------------

optsCheck = IterativeLinearSystemOptions(solver="gcrotmk",tol=siteLinearTol,maxIter=500) 
optionsOrtho = {"nSweep":nsweepOrtho, "convTol":orthoTol, "optShift":optShift, "bondDimensionAdaptions":bondDimensionAdaptions}
optionsLinear = {"nSweep":nsweepLinear, "iterativeLinearSystemOptions":optsCheck,"convTol":globalLinearTol,"bondDimensionAdaptions":bondDimensionAdaptions}
optionsFitting = {"nSweep":nsweepFitting, "convTol":fittingTol,"bondDimensionAdaptions":bondDimensionAdaptions}
options = {"orthogonalizationArgs":optionsOrtho, "linearSystemArgs":optionsLinear, "stateFittingArgs":optionsFitting}
status = {"eShift":zpve, "convertUnit":"cm-1"}

fileHeader("out",options,rmin,nc, maxit,eps,MAX_D)
fileHeader("plot",options,rmin,nc,maxit,eps,MAX_D,printInfo=False)

#m0 = 10 
Y = []
for i in range(m0):
    #tns.setRandom(dtype=complex)
    #Y.append(TTNSVector(tns,options))
    Y.append(TTNSVector(tnsList[i],options))

ev, tnsList = feastDiagonalization(Hop,Y,nc,quad,rmin,rmax,eps,maxit)
target = (rmin+rmax)*0.5
print("Eigenvalues",util.au2unit(ev,"cm-1")-zpve)
#writeFile(status,"out","results",ev,target)
fileFooter("out")
fileFooter("plot",printInfo=False)
# -----------------   EOF  -----------------------
