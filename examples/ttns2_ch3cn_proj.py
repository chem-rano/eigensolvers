import util
from magic import *
import mctdh_stuff
import basis
import copy
import sys
import time
from mpiWrapper import MPI
import operatornD
from ttns2.driver import eigenStateComputations
from ttns2.diagonalization import IterativeDiagonalizationOptions
from ttns2.parseInput import parseTree
from ttns2.contraction import TruncationEps
from ttns2.misc import mpsToTTNS, getVerbosePrinter
from inexact_Lanczos import inexactLanczosDiagonalization
from ttnsVector import TTNSVector
from ttns2.diagonalization import IterativeLinearSystemOptions
from ttns2.driver import computeResidual
from ttns2.state import loadTTNSFromHdf5

#######################################################
MAX_D = 10 
MAX_D = 2 
N_BLOCK = 1
zpve = 9837.4069  

L = 10 
maxit = 20
eConv = 1e-6
EPS = 5e-9
bondAdaptLinear = [TruncationEps(EPS, maxD=MAX_D, offset=2, truncateViaDiscardedSum=False)] * 1
bondAdaptOrtho = [TruncationEps(EPS, maxD=MAX_D, offset=2, truncateViaDiscardedSum=False)] * 1
bondAdaptFit = [TruncationEps(EPS, maxD=MAX_D, offset=2, truncateViaDiscardedSum=False)]
#######################################################
_print = getVerbosePrinter(True)
_print("# EPS=",EPS)

fOp = '/home/madhumitarano/data/lanczos/ch3cn.op'
# this operator file is used for HRL's 2019 jcp work
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
        2> [x5 x6]
    1> 3 3
        2> [x7 x8]
        2> [x9 x10]
    1> 3 3
        2> 3 3
            3> [x3]
            3> 3 3
                4> [x2]
                4> [x4]
        2> 3
            3> [x11 x12]
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
#tns.toPdf("guess_forDMRG.pdf")
tns.label = "CH3CN using Carrington PES"

print("------------------ projected DMRG guess ----------")
filename = "demo/lanczosSolution1.h5"
isGuessTarget = False # if the guess energy is target, else minimum of the last vectors

guess = loadTTNSFromHdf5(filename)[0]
guessE = loadTTNSFromHdf5(filename)[1]["energy"]
guess.normalize()  # for Lanczos guess, as these are approx. fitted
PROJECTION_SHIFT = util.unit2au(8499,"cm-1")
print("-------------------------------")
###############
optionsOrtho = {"nSweep":40, "convTol":1e-2, "bondDimensionAdaptions":bondAdaptOrtho}
optsCheck = IterativeLinearSystemOptions(solver="gcrotmk",tol=1e-2,maxIter=10)
optionsLinear = {"nSweep":5, "iterativeLinearSystemOptions":optsCheck,
        "convTol":5e-2,"bondDimensionAdaptions":bondAdaptLinear,
        "shiftAndInvertMode":True, "optValUnit":"cm-1","optShift":util.unit2au(zpve,"cm-1")}
optionsFitting = {"nSweep":1000, "convTol":1e-9,"bondDimensionAdaptions":bondAdaptFit}
options = {"orthogonalizationArgs":optionsOrtho, "linearSystemArgs":optionsLinear, "stateFittingArgs":optionsFitting}
###############
# previous Lanczos trees for RenormalizedStateProjector
primaryFiles = np.loadtxt("demo/primaryVectors.dat",dtype=str,usecols=(0),skiprows=2)
secondaryFiles = np.loadtxt("demo/secondaryVectors.dat",dtype=str,usecols=(0),skiprows=2)
prefix = primaryFiles[0]

convergedStates = []
convergedEnergies = []


# First get target energy
if isGuessTarget:
    target = guessE
elif not isGuessTarget:
    targetEnegies = []
    for i in range(1,len(primaryFiles)):# first is prefix
         filename = prefix + primaryFiles[i]
         targetEnegies.append(loadTTNSFromHdf5(filename)[1]["energy"])
    minEnergy = min(targetEnegies)
    target = minEnergy
    # add this vector in convergedStae
    indx = targetEnegies.index(minEnergy) +1 # first is prefix
    filename = prefix + primaryFiles[indx]
    convergedStates.append(loadTTNSFromHdf5(filename)[0])
    convergedEnergies.append(loadTTNSFromHdf5(filename)[1]["energy"])
    # and remove from primaryFiles
    primaryFiles = np.delete(primaryFiles,indx)
numTNSList = len(convergedStates) # zero for empty list

vectors = []
energies = []
# second get converged states and energies
files = np.concatenate((primaryFiles[1:],secondaryFiles[1:]))
for i in range(len(files)):
    filename = prefix + files[i]
    vectors.append(loadTTNSFromHdf5(filename)[0])
    energies.append(loadTTNSFromHdf5(filename)[1]["energy"])

idx = np.argsort(np.abs(np.array(energies) - target))

# restricting list to L items
for i in range(L-numTNSList):
    convergedStates.append(vectors[idx[i]])
    convergedEnergies.append(energies[idx[i]])

target = util.au2unit(target,"cm-1")-zpve
# for monitoring convergedStates, below are only for printing
print(f"Length of convergedStates for projection: {len(convergedStates)}")
print(f"Target: {target}")
excitationConvergedEnergies = util.au2unit(np.array(convergedEnergies),"cm-1")-zpve
print(f"excitationConvergedEnergies: {excitationConvergedEnergies}")
# end of monitoring convergedStates

assert(len(convergedStates) <= L)
# set default and label
for i in range(len(convergedStates)):
    convergedStates[i].setDefault()
    convergedStates[i].label = f"CH3CN using Carrington PES; \
            {i}-th converged state"

operator = {
    "RenormalizedSoPOperator": [Hop,],
    "RenormalizedStateProjector": [convergedStates, np.array(convergedEnergies) + PROJECTION_SHIFT ]}

###############

guess = TTNSVector(guess,options)
guess.ttns.label = "CH3CN using Carrington PES"
guess.ttns.setDefault()
# Need to update options
options["linearSystemArgs"]["auxList"] = convergedStates
guess.options = options
# NOTE: Currently RenormalizedStateProjector only works for bra==ket,
#       so it will not work for a getting the matrix representation in the Krylov space
#       Thus only use it for solving the linear system (which should be enough)
sigma = util.unit2au((target+zpve),unit="cm-1")
ev, tnsList, status = inexactLanczosDiagonalization(Hop,guess,sigma,L,maxit,eConv,
                                                    Hsolve=operator,checkFitTol=1e-3,
                                                    eShift=zpve,convertUnit="cm-1",
                                                    saveTNSsEachIteration=False,
                                                    outFileName="iterations.out",
                                                    summaryFileName="summary.out")

print("Eigenvalues",util.au2unit(ev,"cm-1")-zpve)
# ----------------- Saving Lanczos tress ---------
directory = "finalLanczosTNSs/"
if not os.path.exists(directory):
    os.makedirs(directory)

nvectors = len(tnsList)
for ivec in range(nvectors):
    filename = directory + "lanczosSolution"+str(ivec)+".h5"
    Info = {"energy": ev[ivec],"converged":status["isConverged"],
        "L":L,"target":target,"MAX_D":MAX_D,"N_BLOCK":N_BLOCK} 
    tnsList[ivec].ttns.saveToHDF5(filename,additionalInformation=Info)

# ----------------- Residuals --------------------
outfile = open("iterations.out","a")
outfile.write("Norm of residuals (H\Psi - E\Psi)\n\n")
formatStyle = "{:20} :: {:<20}"
line = formatStyle.format("Eigenvalue","Norm in cm-1")
outfile.write(line+"\n")

ntotal = len(ev)
residual_norm = np.empty((ntotal),dtype=float)
for i in range(ntotal):
    psi = tnsList[i].normalize()
    Enew = TTNSVector.matrixRepresentation(Hop,[psi])[0,0]
    residual = computeResidual(psi.ttns,Hop,Enew,
            nSweep=15,convTol=1e-2)
    residual_norm[i] = util.au2unit(residual.norm(),"cm-1")

    Enew = util.au2unit(Enew, "cm-1")-zpve
    line = formatStyle.format(Enew,residual_norm[i])
    outfile.write(line+"\n")

outfile.write("\n\n");outfile.close()
# -----------------   EOF  -----------------------
