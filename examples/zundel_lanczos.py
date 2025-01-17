import sys # must be done before magic
import copy
import time
from magic import *
import operator1D
import operatornD
import basis
from mpiWrapper import MPI
from ttns2.driver import eigenStateComputations
from ttns2.diagonalization import IterativeDiagonalizationOptions
from ttns2.parseInput import parseTree
from ttns2.contraction import TruncationEps
from ttns2.misc import oldToNewTTNS
from ttns2.diagonalization import IterativeLinearSystemOptions
from ttns2.sweepAlgorithms import LinearSystem, StateFitting, ApproxGreensFunction
from ttns2.driver import getRenormalizedOp
from ttns2.driver import bracket
from ttns2.renormalization import getRenormalizedOps
from ttns2.renormalization import SumOfOperators
from ttns2.state import loadTTNSFromHdf5
from ttnsVector import TTNSVector
from inexact_Lanczos import inexactLanczosDiagonalization
from util_funcs import get_pick_function_maxOvlp
from ttns2.driver import computeResidual

# MPI.activateMPI()
def _print(*args, **kwargs):
    MPI.barrier()
    if MPI.isMaster():
        print(*args, **kwargs)
        util.flushStdout()             
    MPI.barrier()

import socket 
#import mps
import mctdh_stuff
from mctdh_stuff.utilities import importHeidelCPD

import warnings
#_print("#numpy version, file", np.__version__,np.__file__)
# Zundel KEO has non-hermitian terms
warnings.filterwarnings(action="ignore",message=".*not hermitian!.*")
warnings.filterwarnings(action="ignore",message=".*non-hermitian.*")
#############################
MAX_D = 40 
EPS = 1e-6
convTol = 5e-7
N_BLOCK = 1 
#############################

#############################
# Operator
#############################
HopFile = "/home/larsson/borgstore/larsson/Zundel/RUNS/h5o2p_KEO.op"
fnameBase = "/home/larsson/borgstore/larsson/Zundel/surfaces_marx/mccpd_zundel"

fnameDvr = fnameBase + "/dvr"
fnameCpd = fnameBase + "/cpd"

# ATTENTION: Ordering in operator is different, compared to heidel input!
# => make it as in operator
a_orderingOper =  "x y a R z r1a r2a va ua la r1b r2b vb ub lb".split()
a_orderingInput = "x y z R a r1a r2a va ua la r1b r2b vb ub lb".split()
orderingOper = { o:i for i,o in enumerate(a_orderingOper)}
orderingInput = { o:i for i,o in enumerate(a_orderingInput)}
a_modeComb = ["x y a".split(), "la lb".split(), "ua ub".split(), "r1a r2a va".split(), "r1b r2b vb".split(),
            "z R".split()]
basOpts = [
    basis.Hermite.getOptions(xRange=[-0.8,0.8], N=7), # x
    basis.Hermite.getOptions(xRange=[-0.8,0.8], N=7), # y
    basis.Hermite.getOptions(xRange=[-0.5,0.5], N=19), # z
    basis.SincAB.getOptions(xRange=[3.9,6.0],N=20,xixf=True), # R
    basis.SincAB.getOptions(xRange=[0,3.14159265359], N=11, xixf=True), # a
    basis.Hermite.getOptions(xRange=[0.5,1.8], N=9), # r1a
    basis.Hermite.getOptions(xRange=[2.2,3.8], N=9), # r2a
    basis.SincAB.getOptions(xRange=[-.5,.5], N=9,xixf=True), # va
    basis.SincAB.getOptions(xRange=[-.5,.5], N=9,xixf=True), # ua
    basis.SincAB.getOptions(xRange=[1.3415926535897,4.9415926535897], N=19, xixf=True), # la
    basis.Hermite.getOptions(xRange=[0.5,1.8], N=9), # r1b
    basis.Hermite.getOptions(xRange=[2.2,3.8], N=9), # r2b
    basis.SincAB.getOptions(xRange=[-.5,.5], N=9, xixf=True), # vb
    basis.SincAB.getOptions(xRange=[-.5,.5], N=9, xixf=True), # ub
    basis.SincAB.getOptions(xRange=[-1.8,1.8], N=19, xixf=True), # lb
]
bases = [basis.basisFactory(o) for o in basOpts]
basisDict = {l:b for l,b in zip(a_orderingInput, bases)}
# Reorder, according to oper file
bases = [ bases[orderingInput[i]] for i in a_orderingOper]
nB = [b.N for b in bases]

kmin=3
_operatorDict = {
    "usin":  lambda x: np.sqrt( 1 - x**2),
    "usini": lambda x: np.sqrt( 1 - x**2)**-1,
    "usin2": lambda x: np.sqrt( 1 - x**2)**2,
    "usin2i": lambda x: np.sqrt( 1 - x**2)**-2,
    "rrfac": lambda x: (x-kmin)**2,
    "rfac": lambda x: (x-kmin),
    "urfac": lambda x: (x-kmin)**-1,
    "uurfac": lambda x: (x-kmin)**-2,
    "cos": lambda x: np.cos(x),
    "sin": lambda x: np.sin(x),
}
operatorDict = {}
for k,v in _operatorDict.items():
    operatorDict[k] = operator1D.fx(v, k)

for k,v in  _operatorDict.items():
    operatorDict["dq*"+k+"*dq"] =  operator1D.general( getMat=lambda bas,v=v: bas.mat_dx @ np.diag( v(bas.xi) ) @ bas.mat_dx )
    operatorDict["dq*"+k+"*q"] =  operator1D.general( getMat=lambda bas,v=v: bas.mat_dx @ np.diag( v(bas.xi) *bas.xi) )
    operatorDict["dq*"+k] =  operator1D.general( getMat=lambda bas,v=v: bas.mat_dx @ np.diag( v(bas.xi) ) )
    operatorDict[k+"*dq"] =  operator1D.general( getMat=lambda bas,v=v: np.diag( v(bas.xi) ) @ bas.mat_dx )
    operatorDict[k+"*q"] =  operator1D.general( getMat=lambda bas,v=v: np.diag( v(bas.xi) * bas.xi ) )
    operatorDict["q*"+k] =  operator1D.general( getMat=lambda bas,v=v: np.diag( v(bas.xi) * bas.xi ) )
    operatorDict["q^-2*"+k] =  operator1D.general( getMat=lambda bas,v=v: np.diag( v(bas.xi) * bas.xi**-2 ) )
    operatorDict["q^-1*"+k] =  operator1D.general( getMat=lambda bas,v=v: np.diag( v(bas.xi) * bas.xi**-1 ) )
operatorDict["usin*dq*usin"] = operator1D.general( getMat=lambda bas:  np.diag( _operatorDict["usin"](bas.xi)) @ bas.mat_dx @ np.diag( _operatorDict["usin"](bas.xi)) )
operatorDict["usin*q*dq"] = operator1D.general( getMat=lambda bas:  np.diag( _operatorDict["usin"](bas.xi)*bas.xi) @ bas.mat_dx)
operatorDict["q*dq"] = operator1D.general( getMat=lambda bas:  np.diag( bas.xi ) @ bas.mat_dx)
operatorDict["dq*q"] = operator1D.general( getMat=lambda bas: bas.mat_dx @  np.diag( bas.xi ) )
operatorDict["cdq"] = operator1D.general( getMat=lambda bas: 0.5 * (np.diag(np.cos(bas.xi)) @ bas.mat_dx + bas.mat_dx @ np.diag(np.cos(bas.xi))) )
operatorDict["sdq"] = operator1D.general( getMat=lambda bas: 0.5 * (np.diag(np.sin(bas.xi)) @ bas.mat_dx + bas.mat_dx @ np.diag(np.sin(bas.xi))) )
operatorDict["qdq"] = operator1D.general( getMat=lambda bas: 0.5 * (np.diag(bas.xi) @ bas.mat_dx + bas.mat_dx @ np.diag(bas.xi)) )
operatorDict["qdq*qdq"] = operator1D.general( getMat=lambda bas: 0.25 * (np.diag(bas.xi) @ bas.mat_dx + bas.mat_dx @ np.diag(bas.xi)) @  (np.diag(bas.xi) @ bas.mat_dx + bas.mat_dx @ np.diag(bas.xi)) )
operatorDict["udq"] = operator1D.general( getMat=lambda bas: 0.5 * (np.diag( np.sqrt(1-bas.xi**2) ) @ bas.mat_dx + bas.mat_dx @ np.diag( np.sqrt(1-bas.xi**2) )) )
operatorDict["uqdq"] = operator1D.general( getMat=lambda bas: 0.5 * (np.diag( np.sqrt(1-bas.xi**2) * bas.xi ) @ bas.mat_dx + bas.mat_dx @ np.diag( np.sqrt(1-bas.xi**2) * bas.xi )) )
operatorDict["udq2"] = operator1D.general( getMat=lambda bas: 0.5 * (np.diag( 1-bas.xi**2 ) @ bas.mat_dx + bas.mat_dx @ np.diag( 1-bas.xi**2 ) ) )

for k,v in operatorDict.items():
    if isinstance(v,operator1D.general):
        v.str = k

def getHamiltonian(verbose=True):
    # isMaster, so normal print
    Top = mctdh_stuff.translateOperatorFile(HopFile,operatorDict)
    Top.storeMatrices(bases)
    Top = operatornD.contractSoPOperatorSimpleUsage(Top) # 378 -> 335
    operatornD.absorbCoeff(Top)
    if True:
        # TODO put this in operator BUT DO NOT SUBTRACT
        # ATTENTION offset; Marx-fläche ist nicht geshifted
        offset = +0.8844420921 # eine referenzgeometrie; siehe Email von Markus am 27.01.22
        _print("# ATTENTION: Shift surface by",offset, f"={util.au2unit(offset,'cm-1')} cm-1")
        TopOffset = operatornD.operatorSumOfProduct(nDim=Top.nDim,nSum=1)
        TopOffset.coeff[0] = offset
        operatornD.absorbCoeff(TopOffset, nBas=nB, force=True)
        Top.append(TopOffset)
    Vop, V_modeLabels, V_modeCombs, V_gdim = importHeidelCPD(fnameDvr,fnameCpd,verbose=MPI.isMaster() and verbose)
    operatornD.absorbCoeff(Vop)
    # change DoFlabel
    for i in range(len(a_modeComb)):
        oldLabel = Vop.DoFlabel[i]
        Vop.DoFlabel[i] = str(a_modeComb[i])
        Vop.DoFlabelToIDim[Vop.DoFlabel[i]] = Vop.DoFlabelToIDim[oldLabel]
        del Vop.DoFlabelToIDim[oldLabel]
    hamiltonian = [Top, Vop]
    return hamiltonian

#############################

#_print("# after append CPF: memory=",util.current_memory()[0]/1e3,"gb",flush=True)
#_print("# HopComb +V #coeff=",HopComb.nSum)

tnsString = """
0> [x y a] 10
    1> 10 10
        2> [la lb]
        2> 10 10
            3> [ua ub]
            3> 10 10
                4> [r1a r2a va] 
                4> 10 10
                    5> [r1b r2b vb]
                    5> [z R]
"""
tns = parseTree(tnsString, basisDict, returnType="TTNS")
np.random.seed(898989)
tns.setRandom()
#tns.toPdf()


bondDimensionAdaptions = [TruncationEps(min(EPS*10,0.01), maxD=max(min(10,int(MAX_D*0.2)),1), offset=1, truncateViaDiscardedSum=True)] * 4
bondDimensionAdaptions.extend([TruncationEps(EPS, maxD=max(min(20,int(MAX_D*0.7)),1), offset=2)] * 4)
bondDimensionAdaptions.extend([TruncationEps(EPS, maxD=MAX_D, offset=2)] * 2)
noises = [1e-6] * 4 + [1e-7] * 4 + [1e-8] * 4

davidsonOptions = [IterativeDiagonalizationOptions(tol=1e-7, maxIter=500,maxSpaceFac=200)] * 8
davidsonOptions.append(IterativeDiagonalizationOptions(tol=1e-8, maxIter=500,maxSpaceFac=200))

Hop = getHamiltonian()
Top, Vop = Hop
operator = {
    "RenormalizedSoPOperator": [Top,],
    "SoPOperator": [Vop,],
}

print("------------------ DMRG for init guess ----------")
'''
tnsList, energies = eigenStateComputations(tns, Hop,
                                    nStates=N_BLOCK,
                                    nSweep=50,
                                    iterativeDiagonalizationOptions=davidsonOptions,
                                    bondDimensionAdaptions= bondDimensionAdaptions,
                                    saveDir="states",
                                    allowRestart=False,
                                    convTol=convTol,
                                    )
print("-------------------------------")
'''
filename = "/home/madhumitarano/data/PR39/zundel/z_vscf/states/tns_001.h5" 
tnsList = [loadTTNSFromHdf5(filename)[0]]
assert(len(tnsList)==N_BLOCK)
#######################################################
zpve = 12334.13766
L = 10 
maxit = 20
eConv = 1e-6
target = 920 
bondAdaptLinear = [TruncationEps(EPS, maxD=MAX_D, offset=2, truncateViaDiscardedSum=False)] * 1
bondAdaptOrtho = [TruncationEps(EPS, maxD=MAX_D, offset=2, truncateViaDiscardedSum=False)] * 1
bondAdaptFit = [TruncationEps(EPS, maxD=MAX_D, offset=2, truncateViaDiscardedSum=False)]
#######################################################

optionsOrtho = {"nSweep":40, "convTol":1e-2, "bondDimensionAdaptions":bondAdaptOrtho}
optsCheck = IterativeLinearSystemOptions(solver="gcrotmk",tol=1e-2,maxIter=10) 
optionsLinear = {"nSweep":5, "iterativeLinearSystemOptions":optsCheck,"convTol":5e-2,
        "bondDimensionAdaptions":bondAdaptLinear, "shiftAndInvertMode":True, "optValUnit":"cm-1",
        "optShift":util.unit2au(0.0,"cm-1")}
optionsFitting = {"nSweep":1000, "convTol":1e-9,"bondDimensionAdaptions":bondAdaptFit}
options = {"orthogonalizationArgs":optionsOrtho, "linearSystemArgs":optionsLinear, "stateFittingArgs":optionsFitting}
guess = [TTNSVector(t,options) for t in tnsList]

sigma = util.unit2au((target+zpve),unit="cm-1")
pick = get_pick_function_maxOvlp(guess[0])

ev, tnsList, status = inexactLanczosDiagonalization(operator,guess,sigma,L,maxit,
        eConv,checkFitTol=1e-3,convertUnit="cm-1",pick=pick,eShift=zpve)
print(status)
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
outfile = open("iterations_lanczos.out","a")
outfile.write("Norm of residuals (H\Psi - E\Psi)\n\n")
formatStyle = "{:20} :: {:<20}"
line = formatStyle.format("Eigenvalue","Norm in cm-1")
outfile.write(line+"\n")

ntotal = len(ev)
residual_norm = np.empty((ntotal),dtype=float)
for i in range(ntotal):
    psi = tnsList[i].normalize()
    Enew = TTNSVector.matrixRepresentation(operator,[psi])[0,0]
    residual = computeResidual(psi.ttns,operator,Enew,
            nSweep=15,convTol=1e-2)
    residual_norm[i] = util.au2unit(residual.norm(),"cm-1")

    Enew = util.au2unit(Enew, "cm-1")-zpve
    line = formatStyle.format(Enew,residual_norm[i])
    outfile.write(line+"\n")

outfile.write("\n\n");outfile.close()
# -----------------   EOF  -----------------------
