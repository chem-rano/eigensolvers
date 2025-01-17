import sys # must be done before magic
import copy,gc
import time
from magic import *
import operator1D
import operatornD
import basis
from mpiWrapper import MPI
from ttns2.driver import eigenStateComputations
from ttns2.diagonalization import IterativeDiagonalizationOptions
from ttns2.parseInput import parseTree
from ttns2.renormalization import SumOfOperators
from ttns2.contraction import TruncationEps
from ttns2.misc import oldToNewTTNS, _inputStringImpl
from inexact_Lanczos import inexactLanczosDiagonalization
from ttnsVector import TTNSVector
from ttns2.diagonalization import IterativeLinearSystemOptions
from ttns2.driver import computeResidual
from ttns2.state import loadTTNSFromHdf5

import socket 
import mps
import mctdh_stuff
from mctdh_stuff.utilities import importHeidelCPD

import warnings
# Zundel KEO has non-hermitian terms
warnings.filterwarnings(action="ignore",message=".*not hermitian!.*")
warnings.filterwarnings(action="ignore",message=".*non-hermitian.*")
#############################
MAX_D = 70
projectionShift = util.unit2au(44000,"cm-1")
#############################

#############################
# Operator
#############################
HopFile = "/data/larsson/Eigen/op/eigen.op"
fnameBase = "/data/larsson/Eigen/op/mccpd_eigen_sym26"

fnameDvr = fnameBase + "/dvr"
fnameCpd = fnameBase + "/cpd"

a_ordering = """        ROOA  ROOB  uOOB   r1B   uB    aB   r2B   vB   lB   HBx      
        HBy   HBz   ROOC  uOOC  aOOB  r1C  uC    aC   r2C    vC       
        lC    HCx   HCy   HCz   HAx   HAy  HAz   r1A  uA   aOO3     
        r2A   vA    lA""".split()

a_modeComb = [
        "r1A r2A vA".split(),        # 0 water A internal
        "r1B r2B vB".split(),        # 1 water B internal
        "r1C r2C vC".split(),        # 2 water C internal
        "uA lA".split(),              # 3 rocking and wagging WatA
        "uB lB".split(),              # 4 rocking and wagging WatB
        "uC lC".split(),              # 5 rocking and wagging WatC
        "HAx HAy aOO3".split(),      # 6 proton xy + rot water A
        "HBx HBy aB".split(),        # 7 proton xy + rot water B
        "HCx HCy aC".split(),        # 8 proton xy + rot water C
        "ROOA HAz".split(),          # 9 Rz  WatA
        "ROOB HBz".split(),          # 10 Rz  WatB
        "ROOC HCz".split(),          # 11 Rz  WatC
        "aOOB uOOB uOOC".split()]    # 12 pyramidalization + angles between waters
orderingOper = { o:i for i,o in enumerate(a_ordering)}


       
basOpts = [
        basis.Hermite.getOptions(N= 13 , xRange=[4.3, 5.9 ]),                                # ROOA f01O
        basis.Hermite.getOptions(N= 13 , xRange=[4.3, 5.9]),                                 # ROOB f02O
        basis.SincAB.getOptions(N=11,xRange=[ -0.71,-0.1], xixf=True),                       # uOOB f03
        basis.Hermite.getOptions(N=9 , xRange=[2.3, 3.6]),                                   # r1B f04W
        basis.SincAB.getOptions(N= 9 ,xRange=[-0.5, 0.5], xixf=True),                        # uB f05W
        basis.FGHexp.getOptions(N=11, xRange=[0,2*np.pi]),                                   # aB f06W
        basis.Hermite.getOptions(N=7 , xRange=[0.7, 1.6]),                                   # r2B f07W
        basis.SincAB.getOptions(N= 7,xRange=[ -0.3, 0.3], xixf=True),                        # vB f08W
        basis.SincAB.getOptions(N=13 ,xRange=[2.1693704313678, 4.1138148758122 ], xixf=True),# lB f09W
        basis.Hermite.getOptions(N=9 , xRange=[-0.84, 0.84]),                                # HBx f10x
        basis.Hermite.getOptions(N=9 , xRange=[-0.84, 0.84 ]),                               # HBy f11y
        basis.Hermite.getOptions(N=9 , xRange=[1.35,  2.65]),                                # HBz f12z
        basis.Hermite.getOptions(N= 13 , xRange=[4.3,  5.9 ]),                               # ROOC f13O
        basis.SincAB.getOptions(N=11,xRange=[ -0.71, -0.1 ], xixf=True),                     # uOOC f14
        basis.SincAB.getOptions(N=11,xRange=[ 2.14159265359,  3.14159265359 ], xixf=True),   # aOOB f15P
        basis.Hermite.getOptions(N=9 , xRange=[ 2.3, 3.6 ]),                                 # r1C f16W
        basis.SincAB.getOptions(N= 9,xRange=[ -0.5,  0.5], xixf=True),                       # uC f17W
        basis.FGHexp.getOptions(N=11, xRange=[0,2*np.pi]),                                   # aC f18W
        basis.Hermite.getOptions(N=7 , xRange=[ 0.7, 1.6 ]),                                 # r2C f19W
        basis.SincAB.getOptions(N= 7,xRange=[-0.3, 0.3 ], xixf=True),                        # vC f20W
        basis.SincAB.getOptions(N=13 ,xRange=[2.1693704313678, 4.1138148758122 ], xixf=True),# lC f21
        basis.Hermite.getOptions(N=9 , xRange=[-0.84, 0.84 ]),                               # HCx f22x
        basis.Hermite.getOptions(N=9 , xRange=[-0.84, 0.84]),                                # HCy f23y
        basis.Hermite.getOptions(N=9 , xRange=[1.35,  2.65 ]),                               # HCz f24
        basis.Hermite.getOptions(N=9 , xRange=[-0.84, 0.84]),                                # HAx f25x
        basis.Hermite.getOptions(N=9 , xRange=[-0.84, 0.84 ]),                               # HAy f26y
        basis.Hermite.getOptions(N=9 , xRange=[1.35,  2.65]),                                # HAz f27
        basis.Hermite.getOptions(N=9 , xRange=[ 2.3, 3.6 ]),                                 # r1A f28W
        basis.SincAB.getOptions(N= 9,xRange=[-0.5, 0.5], xixf=True),                         # uA f29W
        basis.FGHexp.getOptions(N=11, xRange=[0,2*np.pi]),                                   # aOO3 f30W
        basis.Hermite.getOptions(N=7 , xRange=[ 0.7, 1.6 ]),                                 # r2A f31W
        basis.SincAB.getOptions(N= 7 ,xRange=[ -0.3, 0.3], xixf=True),                       # vA f32W
        basis.SincAB.getOptions(N=13 ,xRange=[2.1693704313678, 4.1138148758122], xixf=True), # lA f33W
        ]

#######################################################
N_BLOCK = 1
target = -3306

L = 10 
maxit = 20
eConv = 1e-16
EPS = 5e-9
bondAdaptLinear = [TruncationEps(EPS, maxD=MAX_D, offset=2, truncateViaDiscardedSum=False)] * 1
bondAdaptOrtho = [TruncationEps(EPS, maxD=MAX_D, offset=2, truncateViaDiscardedSum=False)] * 1
bondAdaptFit = [TruncationEps(EPS, maxD=MAX_D, offset=2, truncateViaDiscardedSum=False)]
#######################################################

treeString = """
0> 70 70 70
  1> 70 17
   2> 70 58
     3> [HAx HAy aOO3]
    3> 28 17
      4> [uA lA]
      4> [r1A r2A vA]
    2> [ROOA HAz]
  1> 70 17
   2> 70 58
     3> [HBx HBy aB]
    3> 28 17
      4> [uB lB]
      4> [r1B r2B vB]
    2> [ROOB HBz]
  1> 70 70
   2> 70 17
    3> 70 57
      4> [HCx HCy aC]
     4> 29 17
       5> [uC lC]
       5> [r1C r2C vC]
     3> [ROOC HCz]
    2> [aOOB uOOB uOOC]
    """
bases = [basis.basisFactory(o) for o in basOpts]
basisDict = {l:b for l,b in zip(a_ordering, bases)}
nB = [b.N for b in bases]

_operatorDict = {
    "cos": lambda x: np.cos(x),
    "sin": lambda x: np.sin(x),
    "qs" : lambda x:np.sqrt(1-x**2),
    "q^-1" : lambda x: 1./x,
    "qs^-1" : lambda x:(1-x**2)**(-1/2),
    "qs^-2" : lambda x:(1-x**2)**(-1),
}
operatorDict = {}
for k,v in _operatorDict.items():
    operatorDict[k] = operator1D.fx(v, k)
    operatorDict[k+"*q"] =  operator1D.fx( lambda x,v=v: v(x) * x, k+"*q")
    operatorDict["q*"+k] =  operatorDict[k+"*q"]
    operatorDict["q^-2*"+k] =  operator1D.fx( lambda x,v=v: v(x) * x**-2, "q^-2*"+k)
    operatorDict["q^-1*"+k] =  operator1D.fx( lambda x,v=v: v(x) * x**-1, "q^-1*"+k)
    operatorDict["q^2*"+k] =  operator1D.fx( lambda x,v=v: v(x) * x**2, "q^2*"+k)

for k,v in  _operatorDict.items():
    operatorDict["dq*"+k+"*dq"] =  operator1D.general( getMat=lambda bas,v=v: bas.mat_dx @ np.diag( v(bas.xi) ) @ bas.mat_dx )
    operatorDict["dq*"+k+"*q"] =  operator1D.general( getMat=lambda bas,v=v: bas.mat_dx @ np.diag( v(bas.xi) *bas.xi) )
    operatorDict["dq*"+k] =  operator1D.general( getMat=lambda bas,v=v: bas.mat_dx @ np.diag( v(bas.xi) ) )
    operatorDict[k+"*dq"] =  operator1D.general( getMat=lambda bas,v=v: np.diag( v(bas.xi) ) @ bas.mat_dx )
_dq = lambda b : b.mat_dx
_qm1 = lambda b: np.diag( 1. / b.xi) 
_q = lambda b: np.diag( b.xi ) 
_q2 = lambda b: np.diag( b.xi ** 2 ) 
_q3 = lambda b: np.diag( b.xi ** 3 ) 
_qs = lambda b:np.diag(_operatorDict["qs"](b.xi))
operatorDict["q*dq"] = operator1D.general( getMat=lambda bas:  _q(bas) @ _dq(bas) )
operatorDict["dq*q"] = operator1D.general( getMat=lambda bas: _dq(bas) @ _q(bas) )
operatorDict["dq*q^2*dq"] = operator1D.general( getMat=lambda bas: _dq(bas) @ _q2(bas) @ _dq(bas) )
operatorDict["q*dq*q^-1"] = operator1D.general( getMat=lambda bas: _q(bas) @ _dq(bas) @ _qm1(bas) ) 
operatorDict["q^-1*dq*q"] = operator1D.general( getMat=lambda bas:  _qm1(bas) @ _dq(bas) @ _q(bas) )
operatorDict["dq*q^2"] = operator1D.general( getMat=lambda bas: _dq(bas)  @  _q2(bas))
operatorDict["q^2*dq"] = operator1D.general( getMat=lambda bas: _q2(bas) @ _dq(bas))
operatorDict["q^-1*dq*q^2*dq*q^-1"] = operator1D.general( getMat=lambda bas: _qm1(bas) @ _dq(bas) @ _q2(bas) @ _dq(bas) @ _qm1(bas) )
operatorDict["dq*q*qs"] = operator1D.general( getMat=lambda bas: _dq(bas) @ _q(bas) @ _qs(bas) )
operatorDict["dq*q*dq"] = operator1D.general( getMat=lambda bas: _dq(bas) @ _q(bas) @ _dq(bas) )
operatorDict["dq*q^3*dq"] = operator1D.general( getMat=lambda bas: _dq(bas) @ _q3(bas) @ _dq(bas) )
operatorDict["q*qs*dq"] = operator1D.general( getMat=lambda bas: _q(bas) @ _qs(bas) @ _dq(bas) )

for k,v in operatorDict.items():
    if isinstance(v,operator1D.general):
        v.str = k
def getHamiltonian(verbose=True):
    # isMaster, so normal print
    Top = mctdh_stuff.translateOperatorFile(HopFile,operatorDict)
    Top.storeMatrices(bases)
    Top = operatornD.contractSoPOperatorSimpleUsage(Top) # 378 -> 335
    operatornD.absorbCoeff(Top)
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


bondDimensionAdaptions = [TruncationEps(min(EPS*10,0.01), maxD=20, offset=1, truncateViaDiscardedSum=True)] * 4
bondDimensionAdaptions.extend([TruncationEps(EPS, maxD=40, offset=2)] * 4)
bondDimensionAdaptions.extend([TruncationEps(EPS, maxD=66, offset=2)] * 4)
bondDimensionAdaptions.extend([TruncationEps(EPS, maxD=100, offset=2)] * 4)
bondDimensionAdaptions.extend([TruncationEps(EPS, maxD=MAX_D, offset=2)] * 2)
noises = [1e-6] * 4 + [1e-7] * 4 * 3 + [1e-8] * 4

davidsonOptions = [IterativeDiagonalizationOptions(tol=1e-7, maxIter=500,maxSpaceFac=200)] * 8
davidsonOptions.extend([IterativeDiagonalizationOptions(tol=1e-8, maxIter=500,maxSpaceFac=200)]*40)
davidsonOptions.extend([IterativeDiagonalizationOptions(tol=1e-9, maxIter=500,maxSpaceFac=200)]*40)
davidsonOptions.extend([IterativeDiagonalizationOptions(tol=1e-10, maxIter=500,maxSpaceFac=200)]*1)


Hop = getHamiltonian()
Top, Vop = Hop
operator = {
    "RenormalizedSoPOperator": [Top,],
    "SoPOperator": [Vop,],
}

tns = parseTree(treeString, basisDict, returnType="TTNS")
np.random.seed(898989)
tns.setRandom()
print("------------------ DMRG for init guess ----------")
# Generate guess
#tnsList, energies = eigenStateComputations(tns, Hop,
#                                           nStates=N_BLOCK,
#                                           nSweep=12,
#                                           returnIfBelowOptVal = util.unit2au((target+1.0),"cm-1"),
#                                           allowRestart=False,
#                                           projectionShift=projectionShift)

# Load guess
tnsList = []

path = "/home/madhumitarano/data/PR39/Eigen/benchmark/D70/3306/L10/states/"
for iBlock in range(N_BLOCK):
    filename = path + "tns_0000"+str(iBlock)+".h5"
    tnsList.append(loadTTNSFromHdf5(filename)[0])

print("-------------------------------")
assert(len(tnsList)==N_BLOCK)
###############
optionsOrtho = {"nSweep":40, "convTol":1e-2, "bondDimensionAdaptions":bondAdaptOrtho}
optsCheck = IterativeLinearSystemOptions(solver="gcrotmk",tol=1e-2,maxIter=10) 
optionsLinear = {"nSweep":5, "iterativeLinearSystemOptions":optsCheck,"convTol":5e-2,
        "bondDimensionAdaptions":bondAdaptLinear, "shiftAndInvertMode":True, "optValUnit":"cm-1",
        "optShift":util.unit2au(0.0,"cm-1")}
optionsFitting = {"nSweep":1000, "convTol":1e-9,"bondDimensionAdaptions":bondAdaptFit}
options = {"orthogonalizationArgs":optionsOrtho, "linearSystemArgs":optionsLinear, "stateFittingArgs":optionsFitting}
guess = [TTNSVector(t,options) for t in tnsList]

sigma = util.unit2au(target,unit="cm-1")
ev, tnsList, status = inexactLanczosDiagonalization(operator,guess,sigma,L,maxit,
        eConv,checkFitTol=1e-3,convertUnit="cm-1")
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
    Enew = TTNSVector.matrixRepresentation(Hop,[psi])[0,0]
    residual = computeResidual(psi.ttns,Hop,Enew,
            nSweep=30,convTol=1e-2)
    residual_norm[i] = util.au2unit(residual.norm(),"cm-1")

    Enew = util.au2unit(Enew, "cm-1")
    line = formatStyle.format(Enew,residual_norm[i])
    outfile.write(line+"\n")

outfile.write("\n\n");outfile.close()
# -----------------   EOF  -----------------------
