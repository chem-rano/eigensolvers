import sys # must be done before magic
import copy,gc
import time

import util
from magic import *
import operator1D
import operatornD
import basis
from mpiWrapper import MPI
from ttns2.driver import eigenStateComputations, processOpListMPIMPO, getRenormalizedOps, getRenormalizedStateProjector
from ttns2.diagonalization import IterativeDiagonalizationOptions
from ttns2.parseInput import parseTree
from ttns2.sweepAlgorithms import VSCF
from ttns2.renormalization import SumOfOperators
from ttns2.contraction import TruncationEps
from ttns2.misc import oldToNewTTNS, _inputStringImpl
from ttns2.state import loadTTNSFromHdf5
# TODO rename a_modeComb
def _print(*args, **kwargs):
    MPI.barrier()
    if MPI.isMaster():
        print(*args, **kwargs)
        util.flushStdout()             
    MPI.barrier()

import socket 
import mps
import mctdh_stuff
from mctdh_stuff.utilities import importHeidelCPD

import warnings
#_print("#numpy version, file", np.__version__,np.__file__)
# Zundel KEO has non-hermitian terms
warnings.filterwarnings(action="ignore",message=".*not hermitian!.*")
warnings.filterwarnings(action="ignore",message=".*non-hermitian.*")
#############################
MAX_D = 100
projectionShift = util.unit2au(44000,"cm-1")
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



#_print("# after append CPF: memory=",util.current_memory()[0]/1e3,"gb",flush=True)
#_print("# HopComb +V #coeff=",HopComb.nSum)

tnsString = """
0> 1 [x y a] 
    1> 1 [la lb] 
        2> 1 [ua ub] 
            3> 1 [r1a r2a va]  
                4> 1 [r1b r2b vb] 
                    5> [z R]
"""
tns = parseTree(tnsString, basisDict, returnType="TTNS")
tns.permutationMode = "MPS"
tns.setRandom()
#tns.toPdf()

hamiltonian = getHamiltonian()
operators, hasMPO = processOpListMPIMPO(hamiltonian)
renormalizedOps = getRenormalizedOps(tns, operators)
renormOp = SumOfOperators(renormalizedOps, shift=0)
iex = -1
util.mkDir("states")
# Theta: 4th excitation; 7: combi with r


all_excitations_base = [
        [0,0,0,0,0,0],
        ]
all_excitations = []
for i in range(1,5):
    for e in all_excitations_base:
        e2 = e.copy()
        e2[-1] = i
        all_excitations.append(e2)
for iex, excitations in enumerate(all_excitations):
    print("#---------------------")
    print("# run",iex,"excitations",excitations)
    print("#---------------------")
    vscf = VSCF(tns, renormOp, excitations, optValUnit="cm-1",
                iterativeDiagonalizationThreshold=900,
                nSweep=100,
                saveDir="."
                )
    converged, energy = vscf.run()
    excString = ""
    for e in excitations:
        excString += str(e)+"_"
    excString = excString[:-1]
    tns.label = f"Zundel Marx Surface. VSCF with excitation {excString} and energy {energy}"
    additionalInformation = {"converged":converged, "energy":energy, "excitations": excitations}
    tns.saveToHDF5(f"states/tns_{iex:03d}.h5", additionalInformation=additionalInformation)

