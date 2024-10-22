import numpy as np
import scipy as sp
from typing import List, Union
from util_funcs import find_nearest, lowdinOrtho
from printUtils import writeFile
import warnings
import time
import util
from numpyVector import NumpyVector
from abstractVector import AbstractVector
from util_funcs import headerBot
from util_funcs import get_pick_function_close_to_sigma
from util_funcs import get_pick_function_maxOvlp
import copy

# -----------------------------------------------------
# Order of inputs
# working function -> operator, vectors, matrix,
# system inputs, status (in last) unless for keyword (optional)
# arguments 
# writeFile -> plotfile, status, args

# -----------------------------------------------------
# Dividing in to functions for better readability 
# and convenient testing
def _getStatus(status,vector,sigma,maxit,maxKrylov,eConv):
    """ 
    Initialize and update status dictionary
    
    In: status -> param dictionary
        vector -> guess vector to get the
        hasExactAddition property 
        sigma -> target after adjustment of eShift (zpve)
        maxit -> maximum Lanczos iteration
        maxKrylov -> maximum Krylov dimension
        eConv -> eigenvalue convergence
    Out: statusUp  -> initialized and updated

    Status contains following information
    (i)     Inputs : vector property, hasExactAddition,
                    maxit, eConv
    (ii)    Stage of iteration
    (iii)   Convergence info
    (iv)    Run time
    (iv)    print choices

    keys: ["eConv","maxit","maxKrylov","ref","flagAddition",
    "outerIter","innerIter","cumIter",
    "isConverged","lindep","futileRestart",
    "startTime","runTime",
    "writeOut", "writePlot", "eShift","convertUnit"]


    "Ref" is a list -> always contains maximum two values
    Nearest eigenvalues are stored as reference for convergence
    check and restart purpose
    First one is for the previous Lanczos iteration & second is for 
    the current Lanczos iteration
    """
    
    statusUp = {"sigma":sigma,"eConv":eConv,"maxit":maxit,
            "maxKrylov":maxKrylov,"ref":[np.inf],
            "checkFit":1e-5,
            "flagAddition":vector.hasExactAddition,
            "outerIter":0, "innerIter":0,"cumIter":0,
            "isConverged":False,"lindep":False,
            "futileRestart":0,
            "startTime":time.time(), "runTime":0.0,
            "Krylov_maxD":[],"fitted_maxD":None,
            "phase":1,
            "writeOut":True,"writePlot":True,"eShift":0.0,"convertUnit":"au"}
    
    if status is not None:
        givenkeys = status.keys()
    
        for item in givenkeys:       # overwrite defaults
            if item in status:
                statusUp[item] = status[item]
    
    return statusUp

def generateSubspace(Hop,vec,sigma,eConv):
    """ Builds Krylov space by solving linear system
    (Hop-sigma) x = vec
    and subsequent normalization if x is nonzero.
    Nonzero is defined by norm > 0.001*eConv

    In: Hop -> Operator (either as matrix or linearOperator)
        vec -> List of Krylov vectors
        sigma -> Eigenvalue target
        eConv -> Eigenvalue convergence tolerance

    Out: New vector x, nonzero
    """

    typeClass = type(vec)
    out = typeClass.solve(Hop,vec,sigma)
    if typeClass.norm(out) > 0.001*eConv:
        out = typeClass.normalize(out)
        nonzero = True
    else:
        nonzero = False
    return out, nonzero

def compressTTNS(vector):
    ''' Compresses bond dimension of the vector 
    Currently, from fitting bond dimension to 
    linear solver max bond dimension'''

    typeClass = vector.__class__
    options = copy.deepcopy(vector.options)
    fitModified = vector.options["linearSystemArgs"]["bondDimensionAdaptions"] # Manual
    vector.options["stateFittingArgs"]["bondDimensionAdaptions"] = fitModified
    vector = typeClass.linearCombination([vector],[1.0])
    return typeClass(vector.ttns,options)



def _convergence(value,ref):
    ''' Computes convergence quantity (absolute error or 
    relative error, current one is relative error )'''
    
    check_ev = abs(value - ref)/max(abs(value), 1e-14)
    #if absConvergenc:check_ev = abs(ev_nearest - ref)    
    return check_ev


def checkConvergence(ev,status):
    ''' Checks eigenvalue convergence
    
    In: ev -> eigenvalues, sorted based on `pick`
        status -> params dictionary
    
    Out: status (dict: updated isConverged, ref)
         '''
    
    isConverged = False
    ev_nearest = ev[0]   # one state for inexact Lanczos
    if _convergence(ev_nearest,status["ref"][-1]) <= status["eConv"]:
        isConverged = True
    status["isConverged"] = isConverged
    status["runTime"] = time.time() - status["startTime"]
    if status["writePlot"]:
        writeFile("plot",status,ev_nearest,status["ref"][-1])
    status["ref"].append(ev_nearest)
    if len(status["ref"]) > 2:
        status["ref"].pop(0)
    return status
 
def basisTransformation(bases: List[AbstractVector],coeffs: np.ndarray):
    ''' Basis transformation with eigenvectors and Krylov bases

    In: bases -> List of bases for combination
        coeffs -> coefficients used for the combination.
            Can be a 1D array if only one vector should be transformed.

    Out: combBases -> combination results'''

    typeClass = bases[0].__class__
    ndim = coeffs.shape
    combBases = []
    if len(ndim)==1:
        combBases.append(typeClass.linearCombination(bases,coeffs))
    else:
        for j in range(ndim[1]):
            combBases.append(typeClass.linearCombination(bases,coeffs[:,j]))
    return combBases

def properFitting(evNew, ev, status):
    ''' Checks the eigenvalue after fitting
    (at the end of Lanczos iteration)
    In : evNew -> energy after fitting sum of states
         ev -> energy of state before fitting
         status -> Param dictionary
    
    Out: properFit -> (bool: True for accurate linear combination)
    '''
    properFit = True
    
    if status["flagAddition"]:
        properFit = True
    else:
        E1 = util.au2unit(evNew,"cm-1")-status["eShift"]
        E2 = util.au2unit(ev,"cm-1")-status["eShift"]
        if _convergence(evNew,ev) > max(status["eConv"],status["checkFit"]):
           econvprint = max(status["eConv"],status["checkFit"])
           print("_convergence(evNew,evSum)",_convergence(evNew,ev),"Conv for fit",econvprint)
           properFit = False
           print(f"Linearcombination inaccurate: After fit: {E1}. Before fit: {E2}")
    return properFit

def terminateRestart(energy,status,num=3):
    """ If the eigenvalue change is lower than eConv,
    counted as an ineffective restart 

    In: energy -> Energy after fitting
        status -> param dictionary
        num (optional) -> Number of futile restarts
                          Default is 3"""
    
    decision = False
    prevEnergy = status["ref"][0]

    if status["lindep"]:    
        if _convergence(energy,prevEnergy) < max(1e-9,status["eConv"]):
            status["futileRestart"] += 1
    
    if status["futileRestart"] > num:
        print("Lindep and did not have fruitful restarts")
        decision = True

    return decision


def analyzeStatus(status):
    ''' Wrapper of decision parameter for iteration, isConverged'
        in a separate function and conclude to a single 
        bool param continueIteration
        to make main function clean

    In: status -> param dictionary'''

    isConverged = status["isConverged"]
    it = status["outerIter"]
    i = status["innerIter"]
    maxit = status["maxit"]
    continueIteration = True
    
    if isConverged:
        continueIteration = False
    
    if it == maxit -1 and i == status["maxKrylov"]-1: 
        if not isConverged: 
            print("Alert: Lanczos iterations is not converged!")
            continueIteration = False

    return continueIteration
# -----------------------------------------------------

# -----------------------------------------------------
#    Inexact Lanczos with AbstractClass interface
#------------------------------------------------------

def inexactDiagonalization(H,v0: Union[AbstractVector,List[AbstractVector]],
                           sigma,L,maxit,eConv,pick=None,status=None):
    '''
    Calculate eigenvalues and eigenvectors using the inexact Lanczos method


    ---Doing inexact Lanczos in canonical orthogonal basis.---
    Input::  H => input matrix or linearoperator to be diagonalized
             v0 => eigenvector guess.
                    Can be a list of `AbstractVectors`.
                    Then, block Lanczos is performed (Krylov space on each of the guesses).
                    Note that the guess vectors should be orthogonal.
             sigma => eigenvalue estimate
             L => Krylov space dimension
             maxit => Maximum Lanczos iterations
             eConv => relative eigenvalue convergence tolerance
             pick (optional) => pick function for eigenstate 
                            Default is get_pick_function_close_to_sigma
             status (optional) => Additional information dictionary
                    (more details see _getStatus doc)

    Output:: ev as inexact Lanczos eigenvalues
             uv as inexact Lanczos eigenvectors
             status for convergence information
    '''
    if issubclass(type(v0), AbstractVector):
        v0 = [v0]
    else:
        assert isinstance(v0, (list, tuple, np.ndarray)), f"{v0=} {type(v0)=}"
    typeClass = type(v0[0])
    nBlock = len(v0)
    Ylist = v0.copy() # Krylov subspace lists.
    Smat = typeClass.overlapMatrix(Ylist)
    assert np.allclose(Smat, np.eye(nBlock), rtol=1e-3,atol=1e-3), f"Input vectors not orthogonalized: {Smat=}"
    Hmat = typeClass.matrixRepresentation(H,Ylist)
    status = _getStatus(status,Ylist[0],sigma,maxit,L,eConv)
    if pick is None:
        pick = get_pick_function_close_to_sigma(status["sigma"])
    assert callable(pick)
    print(f"# Inexact Lanczos with {nBlock} guess vectors") # TODO in status

    for it in range(maxit):
        status["outerIter"] += 1
        status["Krylov_maxD"] = [Ylist[0].ttns.maxD()]
        status["fitted_maxD"] = None
        for i in range(1,L): # starts with 1 because Y0 is used as first basis vector
            status["innerIter"] = i
            status["cumIter"] += 1
            #
            # Generate subspace
            #
            newVectors = []
            for iBlock in range(1,nBlock+1):
                out, nonzero = generateSubspace(H, Ylist[-iBlock], sigma, eConv)
                if not nonzero:
                    # TODO proper return
                    raise RuntimeError(f"zero vector: ||inv(H-sigma)vec||={typeClass.norm(out):5.3e}")
                newVectors.append(out)
            #
            # Orthogonalize and append
            # Note that the new vectors are also orthogonalized against each other.
            # Also extends overlap and Hamiltonian matrices
            #
            lindepProblem = False
            for iBlock in range(nBlock):
                newOrthVec = typeClass.orthogonalize_against_set(newVectors[iBlock],Ylist)
                if newOrthVec is None:
                    lindepProblem = True
                    warnings.warn(f"Linear dependency problem in iteration {it} "
                                  f"and microiteration {i} for block state {iBlock},"
                                  f" abort current Lanczos iteration and restart.")
                    # As extension, in principle I can continue with the remaining block iterations.
                    #   But I assume that this here rarely happens
                    break
                Ylist.append(compressTTNS(newOrthVec))  # TODO generalize to np
                status["Krylov_maxD"].append(Ylist[-1].ttns.maxD()) # TODO generalize
                # Extend matrices
                Smat = typeClass.extendOverlapMatrix(Ylist, Smat)
                Hmat = typeClass.extendMatrixRepresentation(H, Ylist, Hmat)
            # Overlap info
            if status["writeOut"]:
                writeFile("out", status, "iteration")
                writeFile("out", status, "overlap", Smat)
                writeFile("out", status, f"overlap condition number {np.linalg.cond(Smat):5.3e}")
                writeFile("out", status, "maxD")
            if lindepProblem:
                break
            #
            # Diagonalize
            #
            ev, uv = sp.linalg.eigh(Hmat, Smat)
            if status["writeOut"]:
                writeFile("out", status, "hamiltonian", Hmat)
                writeFile("out", status, "eigenvalues", ev)
            # Reorder uv and ev indices based on `pick`
            idx = pick(uv,Ylist,ev)
            assert len(idx) == len(ev), f"{len(ev)=} {len(idx)=}"
            ev = ev[idx]
            uv = uv[:,idx]
            #
            # Checks
            #
            status = checkConvergence(ev,status)
            continueIteration = analyzeStatus(status)
            if not continueIteration:
                break
        
        if not continueIteration:
            # Finish up and then return
            Ylist = basisTransformation(Ylist, uv)
            # TODO give warning if the basis transformation is not accurate
            status["fitted_maxD"] = [item.ttns.maxD() for item in Ylist]
            if status["writeOut"]:
                writeFile("out",status,"fitD")
            break
        else:
            # Simple restart of Lanczos iteration using new eigenvectors
            newGuessList = []
            for iBlock in range(nBlock):
                guess = basisTransformation(Ylist,uv[:,iBlock])
                guess = typeClass.normalize(guess[0])
                newGuessList.append(guess)
            Ylist = newGuessList
            Smat = typeClass.overlapMatrix(Ylist)
            Hmat= typeClass.matrixRepresentation(H,Ylist)
            # Check accuracy of basis transformation
            for iBlock in range(nBlock):
                evNew = Hmat[iBlock,iBlock]
                if not properFitting(evNew,ev[iBlock],status):
                    # TODO add information about block.
                    break
            # TODO vvv why is this commented out?
            #if terminateRestart(evNew,status):break
            status["fitted_maxD"] = [y.ttns.maxD() for y in Ylist]
            if status["writeOut"]:
                writeFile("out",status,"fitD")
    return ev,Ylist,status
# -----------------------------------------------------
if __name__ == "__main__":
    n = 100
    ev = np.linspace(1,300,n)
    np.random.seed(10)
    Q = sp.linalg.qr(np.random.rand(n,n))[0]
    A = Q.T @ np.diag(ev) @ Q

    target = 30
    maxit = 4
    L = 6 
    eConv = 1e-8

    options = {"linearSolver":"gcrotmk","linearIter":1000,"linear_tol":1e-04}
    optionDict = {"linearSystemArgs":options}
    status = {"writeOut": False,"writePlot": False}
    Y0 = NumpyVector(np.random.random((n)),optionDict)
    sigma = target

    headerBot("Inexact Lanczos")
    print("{:50} :: {: <4}".format("Sigma",sigma))
    print("{:50} :: {: <4}".format("Krylov space dimension",L+1))
    print("{:50} :: {: <4}".format("Eigenvalue convergence tolarance",eConv))
    print("\n")
    t1 = time.time()
    pick =  get_pick_function_close_to_sigma(sigma)
    #pick =  get_pick_function_maxOvlp(Y0)
    lf,xf,status =  inexactDiagonalization(A,Y0,sigma,L,maxit,eConv,pick=pick,status=status)
    t2 = time.time()

    print("{:50} :: {: <4}".format("Eigenvalue nearest to sigma",round(find_nearest(lf,sigma)[1],8)))
    print("{:50} :: {: <4}".format("Actual eigenvalue nearest to sigma",round(find_nearest(ev,sigma)[1],8)))
    print("{:50} :: {: <4}".format("Time taken (in sec)",round((t2-t1),2)))
    headerBot("Lanczos",yesBot=True)
