import numpy as np
import scipy as sp
from typing import List, Union
from util_funcs import find_nearest, lowdinOrtho, basisTransformation
from printUtils import LanczosPrintUtils
import warnings
import time
import util
from abstractVector import AbstractVector
from numpyVector import NumpyVector
from ttnsVector import TTNSVector
from util_funcs import headerBot
from util_funcs import get_pick_function_close_to_sigma
from util_funcs import get_pick_function_maxOvlp
import copy

# -----------------------------------------------------
# Dividing in to functions for better readability 
# and convenient testing
def _getStatus(status, guessVector, nBlock):
    """ 
    Initialize and update status dictionary
    
    In: status -> status input dictionary
        guessVector -> guess vector
        nBlock -> Lanczos blocks
    Out: statusUp  -> initialized and updated

    Status contains following information
    (i)     Block info (number of blocks)
    (ii)    Stage of iteration
    (iii)   Convergence info
    (iv)    Run time
    (v)     Number of phases

    keys: ["ref","nBlock","flagAddition",
    "outerIter","innerIter","cumIter",
    "isConverged","lindep","futileRestarts",
    "startTime","runTime","phase"]


    "ref" is a list -> always contains maximum two values
    Nearest eigenvalues are stored as reference for convergence
    check and restart purpose
    First one is for the previous Lanczos iteration & second is for 
    the current Lanczos iteration
    """
    
    statusUp = {"ref":[np.inf],"nBlock":nBlock,
            "flagAddition":guessVector.hasExactAddition,
            "outerIter":0, "innerIter":0,"cumIter":0,
            "isConverged":False,"lindep":False,
            "futileRestarts":0,
            "startTime":time.time(), "runTime":0.0,
            "KSmaxD":[],"fitmaxD":None,
            "phase":1}

    if status is not None:
        givenkeys = status.keys()
    
        for item in givenkeys:       # overwrite defaults
            if item in status:
                statusUp[item] = status[item]
    
    return statusUp

def generateSubspace(Hop, vec:List[AbstractVector],sigma,eConv):
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

def lowdinOrthoMatrix(S,status,printObj=None):
    ''' Calculates transformation matrix from overlap matrix in Ylist basis
    In: lindep (default value is 1e-14, lowdinOrtho())
        printObj (opional): print object

    Out: status (dict: updated lindep)
         uS: transformation matrix
    Additional: prints overlap matrix in detailed
    output file ("iterations_lanczos.out", default)'''
    
    linIndep, uS = lowdinOrtho(S)[1:3]
    status["lindep"] = not linIndep
    if printObj is not None:
        printObj.writeFile("iteration",status)
        printObj.writeFile("overlap",S)
        printObj.writeFile("KSmaxD",status)

    return status, uS
    
def diagonalizeHamiltonian(X,Hmat,printObj=None):
    ''' Solves eigenvalue problem for Hmat using transformation `X`

    In:
        X -> transformation matrix
        Hmat -> previous matrix representation
        printObj (opional): print object

    Out:
         ev -> eigenvalues
         uv -> eigenvectors
    Additional: prints matrix representation,
                eigenvalues in detailed 
    output file ("iterations_lanczos.out", default)'''
    # TODO merge with the one in Feast

    Hmat = X.T.conj()@Hmat@X
    ev, uv = sp.linalg.eigh(Hmat)
        
    if printObj is not None:
        printObj.writeFile("hamiltonian",Hmat)
        printObj.writeFile("eigenvalues",ev)

    return ev,uv


def _convergence(value, ref):
    ''' Computes convergence quantity (absolute error or 
    relative error, current one is relative error )'''
    
    check_ev = abs(value - ref)/max(abs(value), 1e-14)
    return check_ev


def checkConvergence(ev,eConv,status,printObj=None):
    ''' Checks eigenvalue convergence
    
    In: ev -> eigenvalues, sorted based on `pick`
        status -> params dictionary
        printObj (opional): print object 
    
    Out: status (dict: updated isConverged, ref)
         '''
    
    isConverged = False
    ev_nearest = ev[0]   # one state for inexact Lanczos
    if _convergence(ev_nearest,status["ref"][-1]) <= eConv:
        isConverged = True
    status["isConverged"] = isConverged
    status["runTime"] = time.time() - status["startTime"]
    if printObj is not None:printObj.writeFile("summary",ev_nearest,status)
    status["ref"].append(ev_nearest)
    if len(status["ref"]) > 2:status["ref"].pop(0)
    return status
 
def properFitting(evNew, ev, checkFit, status):
    ''' Checks the eigenvalue after fitting
    (at the end of Lanczos iteration)
    In : evNew -> energy after fitting sum of states
         ev -> energy of state before fitting
         checkFit -> checking tolerance of fitted vectors eigenvalues
         status -> Param dictionary
    
    Out: properFit -> (bool: True for accurate linear combination)
    '''
    properFit = True
    
    if status["flagAddition"]:
        properFit = True
    else:
        if _convergence(evNew,ev) > checkFit:
           properFit = False
           print(f"Linearcombination inaccurate: After fit: {evNew}. Before fit: {ev}")
    return properFit

def terminateRestart(energy,eConv,status,num=3):
    """ This module looks if Lanczos restarts are fruitful or not
    
    futileRestarts -> Number of ineffective or futile restarts
    If the eigenvalue change is lower than eConv,
    counted as an ineffective or futile restart and adds
    1 to futileRestarts

    In: energy -> Energy after fitting
        eConv -> eigenvalue convergence
        status -> param dictionary
        num (optional) -> Number of futile restarts
                          Default is 3
    Out: decision (Boolean) -> decision to terminate restart"""
    
    decision = False
    prevEnergy = status["ref"][0]

    if status["lindep"]:
        if _convergence(energy,prevEnergy) < max(1e-9,eConv):
            status["futileRestarts"] += 1

    if status["futileRestarts"] > num:
        print("Lindep and did not have fruitful restarts")
        decision = True

    return decision


def analyzeStatus(status,maxit,L):
    ''' Wrapper of decision parameter for iteration, isConverged'
        in a separate function and conclude to a single 
        bool param continueIteration
        to make main function clean

    In: status -> param dictionary
        maxit -> maximum Lanczos iterations
        L -> Krylov dimension
        
    Out: decision to continue iteration'''

    isConverged = status["isConverged"]
    it = status["outerIter"]
    i = status["innerIter"]
    continueIteration = True
    
    if isConverged:
        continueIteration = False
    
    if it == maxit -1 and i == L-1: 
        if not isConverged: 
            print("Alert: Lanczos iterations is not converged!")
            continueIteration = False

    return continueIteration
# -----------------------------------------------------

# -----------------------------------------------------
#    Inexact Lanczos with AbstractClass interface
#------------------------------------------------------

def inexactLanczosDiagonalization(H,  v0: Union[AbstractVector,List[AbstractVector]],
                                  sigma, L, maxit, eConv, checkFit=1e-7,
                                  writeOut=True, fileRef=None, eShift=0.0, convertUnit="au", pick=None, status=None,
                                  outFileName=None, summaryFileName=None):
    '''
    Calculate eigenvalues and eigenvectors using the inexact Lanczos method


    ---Doing inexact Lanczos in canonical orthogonal basis.---
    Input::  H => diagonalizable input matrix or linearoperator
             v0 => eigenvector guess
                    Can be a list of `AbstractVectors`.
                    Then, block Lanczos is performed (Krylov space on each of the guesses).
                    Note that the guess vectors should be orthogonal.
             sigma => eigenvalue estimate
             L => Krylov space dimension
             maxit => Maximum Lanczos iterations
             eConv => relative eigenvalue convergence tolerance
             checkFit (optional) => checking tolerance of fitted vectors 
             eigenvalues
             writeOut (optional) => writing file instruction
             default : write both iteration_lanczos.out & summary_lanczos.out
             fileRef (optional) => file containg references (e.g. DMRG energies)
                                   used for summary data file
             eShift
             eShift (optional) => shift value for eigenvalues, Hmat elements
             convertUnit (optional) => convert unit for eigenvalues, Hmat elements
             pick (optional) => pick function for eigenstate 
                            Default is get_pick_function_close_to_sigma
             status (optional) => Additional information dictionary
                    (more details see _getStatus doc)
            outFileName (optional): output file name
            summaryFileName (optional): summary file name

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
    if not np.allclose(Smat, np.eye(nBlock), rtol=1e-3, atol=1e-3):
        if nBlock > 1:
            raise RuntimeError(f"Input vectors not orthogonalized: {Smat=}")
        else:
            # gracefully do this. I do not want to do it for nBlock as GS orthogonalization modifies the block space
            Ylist[0].normalize()
            Smat[0,0] = 1
    Hmat = typeClass.matrixRepresentation(H,Ylist)

    status = _getStatus(status,Ylist[0],nBlock)
    if pick is None:
        pick = get_pick_function_close_to_sigma(sigma)
    assert callable(pick)

    printObj = LanczosPrintUtils(Ylist[0],sigma,L,maxit,eConv,checkFit,
            writeOut,fileRef,eShift,convertUnit,pick,status, outFileName, summaryFileName)
    printObj.fileHeader()

    for outerIter in range(maxit):
        status["outerIter"] = outerIter
        if typeClass is TTNSVector:
            status["KSmaxD"] = [Ylist[0].ttns.maxD()]
            status["fitmaxD"] = None
        for innerIter in range(1,L): # starts with 1 because Y0 is used as first basis vector
            status["innerIter"] = innerIter
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
                    if printObj.writeOut:
                        warnings.warn(f"Linear dependency problem in iteration {outerIter} "
                                  f"and microiteration {innerIter} for block state {iBlock},"
                                  f" abort current Lanczos iteration and restart.")
                    # As extension, in principle I can continue with the remaining block iterations.
                    #   But I assume that this here rarely happens
                    break
                Ylist.append(newOrthVec.compress())
                status["KSmaxD"].append(Ylist[-1].maxD())
                # Extend matrices
                Smat = typeClass.extendOverlapMatrix(Ylist, Smat)
                Hmat = typeClass.extendMatrixRepresentation(H, Ylist, Hmat)
            # Overlap info
            if printObj is not None:
                printObj.writeFile("iteration", status)
                printObj.writeFile("overlap", Smat)
                printObj.writeFile("KSmaxD", status)
            if lindepProblem:
                ev = np.array([np.nan] * len(Ylist))
                del uSH, Hmat, Smat # not up to date
                break
            #
            # Diagonalize
            #
            # Transform to orthogonal basis to check once again linear dependencies
            # I could also just solve the generalized eigenvalue problem directly
            # But this way I could avoid the above GS orthogonalization or modify it
            #   to ignore linear dependency problems
            status, uS = lowdinOrthoMatrix(Smat, status, printObj)
            assert not status["lindep"] # should have been taken care of above
            ev, uv = diagonalizeHamiltonian(uS,Hmat,printObj)
            uSH = uS@uv
            del uv
            # Reorder uv and ev indices based on `pick`
            idx = pick(uSH,Ylist,ev)
            assert len(idx) == len(ev), f"{len(ev)=} {len(idx)=}"
            ev = ev[idx]
            uSH = uSH[:,idx]
            #
            # Checks
            #
            status = checkConvergence(ev,eConv,status,printObj)
            continueIteration = analyzeStatus(status,maxit,L)
            
            if not continueIteration:
                break
        if lindepProblem:
            break

        if not continueIteration:
            # Finish up and then return
            Ylist = basisTransformation(Ylist,uSH)
            # TODO give warning if the basis transformation is not accurate
            status["fitmaxD"] = [item.maxD() for item in Ylist]
            if printObj is not None:
                printObj.writeFile("fitmaxD", status)
            break
        else:
            # Simple restart of Lanczos iteration using new eigenvectors
            # Could be improved using thick restart
            newGuessList = []
            for iBlock in range(nBlock):
                guess = basisTransformation(Ylist,uSH[:,iBlock])
                guess = typeClass.normalize(guess[0])
                newGuessList.append(guess)
            Ylist = newGuessList
            Smat = typeClass.overlapMatrix(Ylist)
            Hmat= typeClass.matrixRepresentation(H,Ylist)
            # Check accuracy of basis transformation
            evNew = []
            for iBlock in range(nBlock):
                evNew.append(Hmat[iBlock,iBlock] / Smat[iBlock,iBlock])
                if not properFitting(evNew[iBlock],ev[iBlock],checkFit,status):
                    # TODO add information about block.
                    break
            ##################################################
            if terminateRestart(evNew,eConv,status):
                break
            if typeClass is TTNSVector:
                status["fitmaxD"] = [item.ttns.maxD() for item in Ylist]
                printObj.writeFile("fitmaxD",status)
    
    printObj.writeFile("results",ev)
    printObj.fileFooter()
    
    return ev, Ylist, status
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
    writeOut = True
    Y0 = NumpyVector(np.random.random((n)),optionDict)
    sigma = target

    t1 = time.time()
    pick =  get_pick_function_close_to_sigma(sigma)
    #pick =  get_pick_function_maxOvlp(Y0)
    lf,xf,status =  inexactLanczosDiagonalization(A, Y0, sigma, L, maxit, eConv, pick=pick, writeOut=writeOut)
    t2 = time.time()

    print("{:50} :: {: <4}".format("Eigenvalue nearest to sigma",round(find_nearest(lf,sigma)[1],8)))
    print("{:50} :: {: <4}".format("Actual eigenvalue nearest to sigma",round(find_nearest(ev,sigma)[1],8)))
    print("{:50} :: {: <4}".format("Time taken (in sec)",round((t2-t1),2)))
