import numpy as np
import scipy as sp
from util_funcs import find_nearest, lowdinOrtho
from printUtils import writeFile
import warnings
import time
import util
from numpyVector import NumpyVector
from util_funcs import headerBot

# -----------------------------------------------------
# Order of inputs
# working function -> operator, vectors, matrix,
# system inputs, status (in last)
# writeFile -> plotfile, status, args

# -----------------------------------------------------
# Diving in to functions for better readability 
# and convenient testing
def _getStatus(status,v,maxit,eConv):
    """ 
    Initialize and update status dictionary
    
    Status contains following information
    (i)     Inputs : eConv, maxit
    (ii)    Stage of iteration
    (iii)   Convergence info
    (iv)    Time
    (iv)    print choices

    keys: ["eConv","maxit",
    "outerIter","microIter","cumIter"],
    "isConverged","lindep","properFit",
    "startTime","runTime",
    "writeOut", "writeOut", "eShift","convertUnit"
    """
    
    statusUp = {"eConv":eConv,"maxit":maxit,
            "outerIter":0, "microIter":0,"cumIter":0,
            "isConverged":False,"lindep":False,"properFit":True,
            "startTime":time.time(), "runTime":0.0,
            "writeOut":True,"writePlot":True,"eShift":0.0,"convertUnit":"au",
            "flagAddition":v.hasExactAddition,
            "futileRestart":0}
    
    if status is not None:
        givenkeys = status.keys()
    
        for item in givenkeys:       # overwrite defaults
            if item in status:
                statusUp[item] = status[item]
    
    return statusUp


def generateSubspace(Hop,Ylist,sigma,eConv):
    ''' Builds Krylov space with solving linear system
    and subsequent normalization after checking norm > 0.001*eConv'''

    typeClass = Ylist[0].__class__
    Ysolved = typeClass.solve(Hop,Ylist[-1],sigma)
    if typeClass.norm(Ysolved) > 0.001*eConv:
        Ylist.append(typeClass.normalize(Ysolved))
    else:
        Ylist.append(Ysolved)
        print("Alert: Not normalizing add basis; norm <=0.001*eConv")
    return Ylist

def transformationMatrix(Ylist,S,status):
    ''' Calculates transformation matrix from 
    overlap matrix in Ylist basis
    In: Ylist (list of basis)
    lindep (default value is 1e-14, lowdinOrtho())
    S: previous overlap matrix (for extension purpose)
    
    Out: status (dict: updated lindep)
    uS: transformation matrix
    Additional: prints overlap matrix in detailed 
    output file ("iterations.out", default)'''
    
    typeClass = Ylist[0].__class__
    S = typeClass.extendOverlapMatrix(Ylist,S)
    if status["writeOut"]:
        writeFile("out",status,"iteration")
        writeFile("out",status,"overlap",S)
    linIndep, uS = lowdinOrtho(S)
    status["lindep"] = not linIndep
    return status, uS, S
    
def diagonalizeHamiltonian(Hop,bases,X,qtAq,status):
    ''' Calculates matrix representation of Hop,
    forms truncated matrix (Hmat)
    and finally solves eigenvalue problem for Hmat

    In: Hop -> Hamiltonain operator 
        bases -> list of basis
        X -> transformation matrix
        qtAq -> previous matrix representation 
                (for extension purpose)

    Out: Hmat -> Hamiltonian matrix represenation
                 (mainly for unit tests)
         ev -> eigenvalues
         uv -> eigenvectors
    Additional: prints Hamiltonian matrix, 
                eigenvalues in detailed 
    output file ("iterations.out", default)'''

    typeClass = bases[0].__class__
    qtAq = typeClass.extendMatrixRepresentation(Hop,bases,qtAq)   
    Hmat = X.T.conj()@qtAq@X                      
    ev, uv = sp.linalg.eigh(Hmat)  
    if status["writeOut"]:
        writeFile("out",status,"hamiltonian",Hmat)
        writeFile("out",status,"eigenvalues",ev)
    return Hmat,ev,uv,qtAq


def _convergence(value,ref):
    ''' Computes convergence quantity (absolute error or 
    relative error, current one is relative error )'''
    
    check_ev = abs(value - ref)/max(abs(value), 1e-14)
    #if absConvergenc:check_ev = abs(ev_nearest - ref)    
    return check_ev


def checkConvergence(ev,ref,sigma,eConv,status):
    ''' Checks eigenvalue convergence

    In: ev -> eigenvalues
        ref -> eigenvalue of last iteration
               (i-1 th eigenvalue)
        sigma -> eigenvalue target
        eConv -> convergence threshold 
    
    Out: status (dict: updated isConverged)
         idx -> index of the nearest eigenvalue 
         ref -> updated eigenvalue reference for 
         next convergence check'''
    
    isConverged = False
    idx, ev_nearest = find_nearest(ev,sigma)
    if _convergence(ev_nearest,ref) <= eConv: isConverged = True
    status["isConverged"] = isConverged
    status["runTime"] = time.time() - status["startTime"]
    if status["writePlot"]:
        writeFile("plot",status,ev_nearest,ref)
    ref = ev_nearest
    return status, idx, ref
 
def basisTransformation(newBases,coeffs):
    ''' basis transformation with eigenvectors 
    and Krylov bases'''

    typeClass = newBases[0].__class__
    ndim = coeffs.shape
    oldBases = []
    if len(ndim)==1:
        oldBases.append(typeClass.linearCombination(newBases,coeffs))
    else:
        for j in range(ndim[1]):
            oldBases.append(typeClass.linearCombination(newBases,coeffs[:,j]))
    return oldBases

def checkFitting(qtAq, ev_nearest, eConv, status):
    ''' Checks the eigenvalue after fitting
    (at the end of Lanczos iteration)
    In : qtAq -> matrix element of the basis nearest to sigma
         ev_nearest -> nearest eigenvalue form previous iteration
    
    Out: status -> (dict: updates properFit)
    '''
    if status["flagAddition"]:
        status["properFit"] = True
    else:
        if _convergence(qtAq[0],ev_nearest) > eConv:
            status["properFit"] = False
    return status

def checkRestart(status,qtAq,ref):
    """ If energy has not changed up to 
    at least third decimal place, counts a ineffective restart """
    decision = False
    if _convergence(qtAq[0],ref) < 1e-4:
        status["futileRestart"] += 1
    if status["futileRestart"] > 3:
        decision = True
    return decision


def analyzeStatus(status):
    ''' Wrapper of all decision parameters for iteration
        e.g., isConverged, properFit, lindep'
        in a separate function and conclude to a single 
        bool param continueIteration
        to make main function clean'''

    isConverged = status["isConverged"]
    lindep = status["lindep"]
    it = status["outerIter"]
    maxit = status["maxit"]
    continueIteration = True
    
    if isConverged:
        continueIteration = False
    if isConverged and it == maxit -1: 
        print("Alert: Lanczos iterations is not converged!")
    if not status["properFit"]:
        print("Alert: Linearcombination inaccurate")
        continueIteration = False

    return continueIteration
# -----------------------------------------------------

# -----------------------------------------------------
#    Inexact Lanczos with AbstractClass interface
#------------------------------------------------------

def inexactDiagonalization(H,v0,sigma,L,maxit,eConv,status=None):
    '''
    This is core function to calculate eigenvalues and eigenvectors
    with inexact Lanczos method


    ---Doing inexact Lanczos in canonical orthogonal basis.---
    Input::  H => diagonalizable input matrix or linearoperator
             v0 => eigenvector guess
             sigma => eigenvalue estimate
             L => Krylov space dimension
             maxit => Maximum Lanczos iterations
             eConv => relative eigenvalue convergence tolerance
             status (optional) => Additional information dictionary
                    (more details see _getStatus doc)

    Output:: ev as inexact Lanczos eigenvalues
             uv as inexact Lanczos eigenvectors
             status for convergence information
    '''
    
    typeClass = v0.__class__
    Ylist = [typeClass.normalize(v0)]
    S = typeClass.overlapMatrix(Ylist)
    qtAq = typeClass.matrixRepresentation(H,Ylist)
    ref = np.inf; nCum = 0
    status = _getStatus(status,Ylist[0],maxit,eConv)
  
    for it in range(maxit):
        status["outerIter"] = it
        for i in range(1,L):
            nCum += 1
            status["microIter"] = i
            status["cumIter"] = nCum
            
            Ylist = generateSubspace(H,Ylist,sigma,eConv)
            status, uS, S = transformationMatrix(Ylist,S,status)
            if status['lindep']:
                print("Restarting calculation: Got linearly dependent basis!")
                Ylist = Ylist[:-1]
                S = S[:-1,:-1]
                status['lindep'] = False #For restart!
                break
            ev, uv, qtAq = diagonalizeHamiltonian(H,Ylist,uS,qtAq,status)[1:4]
            status,idx,ref = checkConvergence(ev,ref,sigma,eConv,status)
            continueIteration = analyzeStatus(status)
            uSH = uS@uv
            
            if not continueIteration:
                break
        
        if not continueIteration:
            Ylist = basisTransformation(Ylist,uSH)
            break
        else:
            y = basisTransformation(Ylist,uSH[:,idx])
            Ylist = [typeClass.normalize(y[0])]
            S = typeClass.overlapMatrix(Ylist)
            qtAq=typeClass.matrixRepresentation(H,Ylist)
            status = checkFitting(qtAq,ev[idx],eConv,status)
            if checkRestart(status,qtAq,ref): break

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

    optionDict = {"linearSolver":"gcrotmk","linearIter":1000,"linear_tol":1e-04}
    status = {"writeOut": False,"writePlot": False}
    Y0 = NumpyVector(np.random.random((n)),optionDict)
    sigma = target

    headerBot("Inexact Lanczos")
    print("{:50} :: {: <4}".format("Sigma",sigma))
    print("{:50} :: {: <4}".format("Krylov space dimension",L+1))
    print("{:50} :: {: <4}".format("Eigenvalue convergence tolarance",eConv))
    print("\n")
    t1 = time.time()
    lf,xf,status =  inexactDiagonalization(A,Y0,sigma,L,maxit,eConv,status)
    t2 = time.time()

    print("{:50} :: {: <4}".format("Eigenvalue nearest to sigma",round(find_nearest(lf,sigma)[1],8)))
    print("{:50} :: {: <4}".format("Actual eigenvalue nearest to sigma",round(find_nearest(ev,sigma)[1],8)))
    print("{:50} :: {: <4}".format("Time taken (in sec)",round((t2-t1),2)))
    headerBot("Lanczos",yesBot=True)
