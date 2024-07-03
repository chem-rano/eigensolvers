import numpy as np
import scipy as sp
from util_funcs import find_nearest, lowdinOrtho
from printUtils import writeFile
import warnings
import time
import util
from numpyVector import NumpyVector
from util_funcs import headerBot
from util_funcs import pickStates_sigma, pickStates_maxOvlp

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
            "flagAddition":vector.hasExactAddition,
            "outerIter":0, "innerIter":0,"cumIter":0,
            "isConverged":False,"lindep":False,
            "futileRestart":0,
            "startTime":time.time(), "runTime":0.0,
            "writeOut":True,"writePlot":True,"eShift":0.0,"convertUnit":"au",
            "stateFollowing":"sigma","ovlpRef":None}
    
    if status is not None:
        givenkeys = status.keys()
    
        for item in givenkeys:       # overwrite defaults
            if item in status:
                statusUp[item] = status[item]
    
    return statusUp


def generateSubspace(Hop,Ylist,sigma,eConv):
    ''' Builds Krylov space with solving linear system
    and subsequent normalization after checking norm > 0.001*eConv

    In: Hop -> Operator (either as matrix or linearOperator)
        Ylist -> List of Krylov vectors
        sigma -> Eigenvalue target
        eConv -> Eigenvalue convergence

    Out: Ylist -> Updates list of vectors'''

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
         S: exatended overlap matrix
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

    In: Hop -> Operator (either as matrix or linearOperator)
        bases -> list of basis
        X -> transformation matrix
        qtAq -> previous matrix representation 
                (for extension purpose)

    Out: Hmat -> Matrix represenation
                 (mainly for unit tests)
         ev -> eigenvalues
         uv -> eigenvectors
    Additional: prints matrix representation, 
                eigenvalues in detailed 
    output file ("iterations.out", default)'''

    typeClass = bases[0].__class__
    qtAq = typeClass.extendMatrixRepresentation(Hop,bases,qtAq)   
    Hmat = X.T.conj()@qtAq@X                      
    ev, uv = sp.linalg.eig(Hmat) #NOTE eig is slower 
    ev = ev.real; uv = uv.real
    #ev, uv = sp.linalg.eigh(Hmat)  
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


def checkConvergence(vectors,ev,status):
    ''' Checks eigenvalue convergence
    
    In: vectors -> Krylov vectors (for overalp evaluation)
        ev -> eigenvalues
        status -> params dictionary
    
    Out: status (dict: updated isConverged, ref)
         idx -> index of the nearest eigenvalue
         '''
    
    isConverged = False
    if status["stateFollowing"]=="maxOvlp":
        idx = pickStates_maxOvlp(vectors,status)
    else:
        idx = pickStates_sigma(ev,status["sigma"])
    ev_nearest = ev[idx]
    if _convergence(ev_nearest,status["ref"][-1]) <= status["eConv"]:
        isConverged = True
    status["isConverged"] = isConverged
    status["runTime"] = time.time() - status["startTime"]
    if status["writePlot"]:
        writeFile("plot",status,ev_nearest,status["ref"][-1])
    status["ref"].append(ev_nearest)
    if len(status["ref"]) > 2:status["ref"].pop(0)
    return status, idx
 
def basisTransformation(bases,coeffs):
    ''' Basis transformation with eigenvectors 
    and Krylov bases

    In: bases -> List of bases for combination
        coeffs -> coefficients used for the combination

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

def properFitting(qtAq, ev_nearest, status):
    ''' Checks the eigenvalue after fitting
    (at the end of Lanczos iteration)
    In : qtAq -> Hamiltonian matrix element of nearest eigenvector 
                after fitting
         ev_nearest -> nearest eigenvalue before fitting
         status -> Param dictionary
    
    Out: properFit -> (bool: True for accurate linear combination)
    '''
    properFit = True
    
    if status["flagAddition"]:
        properFit = True
    else:
        if _convergence(qtAq[0],ev_nearest) > status["eConv"]:
           properFit = False
           print("Alert: Linearcombination inaccurate")
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
    status = _getStatus(status,Ylist[0],sigma,maxit,L,eConv)
  
    for it in range(maxit):
        status["outerIter"] = it
        for i in range(1,L):
            status["innerIter"] = i
            status["cumIter"] += 1
            
            Ylist = generateSubspace(H,Ylist,sigma,eConv)
            status, uS, S = transformationMatrix(Ylist,S,status)
            if status['lindep']:
                print("Restarting calculation: Got linearly dependent basis!")
                Ylist = Ylist[:-1] # Excluding the last vector added to the Ylist
                break
            ev, uv, qtAq = diagonalizeHamiltonian(H,Ylist,uS,qtAq,status)[1:4]
            status,idx = checkConvergence(Ylist,ev,status)
            continueIteration = analyzeStatus(status)
            uSH = uS@uv
            
            if not continueIteration:
                break
        
        if not continueIteration:
            Ylist = basisTransformation(Ylist,uSH)
            break
        else:
            y = basisTransformation(Ylist,uSH[:,idx])
            YlistNew = [typeClass.normalize(y[0])]
            S = typeClass.overlapMatrix(YlistNew)
            qtAq=typeClass.matrixRepresentation(H,YlistNew)
            if not properFitting(qtAq,ev[idx],status):break
            if terminateRestart(qtAq[0],status):break
            Ylist = YlistNew # when Lanczos iteration continues

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
    status = {"writeOut": False,"writePlot": False, "stateFollowing":"sigma"}
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
