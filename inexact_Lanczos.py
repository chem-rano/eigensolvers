import numpy as np
import scipy as sp
from util_funcs import find_nearest, lowdinOrtho
from printUtils import *
import warnings
import time
import util


# -----------------------------------------------------
# Diving in to functions for better readability 
# and convenient testing
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

def transformationMatrix(Ylist,status,S,printChoices):
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
    if printChoices["writeOut"]:
        writeFile("out","overlap",S,choices=printChoices)
    linIndep, uS = lowdinOrtho(S)
    status["lindep"] = not linIndep
    return status, uS, S
    
def diagonalizeHamiltonian(Hop,bases,X,qtAq,printChoices):
    ''' Calculates matrix representation of Hop (qtAq),
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
    if printChoices["writeOut"]:
        writeFile("out","hamiltonian",Hmat,choices=printChoices)
        writeFile("out","eigenvalues",ev,choices=printChoices)
    return Hmat,ev,uv,qtAq


def _convergence(value,ref):
    ''' Computes convergence quantity (absolute error or 
    relative error, current one is relative error )'''
    
    check_ev = abs(value - ref)/max(abs(value), 1e-14)
    #if absConvergenc:check_ev = abs(ev_nearest - ref)    
    return check_ev


def checkConvergence(ev,ref,sigma,eConv,status,printChoices):
    ''' checks eigenvalue convergence

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
    status["endTime"] = time.time()
    if printChoices["writePlot"]:
        writePlotfile(status,ev_nearest,ref)
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
    if _convergence(qtAq[0],ev_nearest) > eConv:
        status["properFit"] = False
    return status

def analyzeStatus(status):
    ''' Wrapper of all decision parameters for iteration
        e.g., isConverged, properFit, lindep'
        in a separate function and conclude to a single 
        bool param continueIteration
        to make main function clean'''

    it = status["iteration"]
    isConverged = status["isConverged"]
    lindep = status["lindep"]
    maxit = status["maxit"]
    continueIteration = True
    
    if isConverged or lindep:
        continueIteration = False
    if status['isConverged'] and status['iteration'] == maxit -1: 
        print("Alert: Lanczos iterations is not converged!")
    if status['lindep']: print("Alert: Got linear dependent basis!")
    if not status["properFit"]:
        print("Alert: Linearcombination inaccurate")
        continueIteration = False

    return continueIteration
# -----------------------------------------------------

# -----------------------------------------------------
#    Inexact Lanczos with AbstractClass interface
#------------------------------------------------------

def inexactDiagonalization(H,v0,sigma,L,maxit,eConv,printChoices=None):
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

    Output:: ev as inexact Lanczos eigenvalues
             uv as inexact Lanczos eigenvectors
    '''
    
    typeClass = v0.__class__
    Ylist = [typeClass.normalize(v0)]
    S = typeClass.overlapMatrix(Ylist)
    qtAq = typeClass.matrixRepresentation(H,Ylist)
    ref = np.inf
    nCum = 0
    status = {"eConv":eConv,"maxit":maxit,"properFit":True,
            "startTime":time.time()}
  
    for it in range(maxit):
        status["iteration"] = it
        for i in range(1,L):
            nCum += 1
            status["microIteration"] = i
            status["cummulativeIteration"] = nCum
            if printChoices["writeOut"]:
                writeFile("out","iteration",it,i,nCum)
            
            Ylist = generateSubspace(H,Ylist,sigma,eConv)
            status, uS, S = transformationMatrix(Ylist,status,S,printChoices)
            ev, uv, qtAq = diagonalizeHamiltonian(H,Ylist,uS,qtAq,printChoices)[1:4]
            status,idx,ref = checkConvergence(ev,ref,sigma,eConv,status,printChoices)
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

    return ev,Ylist,status
