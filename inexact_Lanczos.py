import numpy as np
import scipy as sp
from util_funcs import find_nearest, headerBot, lowdinOrtho
from util_funcs import convertEnergy, convertMatrix
from printUtils import *
import warnings
import time
import util
from numpyVector import NumpyVector # delete later


# -----------------------------------------------------
# Diving in to functions for better readability 
# and convenient testing

def transformationMatrix(Ylist):
    typeClass = Ylist[0].__class__
    S = typeClass.overlapMatrix(Ylist)            
    writeFile("out","OVERLAP MATRIX")
    writeFile("out",S)
    linIndep, uS = lowdinOrtho(S)                       
    return linIndep, uS
    
def diagonalizeHamiltonian(H,bases,X,eShift):
    typeClass = bases[0].__class__
    qtAq = typeClass.matrixRepresentation(H,bases)  
    Hmat = X.T.conj()@qtAq@X                      
    evShifted, uv = sp.linalg.eigh(Hmat)  
    writeFile("out","HAMILTONIAN MATRIX")
    writeFile("out",convertMatrix(Hmat,eShift,convertUnit=False))  # manual
    ev = convertEnergy(evShifted,eShift,convertUnit=False)                      
    writeFile("out","Eigenvalues")
    writeFile("out",ev)
    return Hmat,ev,uv
    
def checkConvergence(ev,ref,sigma,eConv):
    isConverged = False
    startTime = time.time()
    idx, ev_nearest = find_nearest(ev,sigma)
    check_ev = abs(ev_nearest - ref)    
    if check_ev <= eConv: isConverged = True
    ref = ev_nearest
    writeFile("plot",ev_nearest,sep="\t",endline=False)
    writeFile("plot",check_ev,check_ev/ev_nearest,time.time()-startTime,sep="\t")
    return isConverged, idx, ref
 
def backTransform(newBases,coeffs):
    typeClass = newBases[0].__class__
    ndim = coeffs.shape
    oldBases = []
    if len(ndim)==1:
        oldBases.append(typeClass.linearCombination(newBases,coeffs))
    else:
        for j in range(ndim[1]):
            oldBases.append(typeClass.linearCombination(newBases,coeffs[:,j]))
    return oldBases
# -----------------------------------------------------

# -----------------------------------------------------
#    Inexact Lanczos with AbstractClass interface
#------------------------------------------------------

def inexactDiagonalization(H,v0,sigma,L,maxit,eConv,eShift:float=0.0):
    '''
    This is core function to calculate eigenvalues and eigenvectors
    with inexact Lanczos method


    ---Doing inexact Lanczos in canonical orthogonal basis.---
    Input::  H => diagonalizable input matrix or linearoperator
             sigma => eigenvalue estimate 
             v0 => eigenvector guess
             eConv => eigenvalue convergence tolerance
             eShift => To convert sigma to total energy, default: 0.0
             Useful for vibrational energy levels; practically excitation 
             energy is more useful to deal with and for calculation purpose 
             we convert it to total energy with eShift is zpve (zero-point 
             vibrational enrgy)

    Output:: ev as inexact Lanczos excitation energies (cm-1)
             uv as inexact Lanczos eigenvectors
    '''
    
    typeClass = v0.__class__
    Ylist = [typeClass.normalize(v0)]
    ref = np.inf
    nCum = 0
    sigmaShifted = convertEnergy(sigma,eShift,convertUnit=False)
    # Information of convergence
    status = {"isConverged":False,"maxit":0, "lindep":False}    
  
    for it in range(maxit):
        for i in range(1,L):
            nCum += 1
            writeIteration("out",it,i,nCum)
            Ysolved = typeClass.solve(H,Ylist[i-1],sigmaShifted)
            if typeClass.norm(Ysolved) > 0.001*eConv:
                Ylist.append(typeClass.normalize(Ysolved))
            else:
                Ylist.append(Ysolved)
                print("Alert: Not normalizing add basis; norm <=0.001*eConv")
            linIndep, uS = transformationMatrix(Ylist)
            if not linIndep: break
            ev, uv = diagonalizeHamiltonian(H,Ylist,uS,eShift)[1:3]
            isConverged,idx,ref = checkConvergence(ev,ref,sigma,eConv)
            uSH = uS@uv
            
            if isConverged:
                break
        
        if isConverged:
            Ylist = backTransform(Ylist,uSH)
            break
        else:
            y = backTransform(Ylist,uSH[:,idx])
            Ylist = [typeClass.normalize(y[0])]
        
        status["isConverged"] = isConverged
        status["maxit"] = it
        status["lindep"] = not linIndep

    return ev,Ylist,status
# -----------------------------------------------------
if __name__ == "__main__":
    n = 100
    ev = np.linspace(1,300,n)
    np.random.seed(10)
    Q = sp.linalg.qr(np.random.rand(n,n))[0]
    A = Q.T @ np.diag(ev) @ Q

    
    sigma = 30
    maxit = 4
    L  = 6 
    eConv = 1e-08
    optionDict = {"linearSolver":"gcrotmk","linearIter":1000,"linear_tol":1e-04}

    Y0 = NumpyVector(np.random.random((n)),optionDict)

    headerBot("Inexact Lanczos")
    print("{:50} :: {: <4}".format("Sigma",sigma))
    print("{:50} :: {: <4}".format("Krylov space dimension",L+1))
    print("{:50} :: {: <4}".format("Eigenvalue convergence tolarance",eConv))
    print("\n")
    t1 = time.time()
    lf,xf,status =  inexactDiagonalization(A,Y0,sigma,L,maxit,eConv)
    if status['isConverged'] and status['maxit'] == maxit -1: print("Alert: Lanczos iterations is not converged!")
    if status['lindep']: print("Alert: Got linear dependent basis!")
    t2 = time.time()

    print("{:50} :: {: <4}".format("Eigenvalue nearest to sigma",round(find_nearest(lf,sigma)[1],8)))
    print("{:50} :: {: <4}".format("Actual eigenvalue nearest to sigma",round(find_nearest(ev,sigma)[1],8)))
    print("{:50} :: {: <4}".format("Time taken (in sec)",round((t2-t1),2)))
    headerBot("Lanczos",yesBot=True)
