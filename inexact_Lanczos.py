import numpy as np
import scipy
from scipy import linalg as la
from scipy.sparse.linalg import LinearOperator
from util_funcs import find_nearest, headerBot
from util_funcs import lowdinOrtho, convert
from printUtils import _writeFile
import warnings
import time
import util
from itertools import compress

# -----------------------------------------------------
#    Inexact Lanczos with AbstractClass interface
#------------------------------------------------------

def inexactDiagonalization(H,v0,sigma,L,maxit,eConv,zpve,startTime,files):
    '''
    This is core function to calculate eigenvalues and eigenvectors
    with inexact Lanczos method


    ---Doing inexact Lanczos in canonical orthogonal basis.---
    Input::  H => diagonalizable input matrix or linearoperator
             sigma => eigenvalue estimate 
             v0 => eigenvector guess
             eConv => eigenvalue convergence tolerance
             zpve => zero-point vibrational energy

    Output:: ev as inexact Lanczos computed eigenvalues
             uv as inexact Lanczos computed eigenvectors

    All energies in a.u.
    '''
    
    typeClass = v0.__class__
    Ylist = [typeClass.normalize(v0)]
    exlevel_last = np.inf # for convergence check
    isConverged = False
    nCum = 0
    sigmaAU = util.unit2au((sigma+zpve),"cm-1")    # in a.u.; since H in  a.u.
    _writeFile(files["plot"],"it","i","nCum",sep="\t",endline=False)
    _writeFile(files["plot"],"ev_nearest","check_ev","rel_ev","time",sep="\t")
  
  
    for it in range(maxit):
        for i in range(1,L):
            nCum += 1
            _writeFile(files["out"],"Lanczos iteration",it+1,"Krylov iteration",i,endline=False)
            _writeFile(files["out"],"Cumulative Krylov iteration",nCum)

            Ysolved = typeClass.solve(H,Ylist[i-1],sigmaAU)
            Ylist.append(typeClass.normalize(Ysolved))    # Do we need it? Canonical: orthonormalization
            
            S = typeClass.overlapMatrix(Ylist)            
            uS = lowdinOrtho(S)[1]                       
            m = uS.shape[1] 
            
            qtAq = typeClass.matrixRepresentation(H,Ylist)  
            Hmat = uS.T.conj()@qtAq@uS                      
            ev, uv = la.eigh(Hmat)                              # Hmat in a.u.=> ev in a.u.
            exlevels = convert(ev,"cm-1",zpve)                  # excitation levels in cm-1
            idx, exlevel_nearest = find_nearest(exlevels,sigma)
            check_ev = abs(exlevel_nearest-exlevel_last)        # excitation levels are alright for check 
            
            _writeFile(files["out"],"OVERLAP MATRIX")
            _writeFile(files["out"],S)
            _writeFile(files["out"],"HAMILTONIAN MATRIX")
            _writeFile(files["out"],convert(Hmat,"cm-1"))
            _writeFile(files["out"],"Eigenvalues")
            
            for ivalue in range(0,len(exlevels),1):
                _writeFile(files["out"],exlevels[ivalue])
            _writeFile(files["plot"],it,i,nCum,sep="\t",endline=False)
            _writeFile(files["plot"],exlevel_nearest,sep="\t",endline=False)
            _writeFile(files["plot"],check_ev,check_ev/exlevel_nearest,time.time()-startTime,sep="\t")
            
            if check_ev <= eConv:
                break
            exlevel_last = exlevel_nearest

        if check_ev <= eConv:
            isConverged = True
            x  = []
            for j in range(m):
                x.append(typeClass.linearCombination(Ylist,uS[:,j]))
            Ylist = x
            break
        else:
            x = [typeClass.linearCombination(Ylist,uS[:,idx])]
            Ylist = x
            _writeFile(files["out"],"Choosing eigevector [idx ",idx,"]: Eigenvalue ",endline=False)
            _writeFile(files["out"],exlevel_nearest)

        if (it == maxit-1) and (not isConverged):
            print("Alert:: Lanczos iterations is not converged!")

        _writeFile(files["out"])# blank line

    return exlevels,Ylist
# -----------------------------------------------------
if __name__ == "__main__":
    n = 100
    ev = np.linspace(1,300,n)
    np.random.seed(10)
    Q = la.qr(np.random.rand(n,n))[0]
    A = Q.T @ np.diag(ev) @ Q

    
    sigma = 30
    maxit = 4
    L  = 3
    conv_tol = 1e-08
    optionDict = {"linearSolver":"gcrotmk","linearIter":1000,"linear_tol":1e-04}

    Y0 = NumpyVector(np.random.random((n)),optionDict)

    headerBot("Inexact Lanczos")
    print("{:50} :: {: <4}".format("Sigma",sigma))
    print("{:50} :: {: <4}".format("Krylov space dimension",L+1))
    print("{:50} :: {: <4}".format("Eigenvalue convergence tolarance",conv_tol))
    print("\n")
    t1 = time.time()
    lf,xf =  inexactDiagonalization(A,Y0,sigma,L,maxit,conv_tol)
    t2 = time.time()

    print("{:50} :: {: <4}".format("Eigenvalue nearest to sigma",round(find_nearest(lf,sigma)[1],8)))
    print("{:50} :: {: <4}".format("Actual eigenvalue nearest to sigma",round(find_nearest(ev,sigma)[1],8)))
    print("{:50} :: {: <4}".format("Time taken (in sec)",round((t2-t1),2)))
    headerBot("Lanczos",yesBot=True)
