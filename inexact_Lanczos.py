import numpy as np
import scipy
from scipy import linalg as la
from scipy.sparse.linalg import LinearOperator
from util_funcs import find_nearest, headerBot
from util_funcs import lowdinOrtho  
import warnings
from numpyVector import NumpyVector
import time

# -----------------------------------------------------
#    Inexact Lanczos with AbstractClass interface
#------------------------------------------------------

def inexactDiagonalization(H,v0,sigma,L,maxit,conv_tol,proceed_ortho:bool = False):
    '''
    This is core function to calculate eigenvalues and eigenvectors
    with inexact Lanczos method
    Input::  H => diagonalizable input matrix or linearoperator
             sigma => eigenvalue estimate 
             v0 => eigenvector guess
             conv_tol => eigenvalue convergence tolerance

    Output:: ev as inexact Lanczos computed eigenvalues
             uv as inexact Lanczos computed eigenvectors
    '''
    
    n = v0.size
    dtype = v0.dtype
    
    Ylist = []
    Ylist.append(v0/v0.norm())
    typeClass = v0.__class__
    ev_last = np.inf # for convergence check
    isConverged = False
    if not proceed_ortho: print("!!! ATTENTION: Doing inexact Lanczos in non-orthogonal basis. \n")
  
  
    for it in range(maxit):
        for i in range(1,L):
            Ysolved = typeClass.solve(H,Ylist[i-1],sigma)
           
            if not proceed_ortho:
                Ylist.append(Ysolved)
                qtq = typeClass.overlapMatrix(Ylist)
                uQ = lowdinOrtho(qtq)[1]
                
                m = uQ.shape[1]
                Ylist_trun = []
                for ivec in range(m):
                    Ylist_trun.append(typeClass.linearCombination(Ylist,uQ[:,ivec]))
                qtAq = typeClass.matrixRepresentation(H,Ylist_trun)
                ev, uvals = la.eigh(qtAq)
                uv = uQ@uvals
                Ylist = Ylist_trun


            else:
                item = typeClass.orthogonalize_against_set(Ysolved,Ylist)
                if item is not None:
                    Ylist.append(item)
                    qtAq = typeClass.matrixRepresentation(H,Ylist)
                    ev, uv = la.eigh(qtAq)
                    
                else:
                    warnings.warn("Linear dependency problem, abort current Lanczos iteration and restart.")
                    break
        
            # Find closest ev and check if this value is converged
            idx, ev_nearest = find_nearest(ev,sigma)
            check_ev = abs(ev_nearest-ev_last)
                    
            if (check_ev <= conv_tol):
                break                # Break to Krylov space expansion
                    
            ev_last = ev_nearest     # Update the last eigenvalue for convergence check


       # If not converged, continue to next iteration with x0 guess as nearest eigenvector
        if (check_ev <= conv_tol):
            isConverged = True
            
            m = len(Ylist)
            x = []
            for j in range(m):
                x.append(typeClass.linearCombination(Ylist,uv[:,j]))
            Ylist = x
            break
        else:
            Ylist = [typeClass.linearCombination(Ylist,uv[:,idx])]

        if (it == maxit-1) and (not isConverged):
            print("Alert:: Lanczos iterations is not converged!")
        
    return ev,Ylist
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
