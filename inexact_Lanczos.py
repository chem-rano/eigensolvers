import numpy as np
import scipy
from scipy import linalg as la
from scipy.sparse.linalg import LinearOperator
from util_funcs import find_nearest, headerBot
import warnings
from numpyVector import NumpyVector
import time

# -----------------------------------------------------
#    Inexact Lanczos with NumpyVector interface
#------------------------------------------------------

def core_func(H,v0,sigma,L,maxit,conv_tol):
    '''
    This is core function to calculate eigenvalues and eigenvectors
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
  
  
    for it in range(maxit):
        # Krylov-space propagation:: Ax0, A^2x0, and up to basis size L
        for i in range(1,L):
            #print("it",it,"i",i)
            # each element in the list are now NumpyVector, not np.array
            Ylist.append(typeClass.solve(H,Ylist[i-1],sigma))
            
            # Orthogonalize the Krylov space
            Ylist = typeClass.orthogonalize(Ylist[:i+1])
            
            m = len(Ylist)
            qtAq = np.zeros((m,m),dtype=dtype)

            # This matrix formation can be included as a function: formMat
            for j in range(m):
                ket = Ylist[j].applyOp(H)
                for i in range(m):
                    qtAq[i,j] = Ylist[i].vdot(ket)
                    qtAq[j,i] = qtAq[i,j]

            ev, uvals = la.eigh(qtAq)
            uv = []
            for j in range(m):
                uv.append(typeClass.linearCombination(Ylist,uvals[:,j]))
        
            # Find closest ev and check if this value is converged
            idx, ev_nearest = find_nearest(ev,sigma)
            check_ev = abs(ev_nearest-ev_last)
            if (check_ev <= conv_tol):
                break                # Break to Krylov space expansion
            # Update the last eigenvalue for convergence check
            ev_last = ev_nearest
       

       # If not converged, continue to next iteration with x0 guess as nearest eigenvector
        if (check_ev <= conv_tol):
            break
        else:
            Ylist = [uv[idx]]
        
    return ev,uv
# -----------------------------------------------------
if __name__ == "__main__":
    n = 100
    ev = np.linspace(1,300,n)
    np.random.seed(10)
    Q = la.qr(np.random.rand(n,n))[0]
    A = Q.T @ np.diag(ev) @ Q

    
    sigma = 100
    maxit = 4
    L  = 8
    conv_tol = 1e-08
    optionDict = {"linearSolver":"gcrotmk","linearIter":1000,"linearTol":1e-04}

    Y0 = NumpyVector(np.random.random((n)),optionDict)

    headerBot("Inexact Lanczos")
    print("{:50} :: {: <4}".format("Sigma",sigma))
    print("{:50} :: {: <4}".format("Krylov space dimension",L+1))
    print("{:50} :: {: <4}".format("Eigenvalue convergence tolarance",conv_tol))
    print("\n")
    t1 = time.time()
    lf,xf =  core_func(A,Y0,sigma,L,maxit,conv_tol)
    t2 = time.time()

    print("{:50} :: {: <4}".format("Eigenvalue nearest to sigma",round(find_nearest(lf,sigma)[1],8)))
    print("{:50} :: {: <4}".format("Actual eigenvalue nearest to sigma",round(find_nearest(ev,sigma)[1],8)))
    print("{:50} :: {: <4}".format("Time taken (in sec)",round((t2-t1),2)))
    headerBot("Lanczos",yesBot=True)
