import numpy as np
import scipy
from scipy import linalg as la
from scipy.sparse.linalg import LinearOperator
from util_funcs import _qr, find_nearest, eigRegularized_list, nearest_degenerate
import warnings

from myVector import myVector

# -----------------------------------------------------
def solve_wk(sigma,H,b,gcrot_tol,gcrot_iter):
    # Solve (EI-H)w = v with iterative solver
    # w ==> x0; v ==> b_in; linOp ==> (EI-H)
    # Ax =b CGS solver :: A == (n x n); x == (n x 1) and b ==> (n x 1)  
    
    n = H.shape[0]
    sigma = sigma*np.eye(n)
    linOp = LinearOperator((n,n),lambda x, sigma=sigma, H=H:(sigma@x - H@x))
    wk,conv = scipy.sparse.linalg.gcrotmk(linOp,b,x0=None, tol=gcrot_tol,atol=gcrot_tol,maxiter = gcrot_iter)
    #wk,conv = scipy.sparse.linalg.gmres(linOp,b,x0=None, tol=gcrot_tol,atol=gcrot_tol,maxiter = gcrot_iter)
    #print(type(wk))
    if conv != 0:
        # adding a single entry into warnings filter
        warnings.simplefilter('error', UserWarning)
        warnings.warn("Warning:: Iterative solver is not converged ")
    return wk

# -----------------------------------------------------
def core_func(H,v0,sigma,L,maxit,conv_tol,eigTol,gcrot_tol,gcrot_iter):
    '''
    This is core function to calculate eigenvalues and eigenvectors
    Input::  H => diagonalizable input matrix or linearoperator
             sigma => eigenvalue estimate 
             v0 => eigenvector guess
             conv_tol => eigenvalue convergence tolerance
             eigTol => tolerance to keep the eigenvectors 

    Output:: ev as inexact Lanczos computed eigenvalues
             uv as inexact Lanczos computed eigenvectors
    '''
    
    print(type(v0))
    n = v0.size
    Ylist = []
    #Ylist.append(v0/la.norm(v0))        # step 1
    #Ylist.append(v0.divide(v0.norm()))  # step 2
    Ylist.append(v0/v0.norm())           # step 3
    ev_last = np.inf # for convergence check
  
  
    for it in range(maxit):
        # Krylov-space propagation:: Ax0, A^2x0, and up to basis size L
        for i in range(1,L):
            print("it",it,"i",i)

            # TODO each element in the list are now myVector, not np.array
            # Then how to include solve_wk() and _qr() in the class? Let's go with util_funcs
            # No, the above way is wrong. Because the _qr() is different for vector and ttns, it needs abstraction
            # So, myVector consist _qr(), myObj._qr(list of myvectors) then returns orthogonal Gram-Schimdt myvectors
            Ylist.append(solve_wk(sigma,H,Ylist[i-1],gcrot_tol[it],gcrot_iter))
            
            # Send last Ylist vector to make new one through iterative solver
            # If QR algorithm finds linear dependent vectors out of them, it prunes them
            # Orthogonalize the Krylov space

            Ylist[:i+1] = _qr(Ylist[:i+1],np.dot)[0]
            ev, uv = eigRegularized_list(H,None,Ylist,eigTol)
            # Find closest ev and check if this value is converged
            idx, ev_nearest = find_nearest(ev,sigma)
            
            check_ev = abs(ev_nearest-ev_last)
            if (check_ev <= conv_tol):
                break                # Break to Krylov space expansion
            # Update the last eigenvalue for convergence check
            ev_last = ev_nearest
            lenY = len(Ylist)
        # If not converged, continue to next iteration with x0 guess as nearest eigenvector
        if (check_ev <= conv_tol):
            break
        else:
            Ylist = [uv[idx,:]]
    #print("Eigenvalue nearest to sigma, ",sigma," : is ", ev_nearest)
        
    return ev,uv
# -----------------------------------------------------
if __name__ == "__main__":
    n = 100
    ev = np.linspace(1,300,n)
    np.random.seed(10)
    Q = la.qr(np.random.rand(n,n))[0]
    A = Q.T @ np.diag(ev) @ Q

    Y0  = np.random.random((n))
    Y0 = myVector(Y0)

    sigma = 40
    maxit = 4
    L  = 20
    conv_tol = 1e-08
    atol = 1e-12
    eigTol = 1e-10
    gcrot_tol = 1e-04
    gcrot_iter = 1000

    lf,xf =  core_func(A,Y0,sigma,L,maxit,conv_tol,eigTol,gcrot_tol,gcrot_iter)
    print(lf)
