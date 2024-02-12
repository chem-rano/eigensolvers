import numpy as np
import scipy
from scipy.sparse.linalg import LinearOperator
from scipy import special
from scipy import linalg as la
from util_funcs import getRes, print_a_range, quad_func, resEigenvalue
from util_funcs import linearDepedency
from numpyVector import NumpyVector
import time


# ***************************************************
# Part 1: main FEAST function for contour integral
# ------------------------------
def feast_core_interface(A,Y,nc,quad,rmin,rmax,eps,maxit):

    typeClass = Y[0].__class__
    m0 = len(Y)
    n = Y[0].size
    #I = np.eye(n)  # no need to store
    #Y1 = Y.copy()     # no need to copy
    ev = np.zeros(m0)
    prev_ev = np.zeros(m0)
    Q = [np.nan for it in range(m0)]
    #Q = typeClass.zeros_like(Y1,dtype = complex)

    # numerical quadrature points.
    gk,wk = quad_func(nc,quad)
    pi = np.pi

    for i in range(maxit):
        for k in range(nc):
            theta = -(pi*0.5)*(gk[k]-1)
            z = (rmin+rmax)/2+ np.exp(1.0j*theta)
            linOp = LinearOperator((n,n), matvec = lambda x, z=z, A=A: z*(np.eye(n))@x-A@x,dtype=complex)
            
            for jloop in range(m0):
                #b_in = (Y[jloop]).applyOp(B)    # no need of multiplication
                b_in = Y[jloop]
                Qkjloop = typeClass.solve(linOp,b_in, x0=None)
                
                Qadd = np.real((Qkjloop*((rmin-rmax)/2*np.exp(1j*theta))).array)*(wk[k]*-0.5)
                Qadd = NumpyVector(Qadd,b_in.optionsDict)
                if k == 0:
                    Q[jloop] = Qadd
                else:
                    Q[jloop] = typeClass.linearCombination([Q[jloop],Qadd],[1.0,1.0])
        
        # ------------------------------------- 
        # eigh in Lowdin orthogonal basis
        qtq = typeClass.overlapMatrix(Q)
        info, uvqTraf = linearDepedency(qtq, tol = 1e-12)
        m = uvqTraf.shape[1]
        for i in range(m):
            Q[i] = typeClass.linearCombination(Q,uvqTraf[:,i])
        qtAq = typeClass.matrixRepresentation(A,Q)
        ev, uv = la.eigh(qtAq)
        
        m0 = len(Q)
        for ivec in range(m0):
            Q[ivec] = typeClass.linearCombination(Q,uv[:,ivec])
        # ------------------------------------- 

        if i == 0: 
            res = None   # Initialize res as None
        else:
            #calculate eigenvalue residuals
            res = resEigenvalue(ev,prev_ev)
            
            #calculate eigenvector residuals
            #R = typeClass.resvecs(A,Q,ev)
            #res = typeClass.resEigenvector(ev,Q,R,eps)

            print("{:10}{:26}".format(i,res))

            if res < eps:
                break
        Y = Q
        prev_ev = ev

        
    return Q,ev


if __name__ == "__main__":
    # ***************************************************
    # Part 1: Call FEAST program with parameter specifications
    # ------------------------------
    n = 100
    ev = np.linspace(1,200,n)
    np.random.seed(10)
    Q = la.qr(np.random.rand(n,n))[0]
    A = Q.T @ np.diag(ev) @ Q
    linOp = LinearOperator((n,n), matvec = lambda x, A=A: A@x)

    # Specify FEAST parameters
    ev_min = 180.0
    ev_max = 184.0
    nc    = 4          # number of contour points
    quad  = "legendre" # Choice of quadrature points # available options, legendre, Hermite (, trapezoidal !)
    m0    = 1         # subspace dimension
    eps   = 1e-08      # residual convergence tolerance
    maxit = 5         # maximum FEAST iterations
    optionsDict = {"linearSolver":"gcrotmk","linearIter":1000,"linear_tol":1e-02}
    
    Y = []
    np.random.seed(10)
    aVector = NumpyVector(np.random.random((n)),optionsDict)
    for i in range(m0):
        Y.append(aVector.copy())

    contour_ev = print_a_range(ev, ev_min, ev_max)
    print("actual",contour_ev)
    qIn,efeast =  feast_core_interface(linOp,Y,nc,quad,ev_min,ev_max,eps,maxit)
    print("feast",efeast)
