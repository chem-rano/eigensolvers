import numpy as np
import scipy
from scipy.sparse.linalg import LinearOperator
from scipy import special
from scipy import linalg as la
from util_funcs import print_a_range, quad_func, eigenvalueResidual
from util_funcs import lowdinOrtho
from numpyVector import NumpyVector
import time

# ***************************************************
# Part 1: main FEAST function for contour integral
# ------------------------------
def feastDiagonalization(A,Y,nc,quad,rmin,rmax,eps,maxit):
    """ FEAST diagonalization of A 

        In A     ::  matrix or linearoperator or SOP operator 
        In Y     ::  Initial guess of m0 vectors (m0 is called as subspace dimension)
        In nc    ::  number of quadrature points
        In quad  ::  quadrature points distribution
                     Avaiable options - "legendre", "hermite", "trapezoidal"
        In rmin  ::  eigenvalue lower limit
        In rmax  ::  eigenvalue upper limit
        In eps   ::  eigenvalue residual convergence tolerance
        In maxit ::  maximum feast iterations

        Out ev   ::  feast eigenvalues
        Out Y    ::  feast eigenvectors
    """

    typeClass = Y[0].__class__
    m0 = len(Y)
    r = (rmax-rmin)*0.5

    ev = np.zeros(m0)
    prev_ev = np.zeros(m0)
    Q = [np.nan for it in range(m0)]

    # numerical quadrature points.
    gk,wk = quad_func(nc,quad)
    pi = np.pi
    
    flag = Y[0].hasExactAddition
    
    for i in range(maxit):
        for k in range(nc):
            theta = -(pi*0.5)*(gk[k]-1)
            z = (rmin+rmax)/2+ np.exp(1.0j*theta)
            
            for jloop in range(m0):
                b_in = Y[jloop]
                Qkjloop = typeClass.solve(A,b_in,z)
                
                if flag:
                    Qadd = (-0.5*wk[k])*typeClass.real(r*np.exp(1j*theta)*Qkjloop)
                else:
                    part1 = np.exp(1j*theta)*Qkjloop
                    part2 = np.exp(-1j*theta)*Qkjloop
                    Qadd = (-0.25*wk[k])*r*linearCombination([part1,part2],[1.0,1.0]) 
                    #Qadd = (-0.25*wk[k])*r*((np.exp(1j*theta)*Qkjloop)+((np.exp(-1j*theta)*Qkjloop.conj()))) #HRL conj()


                if k == 0:
                    Q[jloop] = Qadd
                else:
                    Q[jloop] = typeClass.linearCombination([Q[jloop],Qadd],[1.0,1.0])
       
        # ------------------------------------- 
        # eigh in Lowdin orthogonal basis
        qtq = typeClass.overlapMatrix(Q)
        uQ = lowdinOrtho(qtq)[1]
        m0 = uQ.shape[1]  
        for ivec in range(m0):
            Q[ivec] = typeClass.linearCombination(Q,uQ[:,ivec])
        Q = Q[0:m0]
        qtAq = typeClass.matrixRepresentation(A,Q) 
        ev, uvals = la.eigh(qtAq)
        # ------------------------------------- 
    
        if i == 0: 
            res = None   # Initialize res as None
        else:
            #calculate eigenvalue residuals
            res = eigenvalueResidual(ev,prev_ev)     

            print(f"{i:10d}   {res:20.14f}")

            if res < eps:
                break
       
        uv = uQ@uvals 
        Y = []
        for jvec in range(m0):
            Y.append(typeClass.linearCombination(Q,uv[:,jvec]))
        
        prev_ev = ev

        
    return ev,Y


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
    ev_min = 10.0
    ev_max = 14.0
    nc    = 6          # number of contour points
    quad  = "legendre" # Choice of quadrature points # available options, legendre, Hermite (, trapezoidal !)
    m0    = 4         # subspace dimension
    eps   = 1e-6      # residual convergence tolerance
    maxit = 10         # maximum FEAST iterations
    optionsDict = {"linearSolver":"gcrotmk","linearIter":1000,"linear_tol":1e-02}
    
    Y = []
    np.random.seed(10)
    aVector = NumpyVector(np.random.random((n)),optionsDict)
    for i in range(m0):
        Y.append(aVector.copy())

    contour_ev = print_a_range(ev, ev_min, ev_max)
    print("actual",contour_ev)
    efeast,ufeast =  feastDiagonalization(linOp,Y,nc,quad,ev_min,ev_max,eps,maxit)
    print("feast",efeast)
