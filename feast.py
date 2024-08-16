import numpy as np
import scipy as sp
from scipy.sparse.linalg import LinearOperator
from scipy import special
from scipy import linalg as la
from util_funcs import print_a_range, quad_func, eigenvalueResidual
from util_funcs import lowdinOrtho
from numpyVector import NumpyVector
import time
from magic import ipsh

def _getStatus(status,guess,radius,maxit,nc,eConv):
    
    statusUp = {"radius":radius,"maxit":maxit,
            "nquad":nc,"eConv":eConv,
            "flagAddition":guess[0].hasExactAddition,
            "outerIter":0, "innerIter":0,"cumIter":0,
            "isConverged":False,"lindep":False,
            "startTime":time.time(), "runTime":0.0,
            "writeOut":True,"writePlot":True,"eShift":0.0,"convertUnit":"au"}
    
    if status is not None:
        givenkeys = status.keys()
    
        for item in givenkeys:       # overwrite defaults
            if item in status:
                statusUp[item] = status[item]
    
    return statusUp

def basisTransformation(bases,coeffs,rangeIdx=None,selective=False):
    ''' Basis transformation with eigenvectors 
    and feast bases

    In: bases -> List of bases for combination
        coeffs -> coefficients used for the combination
        rangeIdx -> selective elements of coeffs
        selective -> if to use rangeIdx

    Out: combBases -> combination results'''

    typeClass = bases[0].__class__
    ndim = coeffs.shape
    combBases = []
    
    if not selective:
        if len(ndim) == 1:
            combBases.append(typeClass.linearCombination(bases,coeffs))
        else:
            for j in range(ndim[1]):
                combBases.append(typeClass.linearCombination(bases,coeffs[:,j]))
    
    else:
        for j in rangeIdx:
            combBases.append(typeClass.linearCombination(bases,coeffs[:,j]))
    return combBases

def calculateQuadrature(gfVector,angle,radius,weight):
    ''' Calculates k-th quadrature for real and complex classes'''

    typeClass = gfVector.__class__
    flag = gfVector.hasExactAddition
    if flag:
        #Qadd = (-0.5*weight)*typeClass.real(radius*np.exp(1j*angle)*gfVector)
        Qadd = (0.5*weight)*typeClass.real(radius*np.exp(1j*angle)*gfVector) # solved A-z instead z-A
    else:
        part1 = gfVector 
        part2 = typeClass.conjugate(gfVector)
        c1 = np.exp(1j*angle)
        c2 = np.exp(-1j*angle)
        mult = -0.25*weight*radius
        Qadd = typeClass.linearCombination([part1,part2],[c1,c2])*mult

    return Qadd

def updateQ(Q,im0,Qadd,k):
    ''' Adds k-th quadrature solution to the existing Q'''

    typeClass = Qadd.__class__
    if k == 0:
        Q[im0] = Qadd
    else:
        Q[im0] = typeClass.linearCombination([Q[im0],Qadd],[1.0,1.0])
    return Q
       
def transformationMatrix(vectors):
    ''' Calculates transformation matrix from 
    overlap matrix in Q basis
    In: vectors (list of basis)
        lindep (default value is 1e-14, lowdinOrtho())
    
    Out: uS: transformation matrix'''
    
    typeClass = vectors[0].__class__
    S = typeClass.overlapMatrix(vectors)
    uS, idx = lowdinOrtho(S)[1:3]
    return uS, idx

def diagonalizeHamiltonian(Hop,vectors,X):
    ''' Calculates matrix representation of Hop,
    forms truncated matrix (Hmat)
    and finally solves eigenvalue problem for Hmat

    In: Hop -> Operator (either as matrix or linearOperator)
        vectors -> list of basis
        X -> transformation matrix

    Out: Hmat -> Matrix represenation
                 (mainly for unit tests)
         ev -> eigenvalues
         uv -> eigenvectors'''

    typeClass = vectors[0].__class__
    qtAq = typeClass.matrixRepresentation(Hop,vectors)   
    Hmat = X.T.conj()@qtAq@X
    ev, uv = sp.linalg.eigh(Hmat)
    return Hmat,ev,uv

# ***************************************************
# Part 1: main FEAST function for contour integral
# ------------------------------
def feastDiagonalization(A,Y,nc,quad,rmin,rmax,eConv,maxit,status=None):
    """ FEAST diagonalization of A 

        In A     ::  matrix or linearoperator or SOP operator 
        In Y     ::  Initial guess of m0 vectors (m0 is called as subspace dimension)
        In nc    ::  number of quadrature points
        In quad  ::  quadrature points distribution
                     Avaiable options - "legendre", "hermite", "trapezoidal"
        In rmin  ::  eigenvalue lower limit
        In rmax  ::  eigenvalue upper limit
        In eConv ::  eigenvalue residual convergence tolerance
        In maxit ::  maximum feast iterations

        Out ev   ::  feast eigenvalues
        Out Y    ::  feast eigenvectors
    """

    typeClass = Y[0].__class__
    m0 = len(Y)
    r = (rmax-rmin)*0.5
    
    # initialize Q
    Q = [np.nan for it in range(m0)]

    # numerical quadrature points.
    gk,wk = quad_func(nc,quad)
    pi = np.pi
    status = _getStatus(status,Y,r,nc,maxit,eConv) # will be used as lanczos 
    
    for it in range(maxit):
        status["outerIter"] = it
        for k in range(nc):
            status["innerIter"] = k
            status["cumIter"] += 1
            #print("iteration",it,"quadratue",k)
            
            theta = -(pi*0.5)*(gk[k]-1)
            z = ((rmin+rmax)*0.5)+ (r*np.exp(1.0j*theta))
            
            for im0 in range(m0):
                b = Y[im0]
                Qsolved = typeClass.solve(A,b,z)  # alright: complex128
                Qe = calculateQuadrature(Qsolved,theta,r,wk[k]) # alright: float64
                Q = updateQ(Q,im0,Qe,k)
        
        # eigh in Lowdin orthogonal basis
        uS, idx = transformationMatrix(Q)
        ev, uv = diagonalizeHamiltonian(A,Q,uS)[1:3]
        ev, rangeIdx = print_a_range(ev,rmin,rmax)
        
        uSH = uS@uv
        Y = basisTransformation(Q,uSH,rangeIdx,selective=True)

        if it != 0: 
            res = eigenvalueResidual(ev,ref_ev[idx])
            print("iteration",it,"residual",res)
            if res < eConv:
                break
       
        m0 = len(Y);Q = Q[0:m0]
        ref_ev = ev

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
    ev_min = 160.0
    ev_max = 166.0
    nc    = 8          # number of contour points
    quad  = "legendre" # Choice of quadrature points # available options, legendre, Hermite (, trapezoidal !)
    m0    = 6         # subspace dimension
    eps   = 1e-6      # residual convergence tolerance
    maxit = 4         # maximum FEAST iterations
    options = {"linearSolver":"gcrotmk","linearIter":1000,"linear_tol":1e-02}
    optionsDict = {"linearSystemArgs":options}
    
    Y0    = np.random.random((n,m0)) # eigenvector initial guess
    for i in range(m0):
         Y0[:,i] = np.ones(n) * (i+1)
    Y1 = la.qr(Y0,mode="economic")[0]


    Y = []
    for i in range(m0):
        Y.append(NumpyVector(Y1[:,i], optionsDict))

    contour_ev = print_a_range(ev, ev_min, ev_max)[0]
    print("--- actual eigenvalues",contour_ev,"---\n")
    efeast,ufeast =  feastDiagonalization(linOp,Y,nc,quad,ev_min,ev_max,eps,maxit)
    print("\n---feast eigenvalues",efeast,"---")
