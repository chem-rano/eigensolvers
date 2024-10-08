import numpy as np
import scipy as sp
from scipy.sparse.linalg import LinearOperator
from scipy import special
from scipy import linalg as la
from util_funcs import select_within_range, quad_func, eigenvalueResidual
from util_funcs import lowdinOrtho
from numpyVector import NumpyVector
import time
import math
from magic import ipsh

def _getStatus(status,guess,radius,maxit,nquad,eConv):
    ''' Dictionary for storing paramters, stage of the computation
    and status.
    guess: Guess vector to access exact addition property
    radius: Coutour radius for eigenvalue target rmin to rmax, (rmax-rmin)/2
    maxit: Maximum FEAST iterations
    nquad: Number of quadrature points
    eConv: Eigenvalue convergence
    
    Additional
    efactor: ellipse factor for countour (circle when efactor=1)
    flagAddition: Boolean, True if linear combination is accurate 
    isConverged: Status of eigenvalue residual convergence
    startTime: Starting time 
    runTime: run time in seconds
    writeOut: Instruction to write output file
    writePlot:Instruction to write plot data file
    eShift: shift for eigenvalue conversion (e.g, zpve)
    convertUnit: converting unit of eigenvalue, H elements
    '''
    
    statusUp = {"radius":radius,"maxit":maxit,
            "nquad":nquad,"eConv":eConv,
            "efactor":1.0,
            "flagAddition":guess[0].hasExactAddition,
            "outerIter":0, "innerIter":0,"cumIter":0,
            "isConverged":False,
            "startTime":time.time(), "runTime":0.0,
            "writeOut":True,"writePlot":True,"eShift":0.0,"convertUnit":"au"}
    
    if status is not None:
        givenkeys = status.keys()
    
        for item in givenkeys:       # overwrite defaults
            if item in status:
                statusUp[item] = status[item]
    
    return statusUp

def basisTransformation(bases,coeffs):
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
    
    if len(ndim) == 1:
        combBases.append(typeClass.linearCombination(bases,coeffs))
    else:
        for j in range(ndim[1]):
            combBases.append(typeClass.linearCombination(bases,coeffs[:,j]))
    
    return combBases

def calculateQuadrature(Amat,guess_b,z,radius,angle,weight,status):
    ''' Calculates k-th quadrature Qquad_k
    
    For Hermitian matrix:
    Qquad_k=-0.25*w_k*r{exp(i*angle_k)G(z)Y+exp(-i*theta)G\dagger(z)Y}
    
    For real symm matrix:
    Qquad_k=-0.50*w_k*Real{r*exp(i*angle_k)G(z)Y}
    
    G(z)Y == Qe => From liner solver: (z*I-A)Qe = Y
    For numpyVector => (z-A) is solved; prefactor = +1
    For ttnsVector => (A-z) is solved; prefactor = -1
    
    In: Amat => Matrix A for the problem Ax = ex
                Either as ndarray, linear operator or SOP
        guess_b => Guess for b for solving Qe as in linear system Ax = b
        z => k-th coutour point 
        radius => radius of the contour
        angle => k-th countor angle
        weight => k-th quadrature distribution weight
    
    Out: Qquad_k => k-th quadrature vector
    N.T.: for hasExactAddition: exp(i*theta) is expanded as 
    efactor*cos(theta)+isin(theta)
    This efactor is implemented in Polizzi's code.
    '''

    b = guess_b # copying of guess to unalter guess
    typeClass = b.__class__
    efactor = status["efactor"] 
    
    if b.hasExactAddition:
        prefactor = +1
        Qe = typeClass.solve(Amat,b,z)  # complex128
        mult = prefactor*(-0.50*weight*radius)*(efactor*math.cos(angle)+math.sin(angle)*1.00j)
        Qquad_k = typeClass.real(mult*Qe)
    else:
        prefactor = -1  
        mult = prefactor*(-0.25*weight*radius)
        part1 = typeClass.solve(Amat,b,z,opType="gen")
        part2 = typeClass.solve(Amat,b,z.conj(),opType="gen") #NOTE:assuming Amat is hermitian
        c1 = mult*np.exp(1j*angle)
        c2 = mult*np.exp(-1j*angle)
        Qquad_k = typeClass.linearCombination([part1,part2],[c1,c2])

    return Qquad_k

def updateQ(Q,im0,Qquad_k,k):
    ''' Adds k-th quadrature solution to the existing Q
    In: Q => Q vectors
        im0 => im0-th vector to be updated
        Qquad_k => k-th quadrature for the im0-th 
                vector to be updated
        k => quadrature point

    Out: Q => updated Q vectors'''

    typeClass = Qquad_k.__class__
    if k == 0:
        Q[im0] = Qquad_k
    else:
        Q[im0] = typeClass.linearCombination([Q[im0],Qquad_k],[1.0,1.0])
    return Q
       
def transformationMatrix(vectors,lindep=1e-14):
    ''' Calculates transformation matrix from 
    overlap matrix in Q basis
    In: vectors (list of basis)
        lindep (default value is 1e-14, lowdinOrtho())
    
    Out: uS: transformation matrix'''
    
    typeClass = vectors[0].__class__
    S = typeClass.overlapMatrix(vectors)
    uS, idx = lowdinOrtho(S,lindep)[1:3]
    return uS, idx

def diagonalizeHamiltonian(Hop,vectors,X):
    ''' Calculates matrix representation of Hop,
    forms truncated matrix (Hmat)
    and finally solves eigenvalue problem for Hmat

    In: Hop -> Operator (either as matrix or linearOperator)
        vectors -> list of basis
        X -> transformation matrix
             (transforms vectors to an orthogonal basis)

    Out: Hmat -> Matrix represenation
                 (mainly for unit tests)
         ev -> eigenvalues
         uv -> eigenvectors in the basis defined through `X`'''

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
    assert rmax > rmin
    r = (rmax-rmin)*0.5
    

    # numerical quadrature points.
    gk,wk = quad_func(nc,quad)
    pi = np.pi
    status = _getStatus(status,Y,r,maxit,nc,eConv) # will be used as lanczos 
    efactor = status["efactor"]
    
    for it in range(maxit):
        status["outerIter"] = it
        # initialize Q
        Q = [np.nan for it in range(m0)]
        for k in range(nc):
            status["innerIter"] = k
            status["cumIter"] += 1
            print("iteration",it,"quadrature",k)
            
            theta = -(pi*0.5)*(gk[k]-1)
            z = (rmin+rmax) * 0.5 + r*math.cos(theta)+r*efactor*1.0j*math.sin(theta)
            
            for im0 in range(m0):
                Qquad_k = calculateQuadrature(A,Y[im0],z,r,theta,wk[k],status) # float64
                Q = updateQ(Q,im0,Qquad_k,k)
        
        # eigh in Lowdin orthogonal basis
        uS, idx = transformationMatrix(Q)
        ev, uv = diagonalizeHamiltonian(A,Q,uS)[1:3]
        
        uSH = uS@uv
        Y = basisTransformation(Q,uSH)
        
        if it != 0: 
            res = abs(eigenvalueResidual(ev,ref_ev[idx],rmin,rmax))
            print("iteration",it,"residual",res)
            if res < eConv:
                break
       
        m0 = len(Y);del Q
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

    contour_ev = select_within_range(ev, ev_min, ev_max)[0]
    print("--- actual eigenvalues",contour_ev,"---\n")
    efeast,ufeast =  feastDiagonalization(linOp,Y,nc,quad,ev_min,ev_max,eps,maxit)
    print("\n---feast eigenvalues",efeast,"---")
