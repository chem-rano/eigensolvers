import numpy as np
import scipy as sp
from scipy.sparse.linalg import LinearOperator
from scipy import special
from scipy import linalg as la
from util_funcs import select_within_range, quadraturePointsWeights, eigenvalueResidual
from util_funcs import lowdinOrtho, basisTransformation
from numpyVector import NumpyVector
from abstractVector import AbstractVector
import time
import math
from magic import ipsh
from printUtils import FeastPrintUtils

def _getStatus(status,guess):
    ''' Dictionary for storing info of stage of the computation
    In: guess: Guess vector to access exact addition property
    
    flagAddition: Boolean, True if linear combination is accurate 
    isConverged: Status of eigenvalue residual convergence
    phase: Stage of phase calculations
    startTime: Starting time 
    runTime: run time in seconds

    Out: StatusUp: Initiated and updated dictionary
    '''

    statusUp ={"flagAddition":guess[0].hasExactAddition,
               "outerIter":0, "quadrature":0,
               "isConverged":False,
               "phase":1,
               "residual":None,
               "startTime":time.time(), "runTime":0.0}
    
    if status is not None:
        givenkeys = status.keys()
    
        for item in givenkeys:       # overwrite defaults
            if item in status:
                statusUp[item] = status[item]
    
    return statusUp

def calculateQuadrature(Amat,guess_b,z,radius,angle,weight,contourEllipseFactor):
    ''' Calculates k-th quadrature Qquad_k assuming `Amat` is Hermitian
    
    For Hermitian matrix:
    Qquad_k=-0.25*w_k*r{exp(i*angle_k)G(z)Y+exp(-i*theta)G\dagger(z)Y}
    
    For real symm matrix:
    Qquad_k=-0.50*w_k*Real{r*exp(i*angle_k)G(z)Y}
    
    G(z)Y == Qe => From liner solver: (z*I-A)Qe = Y
    
    In: Amat => Matrix A for the problem Ax = ex
                Either as ndarray, linear operator or SOP
        guess_b => Guess for b for solving Qe as in linear system Ax = b
        z => k-th coutour point 
        radius => radius of the contour
        angle => k-th countor angle
        weight => k-th quadrature distribution weight
        contourEllipseFactor => contour shape factor (see below)
    
    Out: Qquad_k => k-th quadrature vector
    N.T.: exp(+/-i*theta) is expanded as 
    contourEllipseFactor*cos(theta)+/-isin(theta)
    This is to change countour shape
    e.g., contourEllipseFactor = 1.0, circular contour
          contourEllipseFactor = 0.3, ellipse contour
    
    This contourEllipseFactor is implemented in Polizzi's code.
    It is necessary for testing fortran data.
    '''

    b = guess_b # copying of guess to unalter guess
    typeClass = b.__class__
    if abs(z.imag) < 1e-15:
        # Some contours are on the real axis only
        opType = "her"
        z = z.real
    else:
        # Assuming Amat is Hermitian
        # TODO `sym` seems to have some numerical stability problems in scipy.solve
        #       TTNS unit tests don't converge.
        #opType = "sym"
        opType = "gen"

    if b.hasExactAddition:
        Qe = typeClass.solve(Amat,b,z, opType=opType)  # complex128
        mult = -0.50*weight*radius*(contourEllipseFactor*math.cos(angle)+math.sin(angle)*1j)
        Qquad_k = typeClass.real(mult*Qe)
    else:
        # Polizzi (12)
        mult = -0.25*weight*radius
        part1 = typeClass.solve(Amat,b,z,opType=opType)
        part2 = typeClass.solve(Amat,b,z.conj(),opType=opType)
        c1 = mult*(contourEllipseFactor*math.cos(angle)+math.sin(angle)*1j)
        c2 = mult*(contourEllipseFactor*math.cos(angle)-math.sin(angle)*1j)
        #print("Fit: calculateQuadrature")
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
        #print("Fit: Update quadrature")
        Q[im0] = typeClass.linearCombination([Q[im0],Qquad_k],[1.0,1.0])
    return Q
       
def transformationMatrix(vectors,lindep=1e-14,printObj=None):
    ''' Calculates transformation matrix from 
    overlap matrix in Q basis
    In: vectors (list of basis)
        lindep (default value is 1e-14, lowdinOrtho())
        printObj (opional): print object 
    
    Out: uS: transformation matrix
    idx : (Boolean array) indices of the returned vectors
          True if the element is linealy independent'''
    # TODO merge with the one in Lanczos
    
    typeClass = vectors[0].__class__
    S = typeClass.overlapMatrix(vectors)
    if printObj is not None:printObj.writeFile("overlap",S)
    idx, _, uS = lowdinOrtho(S,lindep)
    return uS, idx

def diagonalizeHamiltonian(Hop,vectors,X,printObj=None):
    ''' Calculates matrix representation of Hop,
    forms truncated matrix (Hmat)
    and finally solves eigenvalue problem for Hmat

    In: Hop -> Operator (either as matrix or linearOperator)
        vectors -> list of basis
        X -> matrix to transform into an orthogonal basis
             (transforms vectors to an orthogonal basis)
        printObj (opional): print object 

    Out: Hmat -> Matrix represenation
                 (mainly for unit tests)
         ev -> eigenvalues
         uv -> eigenvectors in the basis defined through `X`'''
    # TODO merge with the one in Lanczos

    typeClass = vectors[0].__class__
    HvecRepr = typeClass.matrixRepresentation(Hop,vectors)
    Hmat = X.T.conj()@HvecRepr@X
    ev, uv = sp.linalg.eigh(Hmat)
    
    if printObj is not None:
        printObj.writeFile("hamiltonian",Hmat)
        printObj.writeFile("eigenvalues",ev)

    return Hmat,ev,uv

# ***************************************************
# Part 1: main FEAST function for contour integral
# ------------------------------
def feastDiagonalization(A, Y: list[AbstractVector],
                         nc, quad, eMin, eMax, eConv, maxit, contourEllipseFactor=1.0,
                         writeOut=True, eShift=0.0, convertUnit="au",
                         outFileName=None, summaryFileName=None):
    """ FEAST diagonalization of A

    See Polizzi, PRB, 79, 115112 (2009) 10.1103/PhysRevB.79.115112
    and Baiardi, Kelemen, Reiher, JCTC, 18, 1415 (2021) 10.1021/acs.jctc.1c00984

        In A       ::  matrix or linearoperator or SOP operator
                    Note: Must be Hermitian. Otherwise, `calculateQuadrature` needs to be adapted.
        In Y       ::  Initial guess of vectors.
        In nc      ::  number of quadrature points
        In quad    ::  quadrature points distribution
                       Avaiable options - "legendre", "hermite", "trapezoidal"
        In eMin    ::  eigenvalue lower limit
        In eMax    ::  eigenvalue upper limit
        In eConv   ::  eigenvalue residual convergence tolerance
                Residual is calculated through Sum |E - Eprev| / sum(abs(E)
                    where E (Eprev) is the eigenvalue vector of the current (previous) iteration
        In maxit   ::  maximum feast iterations
        In contourEllipseFactor (optional) ::  Countor shape factor
                See `calculateQuadrature`
        In writeOut (optional):: Instruction to writing output files 
        In eShift (optional)   :: shift value for printing. Assuming `A` is shifted by this value.
        In convertUnit (optional):: unit for printing
        In outFileName (optional): output file name
        In summaryFileName (optional): summary file name

        Out ev   ::  feast eigenvalues
        Out Y    ::  feast eigenvectors
    """
    typeClass = type(Y[0])
    N_SUBSPACE = len(Y)
    assert eMax > eMin
    eRadius = (eMax - eMin) * 0.5
    

    # numerical quadrature points.
    gk, wk = quadraturePointsWeights(nc, quad, positiveHalf=True)
    pi = np.pi
    
    status = _getStatus(None,Y)
    printObj = FeastPrintUtils(Y[0], nc, quad, eMin, eMax, eConv, maxit, writeOut, eShift,
                               convertUnit, status, outFileName, summaryFileName)
    printObj.fileHeader()
    
    for it in range(maxit):
        status["outerIter"] = it
        printObj.writeFile("iteration",status)
        # initialize Q
        Q = [np.nan for it in range(N_SUBSPACE)]
        for k in range(len(gk)):
            status["quadrature"] = k

            # Polizzi (13,14); Baiardi uses slightly different equation
            theta = -(pi*0.5)*(gk[k]-1) # Polizzi (13)
            # z =(eMin + eMax) * 0.5 + eRadius  * exp(2pi i theta)
            # here changed to allow for ellipse and not circle on imag axis
            z = (eMin + eMax) * 0.5 + eRadius * (math.cos(theta) + contourEllipseFactor * 1.0j * math.sin(theta) )
            
            for im0 in range(N_SUBSPACE):
                Qquad_k = calculateQuadrature(A,Y[im0],z,eRadius,theta,wk[k],contourEllipseFactor)
                Q = updateQ(Q,im0,Qquad_k,k)
        
        # eigh in Lowdin orthogonal basis
        uS, idx = transformationMatrix(Q,printObj=printObj)
        ev, uv = diagonalizeHamiltonian(A,Q,uS,printObj)[1:3]
        
        uSH = uS@uv
        del uv
        Y = basisTransformation(Q,uSH)
        del Q

        if it != 0:
            if len(ref_ev) > len(ev):
                # TODO add unit test for this case # not priority
                # Get elements in ref_ev that are closest to ev
                indices = np.argmin(np.abs(ref_ev[:, None] - ev[None, :]) , axis=0)
                ref_ev = ref_ev[indices]
            elif len(ref_ev) < len(ev):
                raise RuntimeError(f"{ref_ev=} but {ev=}. Enlarged space?")
            residual = eigenvalueResidual(ev, ref_ev, eMin, eMax)
            status["runTime"] = time.time() - status["startTime"]
            status["residual"] = residual
            printObj.writeFile("summary",ev,residual,status)
            
            if residual < eConv:
                break

        if N_SUBSPACE != len(Y): print(f"Alert! Got {N_SUBSPACE-len(Y)} \
                dependent vectors")

        N_SUBSPACE = len(Y)
        ref_ev = ev

    printObj.writeFile("results",ev)
    printObj.fileFooter()

    return ev,Y,status


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
    options = {"linearSystemArgs":options}
    
    Y0    = np.random.random((n,m0)) # eigenvector initial guess
    for i in range(m0):
         Y0[:,i] = np.ones(n) * (i+1)
    Y1 = la.qr(Y0,mode="economic")[0]


    Y = []
    for i in range(m0):
        Y.append(NumpyVector(Y1[:,i], options))

    contour_ev = select_within_range(ev, ev_min, ev_max)[0]
    print("--- actual eigenvalues",contour_ev,"---\n")
    efeast,ufeast =  feastDiagonalization(linOp,Y,nc,quad,ev_min,ev_max,eps,maxit)[0:2]
    print("\n---feast eigenvalues",efeast,"---")
