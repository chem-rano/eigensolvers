import numpy as np
import scipy
from scipy import linalg as la
from abstractVector import AbstractVector
from scipy.sparse.linalg import LinearOperator
import warnings

####################################################################
# Creates a numpyyVector class, which have defined elementary operations
####################################################################

# file1:   abs_funcs.py holding these abstract functions list
# file2:   numpyVector.py :: specifications of the tasks for each functions 
#          defined earlier for ndarray
# main.py: utilizing these class for main purpose, such as inexact_Lanczos.py

# Assign the task for the functions initiated in abstract class
# -------------------------------------------------------------
class NumpyVector(AbstractVector):
    def __init__(self,array,optionsDict):
        self.array = array
        self.dtype = array.dtype
        self.size = array.size
        self.shape = array.shape
        self.optionsDict = optionsDict

    def __sub__(self,other):
        return NumpyVector((self.array-other.array),self.optionsDict)
        
    def __mul__(self,other):
        return NumpyVector(self.array*other,self.optionsDic)

    def __truediv__(self,other):
        return NumpyVector(self.array/other,self.optionsDict)


    def __len__(self) -> int:
        return len(self.array)
    
    def norm(self) -> float:
        return la.norm(self.array)

    def vdot(self,other,conjugate:bool=True):
        if conjugate:
            return np.vdot(self.array,other.array)
        else:
            return np.dot(self.array.ravel(),other.array.ravel())
    
    def copy(self):
        return NumpyVector(self.array.copy(), self.optionsDict)

    def applyOp(self,other):
        return NumpyVector(other@self.array,self.optionsDict)
    

    def linearCombination(other,coeff):
        '''
        Returns the linear combination of n vectors [v1, v2, ..., vn]
        combArray = c1*v1 + c2*v2 + cn*vn 
        Useful for addition, subtraction: c1 = 1.0/-1.0, respectively

        In:: other == list of vectors
             coeff == list of coefficients, [c1,c2,...,cn]
        '''
        alen = len(other[0])
        dtype = other[0].dtype
        combArray = np.zeros(alen,dtype=dtype)
        for n in range(len(other)):
            combArray += coeff[n]*other[0].array[n]
        return NumpyVector(combArray,other[0].optionsDict)

    
    
    def orthogonalize(xs,lindep=1e-14):
        '''
        Constructs a orthogonal vector space using Gram-Schmidt algorithm
        The current vector space is checked for linear-independency
        Defualt measure of linear-indepency check is 1e-14

        In:: xs == list of vectors
         lindep (Optional) == linear dependency tolerance
         
         dot need not to be specified; myVector has vdot (and hence dot) associated 
        
        '''

        nvec = len(xs)
        dtype = xs[0].dtype
        qs_elem = NumpyVector(np.empty(xs[0].size,dtype=dtype))
        qs = []
        for i in range(nvec):
            qs.append(qs_elem)

        nv = 0
        for i in range(nvec):
            xi = xs[i]
            for j in range(nv):
                qsj = qs[j]
                prod = qsj.vdot(xi,conjugate=False)
                xi -= (qsj*prod)
            innerprod = xi.vdot(xi,conjugate=False)
            norm = np.sqrt(innerprod)
            if innerprod > lindep:
                qs[nv] = xi/norm
                nv += 1
        return qs[:nv]

    def orthogonalize_against_set(x,qs,lindep=1e-14):
        '''
        Orthogonalizes a single vector against the set, qs.
        '''
        nv = len(qs)
        for i in range(nv):
            qsi = qs[i]
            term1 = x.vdot(qsi,conjugate=False)
            term2 = qsi.vdot(qsi,conjugate=False)
            proj = (term1/term2)*qsi
            x = x - proj
        innerprod = x.vdot(x,conjugate=False)
        if innerprod > lindep:
            x = x/np.sqrt(innerprod) # normalize
        return x
        
    def solve(H, b, sigma, x0=None):

        n = H.shape[0]
        sigma = sigma*np.eye(n)
        tol = b.optionsDict["linearTol"]
        maxiter = b.optionsDict["linearIter"]
        linOp = LinearOperator((n,n),lambda x, sigma=sigma, H=H:(sigma@x - H@x))
        if b.optionsDict["linearSolver"] == "gcrotmk":
            wk,conv = scipy.sparse.linalg.gcrotmk(linOp,b.array,x0, tol=tol,atol=tol,maxiter=maxiter)
        elif b.optionsDict["linearSolver"] == "minres":
            wk,conv = cipy.sparse.linalg.minres(linOp,b.array,x0, tol=tol,atol=tol,maxiter=maxiter)

        if conv != 0:
            warnings.simplefilter('error', UserWarning)
            warnings.warn("Warning:: Iterative solver is not converged ")
        return NumpyVector(wk,b.optionsDict)

    def formMat(H,Ylist):
        m = len(Ylist)
        dtype = Ylist[0].dtype
        qtAq = np.zeros((m,m),dtype=dtype)
        for j in range(m):
            ket = Ylist[j].applyOp(H)
            for i in range(m):
                qtAq[i,j] = Ylist[i].vdot(ket)
                qtAq[j,i] = qtAq[i,j]
        return qtAq
    # -----------------------------------------------------
