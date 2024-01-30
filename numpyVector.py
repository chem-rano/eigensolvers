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
    def __init__(self,array):
        self.array = array
        self.dtype = array.dtype
        self.size = array.size
        self.shape = array.shape
    
    def __add__(self, other):
        return NumpyVector(self.array + other.array)

    def __sub__(self,other):
        return NumpyVector(self.array - other.array)

    def __mul__(self,other):
        return NumpyVector(self.array*other)

    def __truediv__(self,other):
        return NumpyVector(self.array/other)


    def __len__(self) -> int:
        return len(self.array)
    
    def norm(self) -> float:
        return la.norm(self.array)

    def vdot(self,other,conjugate:bool=True):
        if conjugate:
            return np.vdot(self.array,other.array)
        else:
            return np.dot(self.array.ravel(),other.array)
    
    def copy(self):
        return NumpyVector(self.array.copy())

    def applyOp(self,other):
        return NumpyVector(other@self.array)
    

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
        return NumpyVector(combArray)

    
    
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

    def solve(H, b, sigma, x0=None, shift=0.0, gcrot_tol=1e-5,gcrot_iter=1000):

        n = H.shape[0]
        sigma = sigma*np.eye(n)
        linOp = LinearOperator((n,n),lambda x, sigma=sigma, H=H:(sigma@x - H@x))
        wk,conv = scipy.sparse.linalg.gcrotmk(linOp,b.array,x0, tol=gcrot_tol,atol=gcrot_tol,maxiter = gcrot_iter)

        if conv != 0:
            warnings.simplefilter('error', UserWarning)
            warnings.warn("Warning:: Iterative solver is not converged ")
        return NumpyVector(wk)

    # -----------------------------------------------------
