import numpy as np
import scipy
from scipy import linalg as la
from abstractVector import AbstractVector
from scipy.sparse.linalg import LinearOperator
import warnings

####################################################################
# Creates a numpyVector class, which has defined elementary operations
####################################################################

# file1:   abs_funcs.py holding these abstract functions list
# file2:   numpyVector.py :: specifications of the tasks for each functions 
#          defined earlier for ndarray
# main.py: utilizing these class for main purpose, such as inexact_Lanczos.py

# Assign the task for the functions initiated in abstract class
# -------------------------------------------------------------
class NumpyVector(AbstractVector):

    def __init__(self,array,optionsDict=dict()):
        self.array = array
        self.size = array.size
        self.shape = array.shape
        self.optionsDict = optionsDict
        self.optionsDict["linearSolver"] = self.optionsDict.get("linearSolver","minres")
        self.optionsDict["linearIter"] = self.optionsDict.get("linearIter",1000)
        self.optionsDict["linear_tol"] = self.optionsDict.get("linear_tol",1e-4)
        self.optionsDict["linear_atol"] = self.optionsDict.get("linear_atol",1e-4)
    
    @property
    def hasExactAddition(self):
        """
        Simplication of vector addition with its complex conjugate.
        For example, c+c* = 2c when c=(a+ib)
        This summation is true for numpy vectors
        But does not exactly same as 2c for TTNS
        """
        return True
        
    @property
    def dtype(self):
        return self.array.dtype
        
    def __mul__(self,other):
        return NumpyVector(self.array*other,self.optionsDict)
    
    def __rmul__(self,other):
        return NumpyVector(self.array*other,self.optionsDict)

    def __truediv__(self,other):
        return NumpyVector(self.array/other,self.optionsDict)
    
    def __imul__(self, other):
        raise NotImplementedError

    def __itruediv__(self, other):
        raise NotImplementedError


    def __len__(self) -> int:
        return len(self.array)
    
    def normalize(self):
        out = self.array/la.norm(self.array)
        return NumpyVector(out, self.optionsDict)

    def norm(self) -> float:
        return la.norm(self.array)

    def real(self):
        return NumpyVector(np.real(self.array),self.optionsDict)

    def vdot(self,other,conjugate:bool=True):
        if conjugate:
            return np.vdot(self.array,other.array)
        else:
            return np.dot(self.array.ravel(),other.array.ravel())
    
    def copy(self):
        return NumpyVector(self.array.copy(), self.optionsDict)

    def applyOp(self,other):
        ''' Apply rmatmul as other@self.array '''
        return NumpyVector(other@self.array,self.optionsDict)
    

    def linearCombination(other,coeff):
        '''
        Returns the linear combination of n vectors [v1, v2, ..., vn]
        combArray = c1*v1 + c2*v2 + cn*vn 
        Useful for addition, subtraction: c1 = 1.0/-1.0, respectively

        In:: other == list of vectors
             coeff == list of coefficients, [c1,c2,...,cn]
        '''
        dtype = other[0].dtype
        combArray = np.zeros(len(other[0]),dtype=dtype)
        for n in range(len(other)):
            combArray += coeff[n]*other[n].array
        return NumpyVector(combArray,other[0].optionsDict)

    

    def orthogonalize_against_set(x,qs,lindep):
        '''
        Orthogonalizes a vector against the previously obtained set of 
        orthogonalized vectors
        x (In): vector to be orthogonalized 
        xs (In): set of orthogonalized vector
        lindep (optional): Parameter to check linear dependency
                          Deafult value is LINDEP_DEFAULT_VALUE
                          See module abstractVector.py
        If it does not find linearly independent vector w.r.t. xs; it returns None
        '''
        nv = len(qs)
        for i in range(nv):
            qsi = qs[i]
            term1 = x.vdot(qsi,conjugate=False)
            term2 = qsi.vdot(qsi,conjugate=False)
            proj = qsi*(term1/term2)
            x = NumpyVector.linearCombination([x,proj],[1.0,-1.0])
            #x = x - proj
        innerprod = x.vdot(x,conjugate=False)
        if innerprod > lindep:
            x = x/np.sqrt(innerprod) # normalize
        else:
            x = None
        return x
        
    def solve(H, b, sigma = None, x0=None):
        ''' Linear equation ((H-sigma*I)x0 =b ) solver'''
            
        n = H.shape[0]
        dtype = np.result_type(sigma, H.dtype, b.dtype)
        linOp = LinearOperator((n,n),matvec = lambda x, sigma=sigma, H=H:(sigma*x-H@x),dtype=dtype)

        tol = b.optionsDict["linear_tol"]
        atol = b.optionsDict["linear_atol"]
        maxiter = b.optionsDict["linearIter"]
        if b.optionsDict["linearSolver"] == "gcrotmk":
            wk,conv = scipy.sparse.linalg.gcrotmk(linOp,b.array,x0, tol=tol,atol=atol,maxiter=maxiter)
        elif b.optionsDict["linearSolver"] == "minres":
            wk,conv = scipy.sparse.linalg.minres(linOp,b.array,x0, tol=tol,maxiter=maxiter)

        if conv != 0:
            warnings.simplefilter('error', UserWarning)
            warnings.warn("Warning:: Iterative solver is not converged ")
        return NumpyVector(wk,b.optionsDict)

    def matrixRepresentation(operator,vectors):
        ''' Calculates and returns matrix in the "vectors" space '''
        m = len(vectors)
        dtype = vectors[0].dtype
        qtAq = np.zeros((m,m),dtype=dtype)
        for j in range(m):
            ket = vectors[j].applyOp(operator)
            for i in range(j,m):
                qtAq[i,j] = vectors[i].vdot(ket)
                qtAq[j,i] = qtAq[i,j].conj()
        return qtAq

    def overlapMatrix(vectors):
        ''' Calculates overlap matrix of vectors'''

        m = len(vectors)
        dtype = vectors[0].dtype
        qtq = np.zeros((m,m),dtype=dtype)
        
        for i in range(m):
            for j in range(i,m):
                qtq[i,j] = vectors[i].vdot(vectors[j],True)
                qtq[j,i] = qtq[i,j].conj()
        return qtq
    # -----------------------------------------------------
    def extendMatrixRepresentation(operator,vectors,qtAq):
        ''' Calculates matrix elements of newly added tensor networks state
        and returns extended matrix representation'''
        m = len(vectors)
        dtype = vectors[0].dtype

        elems = np.empty(m,dtype=dtype)
        ket = vectors[-1].applyOp(operator)
        for i in range(m):
            elems[i] = vectors[i].vdot(ket)
        offD = np.array([elems[:-1]])
        qtAq = np.append(qtAq,offD.conj(),axis=0)
        qtAq = np.append(qtAq,np.array([elems]).T,axis=1)
        return qtAq

    def extendOverlapMatrix(vectors,oMat):
        ''' Calculates overlap elements of newly added tensor networks state
        and returns extended overlap matrix'''
        m = len(vectors)
        dtype = vectors[0].dtype

        elems = np.empty(m,dtype=dtype)
        for i in range(m):
            elems[i] = vectors[i].vdot(vectors[-1],True)
        offD = np.array([elems[:-1]])
        oMat = np.append(oMat,offD.conj(),axis=0)
        oMat = np.append(oMat,np.array([elems]).T,axis=1)
        return oMat
