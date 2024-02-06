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
        self.dtype = array.dtype
        self.size = array.size
        self.shape = array.shape
        self.optionsDict = optionsDict
        self.optionsDict["linearSolver"] = self.optionsDict.get("linearSolver","minres")
        self.optionsDict["linearIter"] = self.optionsDict.get("linearIter",1000)
        self.optionsDict["linear_tol"] = self.optionsDict.get("linear_tol",1e-4)
        self.optionsDict["linear_atol"] = self.optionsDict.get("linear_atol",1e-4)
    
    def __abs__(self):        
        return NumpyVector(abs(self.array),self.optionsDict)
        
    def __mul__(self,other):
        return NumpyVector(self.array*other,self.optionsDict)

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
    
    def list_zeros(other,dtype):
        ''' returns list of vectors (with all elements zero) for list of vectors'''
        nlist = len(other)
        out = []
        for i in range(nlist):
            item = NumpyVector(np.zeros_like(other[i].array,dtype=dtype),other[i].optionsDict)
            out.append(item)
        return out

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

    

    def orthogonalize_against_set(x,qs,lindep=1e-14):
        '''
        Orthogonalizes a vector against the previously obtained set of 
        orthogonalized vectors
        x (In): vector to be orthogonalized 
        xs (In): set of orthogonalized vector
        lindep (optional): Parameter to check linear dependency
                           If condition is not met, returns None
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
        return x
        
    def solve(H, b, x0=None):
        ''' Linear equation (Hx0 =b) solver'''
        
        n = H.shape[0]

        tol = b.optionsDict["linear_tol"]
        atol = b.optionsDict["linear_atol"]
        maxiter = b.optionsDict["linearIter"]
        if b.optionsDict["linearSolver"] == "gcrotmk":
            wk,conv = scipy.sparse.linalg.gcrotmk(H,b.array,x0, tol=tol,atol=atol,maxiter=maxiter)
        elif b.optionsDict["linearSolver"] == "minres":
            wk,conv = scipy.sparse.linalg.minres(H,b.array,x0, tol=tol,maxiter=maxiter)
        
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
            for i in range(m):
                qtAq[i,j] = vectors[i].vdot(ket)
                qtAq[j,i] = qtAq[i,j]
        return qtAq
        


    def eig_in_LowdinBasis(operator,vectors,tol=1e-14):
        ''' Orthogonalizes vectors using Lowdin method
        tol: Tolerance to keep eigenvalues 
        '''
        
        typeClass = vectors[0].__class__
        m = len(vectors)
        dtype = vectors[0].dtype
        qtq = np.zeros((m,m),dtype=dtype)
        
        for i in range(m):
            for j in range(i,m):
                qtq[i,j] = vectors[i].vdot(vectors[j],False)
                qtq[j,i] = qtq[i,j]

        evq, uvq = la.eigh(qtq)
        idx = evq > tol
        evq = evq[idx]
        uvq = uvq[:,idx]
        uvqTraf = uvq * evq**(-0.5)
        
        Q_trun = []
        m = len(evq)
        for i in range(m):
            Q_trun.append(typeClass.linearCombination(vectors,uvqTraf[:,i]))

        AqTraf = typeClass.matrixRepresentation(operator,Q_trun)
        ev, uvTraf = la.eigh(AqTraf)
        
        uv = uvqTraf @ uvTraf  # uqTraf and uvTraf are ndarray
        return ev, uv, Q_trun

    
    # -----------------------------------------------------
    def resvecs(operator,vectors,eigenvalues):
        '''Calculates eigenvector maximum residual'''
        #Residual = ((np.eye(n))@x@np.diag(lest))-A@x
        n = vectors[0].shape
        m0 = len(vectors)
        typeClass = vectors[0].__class__
        #I = np.eye(n)
        diagM = np.diag(eigenvalues)

        # no need to multiply with identity matrix
        residual = []
        for i in range(m0):
            #lElem = vectors[i].applyOp(np.diag(eigenvalues))   # TODO is np.diag needed?
            lElem = vectors[i]*eigenvalues[i]   
            rElem = vectors[i].applyOp(operator)
            residual.append(typeClass.linearCombination([lElem,rElem],[1.0,-1.0]))
        return residual
    
    # -----------------------------------------------------
    def resEigenvector(lest,x,resvecs,eps):
        '''
        Calculates maximum residual inside the contour subspace
        '''
        m0 = len(x)
        n = x[0].shape
        resnorms = np.zeros(m0)
    
        # step1: creates an array with residual norms (divided by x norms) 
        for i in range(m0):
            if (x[i].norm() > 1e-14):
                resnorms[i]= resvecs[i].norm()/x[i].norm()

        # step2: checks if how many resnorms follow convergence
        s = []
        nsubspace = 0                                     # n subspace states satisfying convergence criteria
        for k in range(m0):
            if(resnorms[k] < eps):                        # residual norm is less than a specified small number; 
                nsubspace = nsubspace + 1                 # when feast subspace is identical to exact space; it is zero.
                s.append(k)
    
        # step3: if all resnorms achieved convergence, then return maximum value from them
        #         otherwise go through all residual norms and print out the maximum from them
        maxres = 0.0
        if(nsubspace == 0):
            maxres = np.max(resnorms)
        else:
            for k in range(m0):
                if(resnorms[k] < eps): 
                    tmp = resnorms[k]
                    if(tmp > maxres):                    # finding the maximum residual value
                        maxres = tmp


        return maxres
    # -----------------------------------------------------
