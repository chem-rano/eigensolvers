import numpy as np
from scipy import linalg as la
from abstract import AbstractVector
import warnings

####################################################################
# Creates a NumpyVector class, which have defined elementary operations
# Obj.arguments will be used in the util functions
####################################################################


# file1: abs_funcs.py holding these abstract functions list
# file2: NumpyVector.py :: specifications of the tasks for each functions defined earlier for ndarray
# file2: listTTNS.py  :: same as file2 for TTNS
# main.py: utilizing these class for main purpose, such as inexact_Lanczos.py

# Assign the task for the functions initiated in abstract class
class NumpyVector(AbstractVector):
    def __init__(self,array):
        self.array = array
        self.dtype = array.dtype
        self.size = array.size
        self.shape = array.shape
    
    def __add__(self, other):
        return self.array + other

    def __sub__(self,other):
        return self.array - other
    
    def __mul__(self,other):
        if isinstance(other,int):
            return self.array*other

    def __truediv__(self,other):
        return self.array/other

    def __matmul__(self,other):
        return self.array @ other 
        #return np.matmul(self.arrayIn,other)

    def __len__(other):
        return len(np.array(other))
    
    def copy(self):
        copyArray = self.array
        return copyArray

    def linearCombination(self, other:list,coeff:list) -> np.array:
        '''
        Returns the linear combination of n vectors [v1, v2, ..., vn]
        combArray = self.arrayIn + c1*v1 + c2*v2 + cn*vn 
        Useful for addition, subtraction: c1 = 1.0/-1.0, respectively

        In:: other == list of vectors
             coeff == list of coefficients, [c1,c2,...,cn]
        '''
        alen = len(self.array)
        combArray = self.array
        for n in range(len(other)):
            for ii in range(alen):
                combArray[ii] += coeff[n]*other[n][ii]
        return combArray


    def norm(self):
        return la.norm(self.array)
    
    def dot(self,other,conjugate):
        if not isinstance(other,np.ndarray):
            other = np.array(other.array)
        if conjugate == False:
            return np.dot(self.array,other)
        elif conjugate == True:
            return np.vdot(self.array,other)
    
    # Classmethod receives the class as an implicit first argument
    # If we are only concentrate on arguments, then same works with staticmethod
    @staticmethod
    def orthogonal(xs,lindep=1e-14):
        '''
        Constructs a orthogonal vector space using Gram-Schmidt algorithm
        The current vector space is checked for linear-independency
        Defualt measure of linear-indepency check is 1e-14

        In:: xs == list of vectors
         lindep (Optional) == linear dependency tolerance
         
         dot need not to be specified; NumpyVector has dot associated 
        
        '''

        nvec = len(xs)
        dtype = xs[0].dtype
        qs = np.empty((nvec,xs[0].size), dtype=dtype)

        nv = 0
        for i in range(nvec):
            xi = NumpyVector(xs[i].copy())    #  <class '__main__.myVector'>
            #print(xi.arrayIn)    # [ 1. -1.  1.] 
            #print(len(xi.arrayIn)) # 3
            for j in range(nv):
                qsj = NumpyVector(qs[j])
                prod = qsj.dot(xi,conjugate=True)
                xi -= qsj.array*prod
                xi = NumpyVector(xi)
            innerprod = xi.dot(xi,True)   #.real
            norm = np.sqrt(innerprod)
            if innerprod > lindep:
                qs[nv] = xi/norm
                nv += 1
        return qs[:nv]

    @staticmethod
    def solve_wk(sigma,H,b,gcrot_tol,gcrot_iter):
    # Solve (EI-H)w = v with iterative solver
    # w ==> x0; v ==> b_in; linOp ==> (EI-H)
    # Ax =b CGS solver :: A == (n x n); x == (n x 1) and b ==> (n x 1)  
    

    n = H.shape[0]
    sigma = sigma*np.eye(n)
    linOp = LinearOperator((n,n),lambda x, sigma=sigma, H=H:(sigma@x - H@x))
    wk,conv = scipy.sparse.linalg.gcrotmk(linOp,b,x0=None, tol=gcrot_tol,atol=gcrot_tol,maxiter = gcrot_iter)

    if conv != 0:
        # adding a single entry into warnings filter
        warnings.simplefilter('error', UserWarning)
        warnings.warn("Warning:: Iterative solver is not converged ")
    return wk

# ---------------------------------------------------
if __name__ == "__main__":
    #np.random.seed(10)
    #Y0  = np.random.random((10))
    Y0  = np.array([1.0,2.0,3.0,4.0])
    copyY0 = np.array(Y0,copy =True)
    d = len(Y0)
    Y0 = NumpyVector(Y0)

    print("Multiplication with a number",Y0*2)
    print("Multiplication with an array",Y0@np.ones(d))
    
    # ---------- check linearCombination() -------------
    make_combine = []
    make_combine.append(np.ones(d))
    make_combine.append(np.ones(d))

    dlist = len(make_combine)
    coeff = [1.0 for i in range(dlist)]

    main_combine = copyY0 + coeff[0]*np.ones(d) + coeff[1]*np.ones(d) 
    func_combine = Y0.linearCombination(make_combine,coeff)
    print(str(main_combine) == str(func_combine))
    # --------------------------------------------------
    

    print("Norm", Y0.norm())
    print("dot product",Y0.dot(np.ones(d),conjugate=True))
    # --------------------------------------------------
    xs = []
    Y0  = NumpyVector(np.array([1.0,-1.0,1.0]))
    xs.append(Y0)
    Y0  = NumpyVector(np.array([1.0,0.0,1.0]))
    xs.append(Y0)
    Y0  = NumpyVector(np.array([1.0,1.0,2.0]))
    xs.append(Y0)
   
    #print(type(xs))
    Q = NumpyVector.orthogonal(xs)    
    print(Q)
