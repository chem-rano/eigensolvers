from abc import ABC, abstractmethod
import numpy as np
from scipy import linalg as la
from abstract import Abstract_vector
from implements import implements

# file1: abs_funcs.py holding these abstract functions list
# file2: myNdarray.py :: specifications of the tasks for each functions defined earlier for ndarray
# file2: listTTNS.py  :: same as file2 for TTNS
# main.py: utilizing these class for main purpose, such as inexact_Lanczos.py

# Assign the task for the functions initiated in abstract class
class myVector(Abstract_vector):
    def __init__(self,arrayIn):
        self.arrayIn = arrayIn
        self.dtype = arrayIn[0].dtype
        self.size = arrayIn.size
        self.shape = arrayIn.shape
    
    def __add__(self, other):
        return self.arrayIn + other

    def __sub__(self,other):
        return self.arrayIn - other
    
    def __mul__(self,other):
        if isinstance(other,int):
            return self.arrayIn*other

    def __truediv__(self,other):
        return self.arrayIn/other

    def __matmul__(self,other):
        return self.arrayIn @ other 
        #return np.matmul(self.arrayIn,other)


    def linearCombination(self, other:list,coeff:list) -> np.array:
        '''
        Returns the linear combination of n vectors [v1, v2, ..., vn]
        combArray = self.arrayIn + c1*v1 + c2*v2 + cn*vn 
        Useful for addition, subtraction: c1 = 1.0/-1.0, respectively

        In:: other == list of vectors
             coeff == list of coefficients, [c1,c2,...,cn]
        '''
        alen = len(self.arrayIn)
        combArray = self.arrayIn
        for n in range(len(other)):
            for ii in range(alen):
                combArray[ii] += coeff[n]*other[n][ii]
        return combArray


    def norm(self):
        return la.norm(self.arrayIn)
    
    def dot(self,other):
        return np.dot(self.arrayIn,other)

# ---------------------------------------------------
if __name__ == "__main__":
    #np.random.seed(10)
    #Y0  = np.random.random((10))
    Y0  = np.array([1.0,2.0,3.0,4.0])
    copyY0 = np.array(Y0,copy =True)
    d = len(Y0)
    Y0 = myVector(Y0)

    #print("Multiplication with a number",Y0*2)
    #print("Multiplication with an array",Y0@np.ones(10))
    
    # ---------- check linearCombination() -------------
    make_combine = []
    make_combine.append(np.ones(d))
    make_combine.append(np.ones(d))

    dlist = len(make_combine)
    coeff = [1.0 for i in range(dlist)]

    main_combine = copyY0 + np.ones(d) + np.ones(d) 
    func_combine = Y0.linearCombination(make_combine,coeff)
    print(str(main_combine) == str(func_combine))
    # --------------------------------------------------

    #print("Norm", Y0.norm())
    #print("dot product",Y0.dot(np.ones(10)))
