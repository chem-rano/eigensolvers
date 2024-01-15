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
class myNdarray(Abstract_vector):
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


    #@classmethod
    def linearCombination(self, array:np.ndarray,coeff:float) -> np.array:
        alen = len(array)
        comb = self.arrayIn
        for ii in range(alen):
            comb += coeff[ii]*array[ii]
        return comb


    #@implements()
    def norm(self):
        return la.norm(self.arrayIn)
    
    def dot(self,other):
        return np.dot(self.arrayIn,other)

# ---------------------------------------------------
if __name__ == "__main__":
    np.random.seed(10)
    Y0  = np.random.random((10))
    Y0 = myNdarray(Y0)

    print("Multiplication with a number",Y0*2)
    print("Multiplication with an array",Y0@np.ones(10))
    print(Y0.linearCombination(np.ones(10),np.ones(10)))
    print("Norm", Y0.norm())
    print("dot product",Y0.dot(np.ones(10)))
