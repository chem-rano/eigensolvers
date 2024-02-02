from abc import ABC, abstractmethod
import numpy as np
from scipy import linalg as la

# file1: abstractVector.py holding these abstract functions list
# file2: numpyVector.py :: specifications of the tasks for each functions defined earlier for ndarray
# main.py: utilizing these class for main purpose, such as inexact_Lanczos.py

# Abstract functions are here for initiation/listing
# We name them and pass for later use

# Specify abstractmethod whenever the task should be specified later
class AbstractVector(ABC):
    
    @abstractmethod
    def __mul__(self,other):
        pass
    
    @abstractmethod
    def __truediv__(self,other):
        pass

    @abstractmethod
    def __len__(self):
        pass
        
    @abstractmethod
    def norm(self) -> float:  
        pass

    @abstractmethod
    def vdot(self,other,conjugate=True):
        pass
     
    @abstractmethod
    def copy(self):
        pass
    
    @abstractmethod
    def list_zeros(other):
        pass
    
    @abstractmethod
    def applyOp(self,other):
        ''' Apply rmatmul as other@self.array '''
        pass

    @staticmethod
    def linearCombination(other,coeff):
        '''
        Returns the linear combination of n vectors [v1, v2, ..., vn]
        combArray = c1*v1 + c2*v2 + cn*vn 
        Useful for addition, subtraction: c1 = 1.0/-1.0, respectively

        In:: other == list of vectors
             coeff == list of coefficients, [c1,c2,...,cn]
        '''
        raise NotImplementedError

    
    @staticmethod
    def orthogonalize_against_set(x,xs,lindep=1e-12):
        '''
        Orthogonalizes a vector against the previously obtained set of 
        orthogonalized vectors
        x (In): vector to be orthogonalized 
        xs (In): set of orthogonalized vector
        lindep (optional): Parameter to check linear dependency
                           If condition is not met, returns None
        '''
        raise NotImplementedError
    
    @staticmethod
    def solve(H, b, sigma, x0):
        ''' Linear equation ((H-sigma*I)x0 =b ) solver'''
        raise NotImplementedError

    @staticmethod
    def matrixRepresentation(operator,vectors):
        ''' Calculates and returns matrix in the "vectors" space '''
        raise NotImplementedError

    @staticmethod
    def eig_in_LowdinBasis(operator,vectors,tol=1e-14):
        ''' Orthogonalizes vectors using Lowdin method
        tol: Tolerance to keep eigenvalues 
        '''
        raise NotImplementedError
    

    @staticmethod
    def resvecs(operator,vectors,eigenvalues):
        '''Calculates eigenvector maximum residual'''
        raise NotImplementedError
