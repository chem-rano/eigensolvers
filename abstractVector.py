from abc import ABC, abstractmethod
import numpy as np
from scipy import linalg as la

# file1: abstractVector.py holding these abstract functions list
# file2: numpyVector.py :: specifications of the tasks for each functions defined earlier for ndarray
# file2: listTTNS.py  :: same as file2 for TTNS
# main.py: utilizing these class for main purpose, such as inexact_Lanczos.py

# Abstract functions are here for initiation / listing
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
    def vdot(self,other,conjugate):
        pass
     
    @abstractmethod
    def copy(self):
        pass
    
    @abstractmethod
    def applyOp(self,other):
        pass

    @staticmethod
    def linearCombination(other,coeff):
        raise NotImplementedError
    
    @staticmethod
    def orthogonalize(xs,lindep):
        raise NotImplementedError
    
    @staticmethod
    def orthogonalize_against_set(x,xs,lindep):
        raise NotImplementedError
    
    @staticmethod
    def solve(H, b, sigma, x0):
        raise NotImplementedError
