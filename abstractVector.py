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
    def copy(self):
        pass
    
    @abstractmethod
    def norm(self) -> float:  
        pass

    @abstractmethod
    def dot(self):
        pass
     
    @abstractmethod
    def applyOp(self):
        pass

    @staticmethod
    def linearCombination():
        pass
    
    
    @staticmethod
    def orthogonalize():
        pass
    
    @staticmethod
    def solve():
        pass
    
