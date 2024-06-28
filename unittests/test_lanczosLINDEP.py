import unittest
import sys
from inexact_Lanczos  import *
import numpy as np
from scipy import linalg as la
from numpyVector import NumpyVector
    
class Test_lanczos(unittest.TestCase):

    def setUp(self):
        # This is a specific case where linearly dependent vectors
        # generation happens
        n = 2500
        ev = np.linspace(1,1400,n)
        np.random.seed(10)
        Q = la.qr(np.random.rand(n,n))[0]
        #A = Q.T @ np.diag(ev) @ Q
        A = np.load("A.npy")

        optionDict = {"linearSolver":"gcrotmk","linearIter":5000,"linear_tol":2e-1}
        self.printChoices = {"writeOut": False,"writePlot": False}
        #Y0 = NumpyVector(np.random.random((n)),optionDict)
        Y0 = NumpyVector(np.load("Y0.npy"))
        
        self.guess = Y0
        self.mat = A
        self.ev = ev 
        self.sigma = 1290
        self.eShift = 0.0
        self.L = 100     # make sufficiently large to get LINDEP
        self.maxit = 1000 # same as above
        self.eConv = 1e-12

        evEigh, uvEigh = np.linalg.eigh(A)
        self.evEigh = evEigh
        self.uvEigh = uvEigh

    def test_status(self):
        ''' This specific case face lindep in the first Lanczos iteration, 
        check if status["lindep"] is indeed True or not'''

        status = inexactDiagonalization(self.mat,self.guess,self.sigma,
                self.L,self.maxit,self.eConv,self.printChoices)[2]
        self.assertTrue(status["lindep"]== True)

        
    def test_vectorsNumber(self):
        ''' Testing after getting linear dependency the list must be truncated
            or the length of vectors list should be iKrylov'''

        uvLanczos, status = inexactDiagonalization(self.mat,self.guess,self.sigma,
                self.L,self.maxit,self.eConv,self.printChoices)[1:3]
        iKrylov = status["innerIter"]
        nvectors = len(uvLanczos)
        self.assertTrue(nvectors == iKrylov)

    def test_futileRestarts(self):
        ''' For this specific case, number of futile restarts is larger than 3'''

        eConv = 1e-14 # stoping from early convergence
        status = inexactDiagonalization(self.mat,self.guess,self.sigma,
                self.L,self.maxit,eConv,self.printChoices)[2]
        nfutileRestarts = status["futileRestart"]
        if status["outerIter"] < status["maxit"]-1:
            self.assertTrue(nfutileRestarts > 3)

if __name__ == "__main__":
    unittest.main()
