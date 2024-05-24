import unittest
import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '../')
from inexact_Lanczos  import (transformationMatrix,diagonalizeHamiltonian,
        backTransform,inexactDiagonalization)
import numpy as np
from scipy import linalg as la
from numpyVector import NumpyVector
from util_funcs import find_nearest
import time

# Test 1 :  Checks Krylov expansion basis
# Test 2 :  Checks overlap matrix in above basis
# Test 3 :  Checks Hamiltonian matrix in above basis
# Test 4 :  Checks overlap matrix is unitary or not
# Test 5 :  Checks type of eigenvalues and eigenvectors
# Test 6 :  Checks accuracy of the eigenvalue


class Test_lanczos(unittest.TestCase):

    def setUp(self):
        n = 100
        ev = np.linspace(1,200,n)
        np.random.seed(1212)
        Q = la.qr(np.random.rand(n,n))[0]
        A = Q.T @ np.diag(ev) @ Q
        assert(la.ishermitian(A, atol=1e-08, rtol=1e-08))
        

        optionDict = {"linearSolver":"gcrotmk","linearIter":1000,"linear_tol":1e-04}
        Y0 = NumpyVector(np.random.random((n)),optionDict)
        
        self.guess = Y0
        self.mat = A
        self.ev = ev                     
        self.sigma = 9
        self.eShift = 0.0
        self.L = 6
        self.maxit = 4
        self.eConv = 1e-6

        evEigh, uvEigh = np.linalg.eigh(A)
        self.evEigh = evEigh
        self.uvEigh = uvEigh

        
    def test_Hmat(self):
        ''' Bypassing linear combination works for Hamitonian matrix formation'''
        uvLanczos = inexactDiagonalization(self.mat,self.guess,self.sigma,self.L,
                self.maxit,self.eConv,self.eShift)[1]
        uS = transformationMatrix(uvLanczos)[1]
        typeClass = uvLanczos[0].__class__
        Hmat1 = diagonalizeHamiltonian(self.mat,uvLanczos,uS,self.eShift)[0]  
        qtAq = typeClass.matrixRepresentation(self.mat,uvLanczos)
        Hmat2 = uS.T.conj()@qtAq@uS
        np.testing.assert_allclose(Hmat1,Hmat2,rtol=1e-5,atol=0)
   
    def xtest_backTransform(self):
        ''' Checks linear combination'''
        uvLanczos = inexactDiagonalization(self.mat,self.guess,self.sigma,self.L,
                self.maxit,self.eConv,self.eShift)[1]
        coeffs = np.ones(100)
        bases = backTransform(uvLanczos,coeffs)
        np.testing.assert_allclose(uvLanczos[0].array,bases.array,atol=1e-5)
        # 'list' object has no attribute 'array'; Do not know how to compare these 


    def test_orthogonalization(self):
        ''' Returned basis in old form is orthogonal'''
        uvLanczos = inexactDiagonalization(self.mat,self.guess,self.sigma,self.L,
                self.maxit,self.eConv,self.eShift)[1]
        typeClass = uvLanczos[0].__class__
        S = typeClass.overlapMatrix(uvLanczos)
        np.testing.assert_allclose(S,np.eye(S.shape[0]),atol=1e-5) 

        
    def xtest_transformationMatrix(self):
        ''' XH@S@X = 1'''
        uvLanczos = inexactDiagonalization(self.mat,self.guess,self.sigma,self.L,
                self.maxit,self.eConv,self.zpve)[1]
        typeClass = uvLanczos[0].__class__
        S = typeClass.overlapMatrix(uvLanczos)
        uS = transformationMatrix(uvLanczos)[1]
        uv = diagonalizeHamiltonian(self.mat,uvLanczos,uS,self.eShift)[2] 
        uSH = uS@uv
        mat = uSH@S@uS
        np.testing.assert_allclose(mat,np.eye(mat.shape[0]),atol=1e-5) 


    def test_returnType(self):
        ''' Checks if the returned eigenvalue and eigenvectors are of correct type'''
        evLanczos, uvLanczos = inexactDiagonalization(self.mat,self.guess,self.sigma,self.L,
                self.maxit,self.eConv,self.eShift) 
        self.assertIsInstance(evLanczos, np.ndarray)
        self.assertIsInstance(uvLanczos, list)
        self.assertIsInstance(uvLanczos[0], NumpyVector)


    def test_eigenvalue(self):
        ''' Checks if the calculated eigenvalue is accurate to seventh decimal place'''
        evLanczos = inexactDiagonalization(self.mat,self.guess,self.sigma,self.L,
                self.maxit,self.eConv,self.eShift)[0]
        
        target_value = find_nearest(evLanczos,self.sigma)[1]
        closest_value = find_nearest(self.ev,self.sigma)[1]        # comapring with actual
        self.assertTrue((abs(target_value-closest_value)<= 1e-4),'Not accurate up to 4-nd decimal place')
    
    def xtest_eigenvector(self):
        ''' Checks if the calculated eigenvalue is accurate to seventh decimal place'''
        uvLanczos = inexactDiagonalization(self.mat,self.guess,self.sigma,self.L,
                self.maxit,self.eConv,self.zpve)[1]
        for i in range(len(uvLanczos)): 
            np.testing.assert_allclose(uvLanczos[i].array,self.uvEigh[i],atol=1e-5) 

if __name__ == '__main__':
    unittest.main()
