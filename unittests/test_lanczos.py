import unittest
import sys
from inexact_Lanczos  import (transformationMatrix,diagonalizeHamiltonian,
        basisTransformation,inexactDiagonalization)
import numpy as np
from scipy import linalg as la
from numpyVector import NumpyVector
from util_funcs import find_nearest
import time

class Test_lanczos(unittest.TestCase):

    def setUp(self):
        n = 100
        ev = np.linspace(1,200,n)
        np.random.seed(1212)
        Q = la.qr(np.random.rand(n,n))[0]
        #Q = la.qr(np.random.rand(n,n)+1j*np.random.rand(n,n))[0]
        A = Q.T @ np.diag(ev) @ Q
        assert(la.ishermitian(A, atol=1e-08, rtol=1e-08))
        

        optionDict = {"linearSolver":"gcrotmk","linearIter":1000,"linear_tol":1e-04}
        self.printChoices = {"writeOut": False,"writePlot": False}
        Y0 = NumpyVector(np.random.random((n)),optionDict)
        
        self.guess = Y0
        self.mat = A
        self.ev = ev                     
        self.sigma = 30
        self.eShift = 0.0
        self.L = 6
        self.maxit = 4
        self.eConv = 1e-6

        evEigh, uvEigh = np.linalg.eigh(A)
        self.evEigh = evEigh
        self.uvEigh = uvEigh

        
    def test_Hmat(self):
        ''' Bypassing linear combination works for Hamitonian matrix formation'''
        uvLanczos, status = inexactDiagonalization(self.mat,self.guess,self.sigma,self.L,
                self.maxit,self.eConv,self.printChoices)[1:3]
        typeClass = uvLanczos[0].__class__
        S = typeClass.overlapMatrix(uvLanczos[:-1])
        qtAq = typeClass.matrixRepresentation(self.mat,uvLanczos[:-1])
        uS = transformationMatrix(uvLanczos,S,status)[1]
        typeClass = uvLanczos[0].__class__
        Hmat1 = diagonalizeHamiltonian(self.mat,uvLanczos,uS,qtAq,status)[0]  
        qtAq = typeClass.matrixRepresentation(self.mat,uvLanczos)
        Hmat2 = uS.T.conj()@qtAq@uS
        np.testing.assert_allclose(Hmat1,Hmat2,rtol=1e-5,atol=0)
   
    def test_backTransform(self):
        ''' Checks linear combination'''
        uvLanczos, status = inexactDiagonalization(self.mat,self.guess,self.sigma,self.L,
                self.maxit,self.eConv,self.printChoices)[1:3]
        typeClass = uvLanczos[0].__class__
        S = typeClass.overlapMatrix(uvLanczos[:-1])
        assert len(uvLanczos) > 1
        qtAq = typeClass.matrixRepresentation(self.mat,uvLanczos[:-1])
        uS = transformationMatrix(uvLanczos,S,status)[1]
        uv = diagonalizeHamiltonian(self.mat,uvLanczos,uS,qtAq,status)[2] 
        uSH = uS@uv
        bases = basisTransformation(uvLanczos,uSH)
        np.testing.assert_allclose(uvLanczos[0].array,bases[0].array,atol=1e-5)

    def test_orthogonalization(self):
        ''' Returned basis in old form is orthogonal'''
        uvLanczos = inexactDiagonalization(self.mat,self.guess,self.sigma,self.L,
                self.maxit,self.eConv,self.printChoices)[1]
        typeClass = uvLanczos[0].__class__
        S = typeClass.overlapMatrix(uvLanczos)
        np.testing.assert_allclose(S,np.eye(S.shape[0]),atol=1e-5) 

        
    def test_transformationMatrix(self):
        ''' XH@S@X = 1'''
        uvLanczos,status = inexactDiagonalization(self.mat,self.guess,self.sigma,self.L,
                self.maxit,self.eConv,self.printChoices)[1:3]
        typeClass = uvLanczos[0].__class__
        S = typeClass.overlapMatrix(uvLanczos)
        assert len(uvLanczos) > 1
        S1 = typeClass.overlapMatrix(uvLanczos[:-1])
        qtAq = typeClass.matrixRepresentation(self.mat,uvLanczos[:-1])
        uS = transformationMatrix(uvLanczos,S1,status)[1]
        uv = diagonalizeHamiltonian(self.mat,uvLanczos,uS,qtAq,status)[2] 
        uSH = uS@uv
        mat = uSH.T.conj()@S@uSH
        np.testing.assert_allclose(mat,np.eye(mat.shape[0]),atol=1e-5) 
    
    def test_extension(self):
        ''' Checks if extension of matrix works or not'''
        uvLanczos = inexactDiagonalization(self.mat,self.guess,self.sigma,self.L,
                self.maxit,self.eConv,self.printChoices)[1]
        typeClass = uvLanczos[0].__class__
        assert len(uvLanczos) > 1
        Sfull = typeClass.overlapMatrix(uvLanczos)
        S1 = typeClass.overlapMatrix(uvLanczos[:-1])
        S = typeClass.extendOverlapMatrix(uvLanczos,S1)
        qtAqfull = typeClass.matrixRepresentation(self.mat,uvLanczos)
        qtAq1 = typeClass.matrixRepresentation(self.mat,uvLanczos[:-1])
        qtAq = typeClass.extendMatrixRepresentation(self.mat,uvLanczos,qtAq1)
        np.testing.assert_allclose(S,Sfull,atol=1e-9) 
        np.testing.assert_allclose(qtAq,qtAqfull,atol=1e-9)
        

    def test_returnType(self):
        ''' Checks if the returned eigenvalue and eigenvectors are of correct type'''
        evLanczos, uvLanczos = inexactDiagonalization(self.mat,self.guess,self.sigma,self.L,
                self.maxit,self.eConv,self.printChoices)[0:2] 
        self.assertIsInstance(evLanczos, np.ndarray)
        self.assertIsInstance(uvLanczos, list)
        self.assertIsInstance(uvLanczos[0], NumpyVector)


    def test_eigenvalue(self):
        ''' Checks if the calculated eigenvalue is accurate to seventh decimal place'''
        evLanczos = inexactDiagonalization(self.mat,self.guess,self.sigma,self.L,
                self.maxit,self.eConv,self.printChoices)[0]
        
        target_value = find_nearest(evLanczos,self.sigma)[1]
        closest_value = find_nearest(self.ev,self.sigma)[1]        # comapring with actual
        self.assertTrue((abs(target_value-closest_value)<= 1e-4),'Not accurate up to 4-nd decimal place')
    
    def test_eigenvector(self):
        ''' Checks if the calculated eigenvalue is accurate to fourth decimal place'''
       
        evLanczos, uvLanczos = inexactDiagonalization(self.mat,self.guess,self.sigma,self.L,
                self.maxit,self.eConv,self.printChoices)[0:2]
        idxE = find_nearest(self.evEigh,self.sigma)[0]
        idxT = find_nearest(evLanczos,self.sigma)[0]
        exactVector = self.uvEigh[:,idxE]
        lanczosVector = uvLanczos[idxT].array

        ovlp = np.vdot(exactVector,lanczosVector)
        np.testing.assert_allclose(abs(ovlp), 1, rtol=1e-5, err_msg = f"{ovlp=} but it should be +-1")
        lanczosVector = lanczosVector* ovlp
        np.testing.assert_allclose(exactVector,lanczosVector,rtol=1e-5,atol=1e-4) 


if __name__ == '__main__':
    unittest.main()

