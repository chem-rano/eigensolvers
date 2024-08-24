import unittest
import sys
from feast  import *
from magic import ipsh
import numpy as np
from scipy import linalg as la
from numpyVector import NumpyVector
from util_funcs import find_nearest
import time
from util_funcs import get_a_range

class Test_feast(unittest.TestCase):

    def setUp(self):
        n = 100
        ev = np.linspace(1,200,n)
        np.random.seed(10)
        Q = la.qr(np.random.rand(n,n))[0]
        A = Q.T @ np.diag(ev) @ Q
        linOp = LinearOperator((n,n), matvec = lambda x, A=A: A@x)

        # Specify FEAST parameters
        self.rmin = 160.0
        self.rmax = 166.0
        self.nc = 8            # number of contour points
        self.quad = "legendre" # Choice of quadrature points
        m0 = 6                 # subspace dimension
        self.eConv = 1e-6      # residual convergence tolerance
        self.maxit = 20        # maximum FEAST iterations



        options = {"linearSolver":"gcrotmk","linearIter":1000,"linear_tol":1e-02}
        optionDict = {"linearSystemArgs":options}
        
        Y0    = np.random.random((n,m0)) 
        for i in range(m0):
            Y0[:,i] = np.ones(n) * (i+1)
        Y1 = la.qr(Y0,mode="economic")[0]

        Y = []
        for i in range(m0):
            Y.append(NumpyVector(Y1[:,i], optionDict))

        self.guess = Y
        self.mat = A

        evEigh, uvEigh = np.linalg.eigh(A)
        self.evEigh = evEigh
        self.uvEigh = uvEigh

    def test_Hmat(self):
        ''' Bypassing linear combination works for Hamitonian matrix formation'''
        uvfeast = feastDiagonalization(self.mat,self.guess,self.nc,self.quad,self.rmin,self.rmax,
                self.eConv,self.maxit)[1]
        typeClass = uvfeast[0].__class__
        S = typeClass.overlapMatrix(uvfeast[:-1])
        qtAq = typeClass.matrixRepresentation(self.mat,uvfeast[:-1])
        uS = transformationMatrix(uvfeast)[0]
        typeClass = uvfeast[0].__class__
        Hmat1 = diagonalizeHamiltonian(self.mat,uvfeast,uS)[0]  
        qtAq = typeClass.matrixRepresentation(self.mat,uvfeast)
        Hmat2 = uS.T.conj()@qtAq@uS
        np.testing.assert_allclose(Hmat1,Hmat2,rtol=1e-5,atol=0)
   
    def test_backTransform(self):
        ''' Checks linear combination'''
        uvfeast = feastDiagonalization(self.mat,self.guess,self.nc,self.quad,self.rmin,self.rmax,
                self.eConv,self.maxit)[1]
        typeClass = uvfeast[0].__class__
        S = typeClass.overlapMatrix(uvfeast[:-1])
        assert len(uvfeast) > 1
        qtAq = typeClass.matrixRepresentation(self.mat,uvfeast[:-1])
        uS = transformationMatrix(uvfeast)[0]
        uv = diagonalizeHamiltonian(self.mat,uvfeast,uS)[2] 
        uSH = uS@uv
        bases = basisTransformation(uvfeast,uSH)
        for m in range(len(uvfeast)):
            ovlp = bases[m].vdot(uvfeast[m],True)
            np.testing.assert_allclose(abs(ovlp), 1, rtol=1e-5, err_msg 
                    = f"{ovlp=} but it should be +-1")
            np.testing.assert_allclose(uvfeast[m].array,ovlp*bases[m].array,atol=1e-5)

    def test_orthogonalization(self):
        ''' Returned basis in old form is orthogonal'''
        uvfeast = feastDiagonalization(self.mat,self.guess,self.nc,self.quad,self.rmin,self.rmax,
                self.eConv,self.maxit)[1]
        typeClass = uvfeast[0].__class__
        S = typeClass.overlapMatrix(uvfeast)
        np.testing.assert_allclose(S,np.eye(S.shape[0]),atol=1e-5) 

        
    def test_transformationMatrix(self):
        ''' XH@S@X = 1'''
        uvfeast = feastDiagonalization(self.mat,self.guess,self.nc,self.quad,self.rmin,self.rmax,
                self.eConv,self.maxit)[1]
        typeClass = uvfeast[0].__class__
        S = typeClass.overlapMatrix(uvfeast)
        assert len(uvfeast) > 1
        S1 = typeClass.overlapMatrix(uvfeast[:-1])
        qtAq = typeClass.matrixRepresentation(self.mat,uvfeast[:-1])
        uS = transformationMatrix(uvfeast)[0]
        uv = diagonalizeHamiltonian(self.mat,uvfeast,uS)[2] 
        uSH = uS@uv
        mat = uSH.T.conj()@S@uSH
        np.testing.assert_allclose(mat,np.eye(mat.shape[0]),atol=1e-5) 
    

    def test_returnType(self):
        ''' Checks if the returned eigenvalue and eigenvectors are of correct type'''
        evfeast, uvfeast = feastDiagonalization(self.mat,self.guess,self.nc,self.quad,self.rmin,self.rmax,
                self.eConv,self.maxit)
        self.assertIsInstance(evfeast, np.ndarray)
        self.assertIsInstance(uvfeast, list)
        self.assertIsInstance(uvfeast[0], NumpyVector)


    def test_eigenvalue(self):
        ''' Checks if the calculated eigenvalue is accurate to seventh decimal place'''
        evfeast = feastDiagonalization(self.mat,self.guess,self.nc,self.quad,self.rmin,self.rmax,
                self.eConv,self.maxit)[0]
        
        with self.subTest("All contour eigenvalues"):
            contour_ev = get_a_range(self.evEigh, self.rmin, self.rmax)[0]
            ncontour_ev = len(contour_ev)
            nfeast_ev = len(evfeast)
            # Think in case of orthogonal basis
            self.assertTrue((ncontour_ev <= nfeast_ev),'All eigenvalues within contour must be calculated')

        with self.subTest("eigenvalue accuracy"):
            contour_evs = get_a_range(self.evEigh, self.rmin, self.rmax)[0]
            feast_evs = get_a_range(evfeast, self.rmin, self.rmax)[0]
            for i in range(len(contour_evs)):
                target_value = contour_evs[i]
                closest_value = find_nearest(feast_evs,target_value)[1]
                self.assertTrue((abs(target_value-closest_value)<= 1e-4),'Not accurate up to 4-nd decimal place')
    
    def test_eigenvector(self):
        ''' Checks if the calculated eigenvalue is accurate to fourth decimal place'''
       
        evfeast, uvfeast = feastDiagonalization(self.mat,self.guess,10,self.quad,self.rmin,self.rmax,
                1e-12,20)
        
        contour_evs = get_a_range(self.evEigh, self.rmin, self.rmax)[0]
        for i in range(len(contour_evs)):
            idxE = find_nearest(self.evEigh,contour_evs[i])[0]
            idxT = find_nearest(evfeast,contour_evs[i])[0]
            exactVector = self.uvEigh[:,idxE]
            feastVector = uvfeast[idxT].array

            ovlp = np.vdot(exactVector,feastVector)
            np.testing.assert_allclose(abs(ovlp), 1, rtol=1e-4, err_msg = f"{ovlp=} but it should be +-1")
            feastVector = feastVector * ovlp
            np.testing.assert_allclose(exactVector,feastVector,rtol=1e-3,atol=1e-3)


if __name__ == '__main__':
    unittest.main()

