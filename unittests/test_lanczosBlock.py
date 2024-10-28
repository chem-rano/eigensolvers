import unittest
import sys
from inexact_Lanczos  import (lowdinOrthoMatrix,diagonalizeHamiltonian,
        basisTransformation,inexactLanczosDiagonalization)
import numpy as np
from scipy import linalg as la
from numpyVector import NumpyVector
from util_funcs import find_nearest
import time
from util_funcs import get_pick_function_close_to_sigma

class Test_BlockLanczos(unittest.TestCase):
    def setUp(self):
        n = 100
        nBlock = 3
        iBlock = 5
        ev = np.linspace(1,200,n)
        ev[iBlock:iBlock+nBlock] = ev[iBlock]
        self.evBlock = ev[iBlock:iBlock+nBlock]
        np.random.seed(1212)
        Q = la.qr(np.random.rand(n,n))[0]
        #Q = la.qr(np.random.rand(n,n)+1j*np.random.rand(n,n))[0]
        A = Q.T @ np.diag(ev) @ Q
        assert(la.ishermitian(A, atol=1e-08, rtol=1e-08))

        options = {"linearSolver":"gcrotmk","linearIter":1000,"linear_tol":1e-04}
        optionDict = {"linearSystemArgs":options}
        self.writeOut = False
        Ys = la.qr(np.random.rand(n,nBlock),mode="economic")[0]
        Y0 = [NumpyVector(Ys[:,iBlock],optionDict) for iBlock in range(nBlock)]

        self.nBlock = nBlock
        self.iBlock = iBlock
        self.guess = Y0
        self.mat = A
        self.ev = ev                     
        self.sigma = ev[iBlock] + nBlock/2
        self.eShift = 0.0
        self.L = 6
        self.maxit = 4
        self.eConv = 1e-6

        evEigh, uvEigh = np.linalg.eigh(A)
        self.evEigh = evEigh
        self.uvEigh = uvEigh
        self.pick = get_pick_function_close_to_sigma(self.sigma)

        
    def test_lanczos(self):
        nBlock = self.nBlock
        evLanczos, uvLanczos, status = inexactLanczosDiagonalization(self.mat,self.guess,self.sigma,self.L,
                self.maxit,self.eConv,pick=self.pick,writeOut=self.writeOut)
        with self.subTest("eigenvalue"):
            np.testing.assert_allclose(evLanczos[:nBlock], self.evBlock, rtol=self.eConv)
        with self.subTest("eigenvector"):
            iBlock = self.iBlock
            exactVector = self.uvEigh[:,iBlock:iBlock+nBlock]
            lanczosVector = np.vstack([ uvLanczos[i].array for i in range(nBlock)]).T
            # they are degenerate, so test trace of projector
            ovlp = lanczosVector.T.conj() @ exactVector # nonsymmetric
            trace = np.abs( la.eigvals(ovlp) ).sum()
            self.assertAlmostEqual(trace, 3)


if __name__ == '__main__':
    unittest.main()

