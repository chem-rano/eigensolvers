import unittest
import sys
from inexact_Lanczos  import *
import numpy as np
from scipy import linalg as la
from numpyVector import NumpyVector
import basis


class Test_stateFollowing(unittest.TestCase):

    def setUp(self):
        xRange = [-10,10]
        N=45
        sincInfInfOpts = basis.SincInfInf.getOptions(N=N, xRange=xRange)
        sincInfInf = basis.SincInfInf(sincInfInfOpts)

        T = -sincInfInf.mat_dx2
        V = np.diag(sincInfInf.xi**2)
        H  = T + V
        evEigh, uvEigh = la.eigh(H)
        self.evEigh = evEigh
        self.uvEigh = uvEigh

        self.sigma = 13.1
        options = {"linearSolver":"gcrotmk","linearIter":30000,"linear_tol":1e-04}
        optionDict = {"linearSystemArgs":options}
        self.printChoices = {"writeOut": False,"writePlot": False,"stateFollowing":"maxOvlp"}
        #self.printChoices["target"] = self.sigma
        idx = find_nearest(evEigh,self.sigma)[0]
        self.printChoices["ovlpRef"] = NumpyVector(uvEigh[:,idx],optionDict)
        np.random.seed(13)
        Y0 = NumpyVector(np.random.random((N)),optionDict)
        
        self.guess = Y0
        self.mat = H
        self.eShift = 0.0
        self.L = 16 
        self.maxit = 200  
        self.eConv = 1e-10

    def test_status(self):
        """ Checking the user option is correctly sent to the main code
        This is verified as analyzing the status dictionary"""

        status = inexactDiagonalization(self.mat,self.guess,self.sigma,self.L,
                self.maxit,self.eConv,self.printChoices)[2]
        self.assertTrue(status["stateFollowing"]== "maxOvlp")
        self.assertFalse(status["stateFollowing"]== "sigma")
        
    def test_eigenvalue(self):
        ''' Checks if the calculated eigenvalue is accurate to 10*eConv'''

        evLanczos = inexactDiagonalization(self.mat,self.guess,self.sigma,self.L,
                self.maxit,self.eConv,self.printChoices)[0]

        target = find_nearest(evLanczos,self.sigma)[1]
        closest = find_nearest(self.evEigh,self.sigma)[1]        # comapring with actual
        relError = abs(target-closest)/(max(abs(target), 1e-14))
        self.assertTrue((relError <= 10*self.eConv),'Not accurate up to 10*eConv')

    def test_overlap(self):
        """ Mainly looking at the overap if the eigenvector is inclined to the reference vector"""

        evlanczos, uvlanczos,status = inexactDiagonalization(self.mat,self.guess,self.sigma,self.L,
                self.maxit,self.eConv,self.printChoices)
        self.assertTrue(status["isConverged"]== True)
        refVector = status["ovlpRef"].array
        idx = find_nearest(evlanczos,self.sigma)[0]
        lanczosVector = uvlanczos[idx].array

        ovlp = np.vdot(refVector,lanczosVector)
        np.testing.assert_allclose(abs(ovlp), 1, rtol=1e-2, err_msg = f"{ovlp=} but it should be +-1")

if __name__ == "__main__":
    unittest.main()
