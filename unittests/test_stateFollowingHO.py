import unittest
import sys
from inexact_Lanczos  import *
import numpy as np
from scipy import linalg as la
from numpyVector import NumpyVector
import basis
from util_funcs import get_pick_function_close_to_sigma
from util_funcs import get_pick_function_maxOvlp

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
        self.printChoices = {"writeOut": True,"writePlot": True}
        idx = find_nearest(evEigh,self.sigma)[0]
        ovlpRef = NumpyVector(uvEigh[:,idx+1],optionDict)
        np.random.seed(13)
        Y0 = NumpyVector(np.random.random((N)),optionDict)
        self.pick = get_pick_function_maxOvlp(ovlpRef)
        #self.pick = get_pick_function_close_to_sigma(self.sigma)
        
        self.guess = Y0
        self.mat = H
        self.eShift = 0.0
        self.L = 16 
        self.maxit = 200  
        self.eConv = 1e-10
        self.ovlpRef = ovlpRef

    def test_eigenvalue(self):
        ''' Checks if the calculated eigenvalue is accurate to 10*eConv'''

        evLanczos, uvlanczos,status = inexactDiagonalization(self.mat,self.guess,self.sigma,self.L,
                self.maxit,self.eConv,self.pick,self.printChoices)
        self.assertTrue(status["isConverged"]== True)
        
        typeClass = uvlanczos[0].__class__
        reference = typeClass.matrixRepresentation(self.mat,[self.ovlpRef])[0]
        closest = find_nearest(self.evEigh,reference)[1]        # comapring with actual
        relError = abs(reference-closest)/(max(abs(reference), 1e-14))
        self.assertTrue((relError <= 1e-4),'Not accurate up to 1e-4')

    def test_overlap(self):
        """ Mainly looking at the overap if the eigenvector is inclined to the reference vector"""

        evlanczos, uvlanczos,status = inexactDiagonalization(self.mat,self.guess,self.sigma,self.L,
                self.maxit,self.eConv,self.pick,self.printChoices)
        self.assertTrue(status["isConverged"]== True)
        refVector = self.ovlpRef.array
        idx = find_nearest(evlanczos,self.sigma)[0]
        lanczosVector = uvlanczos[idx].array

        ovlp = np.vdot(refVector,lanczosVector)
        np.testing.assert_allclose(abs(ovlp), 1, rtol=1e-2, err_msg = f"{ovlp=} but it should be +-1")

if __name__ == "__main__":
    unittest.main()
