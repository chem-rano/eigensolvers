import unittest
import sys
from inexact_Lanczos  import (transformationMatrix,diagonalizeHamiltonian,
        basisTransformation,inexactDiagonalization)
import numpy as np
from scipy import linalg as la
from ttnsVector import TTNSVector
from util_funcs import find_nearest
import time
import basis
from ttns2.parseInput import parseTree
from ttns2.contraction import TruncationEps
from util import npu
from ttns2.diagonalization import IterativeLinearSystemOptions
import operatornD, operator1D
from util_funcs import calculateTarget

class Test_lanczos(unittest.TestCase):
    def setUp(self):
        dtype = float  # another option to try: complex
        basisDict = {
            "q0": basis.SincAB(basis.SincAB.getOptions(N=3)),
            "q1": basis.SincAB(basis.SincAB.getOptions(N=2)),
            "q2": basis.SincAB(basis.SincAB.getOptions(N=3)),
            "q3": basis.SincAB(basis.SincAB.getOptions(N=3)),
            "q4": basis.SincAB(basis.SincAB.getOptions(N=3)),
            "z5": basis.SincAB(basis.SincAB.getOptions(N=5)),
        }
        treeString = """
        0> 3 3
            1> 4 [q0]
                2> [q1 q2]
            1> 2 3 4
                2> [q3]
                2> [q4]
                2> [z5]
                """
        ttns = parseTree(treeString, basisDict, returnType="TTNS",dtype=dtype)
        np.random.seed(1212)
        ttns.setRandom()

        bases = list(basisDict.values())
        labels = list(basisDict.keys())
        nBas = [b.N for b in bases]
        Hop = operatornD.operatorSumOfProduct(nDim=len(nBas), nSum=3, DoFlabel=labels)
        for iSum in range(Hop.nSum-1): # have one unit term
            for iDim in range(Hop.nDim):
                Hop[iDim, iSum] = operator1D.general(str=f"{iDim}_{iSum}")
                if dtype == complex:
                    Hop[iDim, iSum].mat = npu.randomComplexHermitian(nBas[iDim])
                else:
                    Hop[iDim, iSum].mat = npu.randomSymmetric(nBas[iDim])
        Hop.obtainMultiplyOp(nBases=nBas)
        H = Hop.toFULLMatrix(nBas)
        assert npu.isHermitian(H)
        
        self.mat = Hop
        evEigh, uvEigh = la.eigh(H)
        self.evEigh = evEigh
        self.uvEigh = uvEigh
        
        MAX_D = 100 
        EPS = 5e-9
        bondDimensionAdaptions = [TruncationEps(EPS, maxD=MAX_D, offset=2, truncateViaDiscardedSum=False)]
        nsweepOrtho = 800
        orthoTol = 1e-08
        optShift = 0.0

        siteLinearTol = 1e-3
        globalLinearTol = 1e-2
        nsweepLinear = 1000

        fittingTol = 1e-9
        nsweepFitting = 1000

        optsCheck = IterativeLinearSystemOptions(solver="gcrotmk",tol=siteLinearTol) 
        optionsOrtho = {"nSweep":nsweepOrtho, "convTol":orthoTol, "optShift":optShift, "bondDimensionAdaptions":bondDimensionAdaptions}
        optionsLinear = {"nSweep":nsweepLinear, "iterativeLinearSystemOptions":optsCheck,"convTol":globalLinearTol,"bondDimensionAdaptions":bondDimensionAdaptions}
        optionsFitting = {"nSweep":nsweepFitting, "convTol":fittingTol,"bondDimensionAdaptions":bondDimensionAdaptions}
        options = {"orthogonalizationArgs":optionsOrtho, "linearSystemArgs":optionsLinear, "stateFittingArgs":optionsFitting}

        tns = TTNSVector(ttns,options)
        self.guess = tns
        self.zpve = 0.0
        self.maxit = 6
        self.L = 10 
        self.eConv = 1e-8
    
    def test_Hmat(self):
        ''' Bypassing linear combination works for Hamitonian matrix formation'''
        
        target = calculateTarget(self.evEigh,4)
        sigma = target + self.zpve
        uvLanczos, status = inexactDiagonalization(self.mat,self.guess,sigma,self.L,
                self.maxit,self.eConv)[1:3]
        assert status["isConverged"]== True
        typeClass = uvLanczos[0].__class__
        S = typeClass.overlapMatrix(uvLanczos[:-1])
        qtAq = typeClass.matrixRepresentation(self.mat,uvLanczos[:-1])
        uS = transformationMatrix(uvLanczos,status,S)[1]
        typeClass = uvLanczos[0].__class__
        Hmat1 = diagonalizeHamiltonian(self.mat,uvLanczos,uS,qtAq)[0]  
        qtAq = typeClass.matrixRepresentation(self.mat,uvLanczos)
        Hmat2 = uS.T.conj()@qtAq@uS
        np.testing.assert_allclose(Hmat1,Hmat2,rtol=1e-5,atol=0)

    def test_orthogonalization(self):
        ''' Returned basis in old form is orthogonal'''
        
        target = calculateTarget(self.evEigh,4)
        sigma = target + self.zpve
        uvLanczos = inexactDiagonalization(self.mat,self.guess,sigma,self.L,
                self.maxit,self.eConv)[1]
        typeClass = uvLanczos[0].__class__
        S = typeClass.overlapMatrix(uvLanczos)
        np.testing.assert_allclose(S,np.eye(S.shape[0]),atol=1e-5) 

    def test_transformationMatrix(self):
        ''' XH@S@X = 1'''
        
        target = calculateTarget(self.evEigh,4)
        sigma = target + self.zpve
        uvLanczos,status = inexactDiagonalization(self.mat,self.guess,sigma,self.L,
                self.maxit,self.eConv)[1:3]
        typeClass = uvLanczos[0].__class__
        S = typeClass.overlapMatrix(uvLanczos)
        assert len(uvLanczos) > 1
        S1 = typeClass.overlapMatrix(uvLanczos[:-1])
        qtAq = typeClass.matrixRepresentation(self.mat,uvLanczos[:-1])
        uS = transformationMatrix(uvLanczos,status,S1)[1]
        uv = diagonalizeHamiltonian(self.mat,uvLanczos,uS,qtAq)[2] 
        uSH = uS@uv
        mat = uSH.T.conj()@S@uSH
        np.testing.assert_allclose(mat,np.eye(mat.shape[0]),atol=1e-5) 
        
    def test_extension(self):
        ''' Checks if extension of matrix works or not'''
        
        target = calculateTarget(self.evEigh,4)
        sigma = target + self.zpve
        uvLanczos = inexactDiagonalization(self.mat,self.guess,sigma,self.L,
                self.maxit,self.eConv)[1]
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
       
        target = calculateTarget(self.evEigh,4)
        sigma = target + self.zpve
        evLanczos, uvLanczos = inexactDiagonalization(self.mat,self.guess,sigma,self.L,
                self.maxit,self.eConv)[0:2] 
        self.assertIsInstance(evLanczos, np.ndarray)
        self.assertIsInstance(uvLanczos, list)
        self.assertIsInstance(uvLanczos[0], TTNSVector)

    def test_eigenvalue(self):
        ''' Checks if the calculated eigenvalue is accurate up to 10*eConv'''
        
        places = [4,8,12,16]
        for p in places:
            target = calculateTarget(self.evEigh,p)
            sigma = target + self.zpve
            evLanczos = inexactDiagonalization(self.mat,self.guess,sigma,self.L,
                    self.maxit,self.eConv)[0]
        
            target_value = find_nearest(evLanczos,sigma)[1]
            closest_value = find_nearest(self.evEigh,sigma)[1]
            self.assertTrue((abs(target_value-closest_value)<= 10*self.eConv),'Not accurate up to 10*eConv')
    
    def test_eigenvector(self):
        ''' Checks if the calculated eigenvector is accurate up to 1e-4
        Provided above test ensures eigenvalues are accurate up to 10*eConv'''

        places = [4,8,12,16]
        for p in places:
            target = calculateTarget(self.evEigh,p)
            sigma = target + self.zpve
            evLanczos, uvLanczos = inexactDiagonalization(self.mat,self.guess,sigma,self.L,
                    self.maxit,self.eConv)[0:2]
        
            idxE = find_nearest(self.evEigh,sigma)[0]
            idxT = find_nearest(evLanczos,sigma)[0]
            
            exactTree= self.uvEigh[idxE]
            lanczosTree= np.ravel(uvLanczos[idxT].ttns.fullTensor(canonicalOrder=True)[0])
            np.testing.assert_allclose(abs(exactTree),abs(lanczosTree),rtol=0,atol=1e-4)

if __name__ == '__main__':
    unittest.main()
