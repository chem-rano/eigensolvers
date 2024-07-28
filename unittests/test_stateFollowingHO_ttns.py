import unittest
import sys
import numpy as np
from scipy import linalg as la
from ttnsVector import TTNSVector
import basis
from util_funcs import get_pick_function_close_to_sigma
from util_funcs import get_pick_function_maxOvlp
import mctdh_stuff
import basis
import operatornD
from ttns2.driver import eigenStateComputations
from ttns2.diagonalization import IterativeDiagonalizationOptions
from ttns2.parseInput import parseTree
from ttns2.contraction import TruncationEps
from inexact_Lanczos import inexactDiagonalization
from ttns2.diagonalization import IterativeLinearSystemOptions
import util
from util_funcs import find_nearest


class Test_stateFollowing(unittest.TestCase):

    def setUp(self):
        MAX_D = 10 
        EPS = 5e-9
        convTol = 1e-5
        N_STATES = 8

        fOp = '../examples/ch3cn.op'  # this one is used for HRL's 2019 jcp work
        Hop = mctdh_stuff.translateOperatorFile(fOp, verbose=False)

        N = 42
        DVRopts = [
            basis.Hermite.getOptions(N=N , HOx0=0, HOw=1, HOm=1),
            basis.Hermite.getOptions(N=N , HOx0=0, HOw=1, HOm=1),
            basis.Hermite.getOptions(N=N , HOx0=0, HOw=1, HOm=1),
            basis.Hermite.getOptions(N=N , HOx0=0, HOw=1, HOm=1),
            basis.Hermite.getOptions(N=N , HOx0=0, HOw=1, HOm=1),
            basis.Hermite.getOptions(N=N , HOx0=0, HOw=1, HOm=1),
            basis.Hermite.getOptions(N=N , HOx0=0, HOw=1, HOm=1),
            basis.Hermite.getOptions(N=N , HOx0=0, HOw=1, HOm=1),
            basis.Hermite.getOptions(N=N , HOx0=0, HOw=1, HOm=1),
            basis.Hermite.getOptions(N=N , HOx0=0, HOw=1, HOm=1),
            basis.Hermite.getOptions(N=N, HOx0=0, HOw=1, HOm=1),
            basis.Hermite.getOptions(N=N, HOx0=0, HOw=1, HOm=1),
        ]
        
        treeString = """
            0> 3 3 3
                1> 3 3
                    2> [x1]
                    2> 3 3
                        3> [x5]
                        3> [x6]
                1> 3 3
                    2> 3 3
                        3> [x7]
                        3> [x8]
                    2> 3 3
                        3> [x9]
                        3> [x10]
                1> 3 3
                    2> 3 3 
                        3> [x3]
                        3> 3 3
                            4> [x2]
                            4> [x4]
                    2> 3 3 
                        3> [x11] 
                        3> [x12]
            """
        bases = [basis.basisFactory(o) for o in DVRopts]
        nBas = [b.N for b in bases]
        Hop.storeMatrices(bases)
        Hop = operatornD.contractSoPOperatorSimpleUsage(Hop)
        operatornD.absorbCoeff(Hop)
        Hop.obtainMultiplyOp(bases)
        basisDict = {l:b for l,b in zip(Hop.DoFlabel, bases)}
        tns = parseTree(treeString, basisDict, returnType="TTNS")
        np.random.seed(13)
        tns.setRandom()
        tns.toPdf()


        davidsonOptions = [IterativeDiagonalizationOptions(tol=1e-7, maxIter=500,maxSpaceFac=200)] * 8
        # tighter convergence 
        davidsonOptions.append(IterativeDiagonalizationOptions(tol=1e-8, maxIter=500,maxSpaceFac=200))
        bondDimensionAdaptions = [TruncationEps(EPS, maxD=MAX_D, offset=2, truncateViaDiscardedSum=False)]
        noises = [1e-6] * 4 + [1e-7] * 4 + [1e-8] * 6
        tnsList, energies = eigenStateComputations(tns, Hop,
                                     nStates=N_STATES,
                                     nSweep=999,
                                     projectionShift=util.unit2au(9999,"cm-1"),
                                     iterativeDiagonalizationOptions=davidsonOptions,
                                     bondDimensionAdaptions= bondDimensionAdaptions,
                                     noises = noises,
                                     allowRestart=False,
                                     saveDir=None,
                                     convTol=convTol)
        bondDimensionAdaptionsOrtho = [TruncationEps(EPS, maxD=10, offset=2, truncateViaDiscardedSum=False)]
        bondDimensionAdaptionsFitting = [TruncationEps(EPS, maxD=10, offset=2, truncateViaDiscardedSum=False)]

        maxit = 10
        L = 6
        eConv = 1e-6 
        zpve = 9837.4069  
        idx = 1 
        #target = energies[idx]
        target = (energies[idx]-zpve)+1.0
        nsweepOrtho = 800
        orthoTol = 1e-08
        optShift = 0.0

        #from magic import ipsh
        #ipsh()
        #quit()
        siteLinearTol = 1e-3
        globalLinearTol = 1e-2
        nsweepLinear = 1000


        fittingTol = 1e-9
        nsweepFitting = 1000

        optsCheck = IterativeLinearSystemOptions(solver="gcrotmk",tol=siteLinearTol,maxIter=70000) 
        optionsOrtho = {"nSweep":nsweepOrtho, "convTol":orthoTol, "optShift":optShift, "bondDimensionAdaptions":bondDimensionAdaptionsOrtho}
        optionsLinear = {"nSweep":nsweepLinear, "iterativeLinearSystemOptions":optsCheck,"convTol":globalLinearTol}
        optionsFitting = {"nSweep":nsweepFitting, "convTol":fittingTol,"bondDimensionAdaptions":bondDimensionAdaptionsFitting}
        #options = {"linearSystemArgs":optionsLinear}
        options = {"orthogonalizationArgs":optionsOrtho, "linearSystemArgs":optionsLinear, "stateFittingArgs":optionsFitting}

        status = {"eShift":zpve, "convertUnit":"cm-1",
                "writeOut": True,"writePlot": True}
        ovlpRef = TTNSVector(tnsList[idx+1],options)
        tns = TTNSVector(tns,options)
        self.pick = get_pick_function_maxOvlp(ovlpRef)
        
        self.target = target
        self.evEigh = energies
        self.uvEigh = tnsList
        self.guess = tns
        self.mat = Hop
        self.eShift = zpve
        self.L = L
        self.eConv = eConv
        self.maxit = maxit
        self.ovlpRef = ovlpRef
        self.status = status

    def test_eigenvalue(self):
        ''' Checks if the calculated eigenvalue is accurate to 10*eConv'''

        sigma = util.unit2au((self.target+self.eShift),unit="cm-1")
        eConvAU = util.unit2au(self.eConv,unit="cm-1")
        evLanczos, uvlanczos,status = inexactDiagonalization(self.mat,self.guess,sigma,self.L,
                self.maxit,self.eConv,self.pick,self.status)
        
        typeClass = uvlanczos[0].__class__
        reference = typeClass.matrixRepresentation(self.mat,[self.ovlpRef])[0]
        closest = find_nearest(self.evEigh,reference)[1]        # comapring with actual
        relError = abs(reference-closest)/(max(abs(reference), 1e-14))
        self.assertTrue((relError <= 1e-4),'Not accurate up to 1e-4')

    def test_overlap(self):
        """ Mainly looking at the overap if the eigenvector is inclined to the reference vector"""

        sigma = util.unit2au((self.target+self.eShift),"cm-1")
        eConvAU = util.unit2au(self.eConv,unit="cm-1")
        evlanczos, uvlanczos,status = inexactDiagonalization(self.mat,self.guess,sigma,self.L,
                self.maxit,eConvAU,self.pick,self.status)
        idx = find_nearest(evlanczos,sigma)[0]

        refTree = np.ravel(self.ovlpRef.ttns.fullTensor(canonicalOrder=True)[0])
        ttnsT = np.ravel(uvLanczos[idx].ttns.fullTensor(canonicalOrder=True)[0])
        ovlp = np.vdot(ttnsT,refTree)
        np.testing.assert_allclose(abs(ovlp), 1, rtol=1e-5, err_msg = f"{ovlp=} but it should be +-1")

if __name__ == "__main__":
    unittest.main()
