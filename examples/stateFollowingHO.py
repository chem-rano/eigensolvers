import sys
from inexact_Lanczos  import *
import numpy as np
from scipy import linalg as la
from numpyVector import NumpyVector
import basis
from printUtils import *
from matplotlib import pyplot as plt
from util_funcs import get_pick_function_close_to_sigma
from util_funcs import get_pick_function_maxOvlp


xRange = [-10,10]
N = 45
sincInfInfOpts = basis.SincInfInf.getOptions(N=N, xRange=xRange)
sincInfInf = basis.SincInfInf(sincInfInfOpts)

T = -sincInfInf.mat_dx2
V = np.diag(sincInfInf.xi**2)
H  = T + V
evEigh, uvEigh = la.eigh(H)

optionsLinear = {"linearSolver":"gcrotmk","linear_tol":1e-04}
options = {"linearSystemArgs":optionsLinear}
status = {"writeOut": True,"writePlot": True,"stateFollowing":"maxOvlp"}
status["actualEvalues"] = evEigh
sigma = 11.1
idx = find_nearest(evEigh,sigma)[0]
ovlpRef = NumpyVector(uvEigh[:,idx+1],options)
pick = get_pick_function_maxOvlp(ovlpRef)
np.random.seed(13)
Y0 = NumpyVector(np.random.random((N)),options)

guess = Y0
mat = H
eShift = 0.0
L = 5 
maxit = 200  
eConv = 1e-10

evlanczos,uvlanczos,status = inexactLanczosDiagonalization(mat,guess,sigma,L,maxit,
        eConv,pick=pick,status=status)
