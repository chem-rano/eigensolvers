import numpy as np
import scipy as sp
from util_funcs import find_nearest
from util_funcs import headerBot
import warnings
import time
import util
from numpyVector import NumpyVector
from inexact_Lanczos import inexactLanczosDiagonalization
from printUtils import *


largerDenserSpetra = False

if largerDenserSpetra:
    n = 2500
    ev = np.linspace(1,1400,n)
    np.random.seed(10)
    target = 1290
    maxit = 20
    L = 50
    eConv = 1e-12
    optionsLinear = {"linearSolver":"gcrotmk","linearIter":5000,"linear_tol":1e-4}
    optionDict = {"linearSystemArgs":optionsLinear}

else:# Default small matrix
    n = 100
    ev = np.linspace(1,300,n)
    np.random.seed(10)
    target = 30
    maxit = 4
    L = 6
    eConv = 1e-8
    optionsLinear = {"linearSolver":"gcrotmk","linearIter":1000,"linear_tol":1e-4}
    optionDict = {"linearSystemArgs":optionsLinear}

    
Q = sp.linalg.qr(np.random.rand(n,n))[0]
A = Q.T @ np.diag(ev) @ Q
Y0 = NumpyVector(np.random.random((n)),optionDict)
sigma = target

lf,xf,status = inexactLanczosDiagonalization(A,Y0,sigma,L,maxit,eConv,pick=None,status=None,writeOut=True)
print(status)
