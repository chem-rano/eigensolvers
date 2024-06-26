import numpy as np
import scipy as sp
from util_funcs import find_nearest
from util_funcs import headerBot
import warnings
import time
import util
from numpyVector import NumpyVector
from inexact_Lanczos import inexactDiagonalization


# Default small matrix
n = 100
ev = np.linspace(1,300,n)
np.random.seed(10)
target = 30
maxit = 4
L = 6
eConv = 1e-8
optionDict = {"linearSolver":"gcrotmk","linearIter":1000,"linear_tol":1e-4}


largerDenserSpetra = False
if largerDenserSpetra:
    n = 2500
    ev = np.linspace(1,1400,n)
    np.random.seed(10)
    target = 1290
    maxit = 20
    L = 50
    eConv = 1e-12
    optionDict = {"linearSolver":"gcrotmk","linearIter":5000,"linear_tol":1e-4}
    
Q = sp.linalg.qr(np.random.rand(n,n))[0]
A = Q.T @ np.diag(ev) @ Q
status = {"writeOut": False,"writePlot": False}
Y0 = NumpyVector(np.random.random((n)),optionDict)
sigma = target

headerBot("Inexact Lanczos")
print("{:50} :: {: <4}".format("Sigma",sigma))
print("{:50} :: {: <4}".format("Krylov space dimension",L+1))
print("{:50} :: {: <4}".format("Eigenvalue convergence tolarance",eConv))
print("\n")
lf,xf,status = inexactDiagonalization(A,Y0,sigma,L,maxit,eConv,status)
print(status)

print("{:50} :: {: <4}".format("Eigenvalue nearest to sigma",round(find_nearest(lf,sigma)[1],8)))
print("{:50} :: {: <4}".format("Actual eigenvalue nearest to sigma",round(find_nearest(ev,sigma)[1],8)))
headerBot("Lanczos",yesBot=True)
