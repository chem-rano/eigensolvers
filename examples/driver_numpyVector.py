import numpy as np
import scipy as sp
from util_funcs import find_nearest, lowdinOrtho
from util_funcs import headerBot
from printUtils import *
import warnings
import time
import util
from numpyVector import NumpyVector
from inexact_Lanczos import inexactDiagonalization

n = 100
ev = np.linspace(1,300,n)
np.random.seed(10)
Q = sp.linalg.qr(np.random.rand(n,n))[0]
A = Q.T @ np.diag(ev) @ Q

target = 30
maxit = 4 
L = 6 
eConv = 1e-8
zpve = 0.0
    
optionDict = {"linearSolver":"gcrotmk","linearIter":1000,"linear_tol":1e-04}
printChoices = {"Iteration details": True,"Plot data": True, "eShift":zpve, "convertUnit":"au"}
Y0 = NumpyVector(np.random.random((n)),optionDict)
sigma = target + zpve

headerBot("Inexact Lanczos")
print("{:50} :: {: <4}".format("Sigma",sigma))
print("{:50} :: {: <4}".format("Krylov space dimension",L+1))
print("{:50} :: {: <4}".format("Eigenvalue convergence tolarance",eConv))
print("\n")
lf,xf,status = inexactDiagonalization(A,Y0,sigma,L,maxit,eConv,printChoices)

print("{:50} :: {: <4}".format("Eigenvalue nearest to sigma",round(find_nearest(lf,sigma)[1],8)))
print("{:50} :: {: <4}".format("Actual eigenvalue nearest to sigma",round(find_nearest(ev,sigma)[1],8)))
headerBot("Lanczos",yesBot=True)
