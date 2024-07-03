import sys
from inexact_Lanczos  import *
import numpy as np
from scipy import linalg as la
from numpyVector import NumpyVector
import basis
from printUtils import *
from matplotlib import pyplot as plt


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
status["ovlpRef"] = NumpyVector(uvEigh[:,idx],options) # checked
np.random.seed(13)
Y0 = NumpyVector(np.random.random((N)),options)

guess = Y0
mat = H
eShift = 0.0
L = 5 
maxit = 200  
eConv = 1e-12

if status["writeOut"]:fileHeader("out",options,sigma,L, maxit,eConv,printInfo=False)
if status["writePlot"]:fileHeader("plot",options,sigma,L,maxit,eConv,printInfo=False)
evlanczos,uvlanczos,status = inexactDiagonalization(mat,guess,sigma,L,maxit,eConv,status)
if status["writeOut"]:fileFooter("out",printInfo=False)
if status["writePlot"]:fileFooter("plot",printInfo=False)
