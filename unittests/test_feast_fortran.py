import unittest
import sys
from feast  import *
from magic import ipsh
import numpy as np
from scipy import linalg as la
from numpyVector import NumpyVector
from util_funcs import find_nearest
import time
from util_funcs import get_a_range
import math
from feast import *

# This tests our FEAST code with outputs of FEAST Fortran code (by Eric Polizzi)

filename = "data_fortranCode.out"

def read_fortranData(k=0):
    amat = np.loadtxt(filename, dtype = float,skiprows =1, max_rows=4)
    guess = np.loadtxt(filename,dtype =complex,skiprows=6, max_rows=3)
    xe = np.loadtxt(filename,dtype =float,skiprows=12, max_rows=8)
    we = np.loadtxt(filename,dtype =float,skiprows=22, max_rows=8)
    theta = np.loadtxt(filename,dtype =float,skiprows=32, max_rows=8)
    zne = np.loadtxt(filename,dtype =complex,skiprows=42, max_rows=8)
    
    Qe = np.loadtxt(filename,dtype =complex,skiprows=62+k*5, max_rows=3)
    Q = np.loadtxt(filename,dtype =float,skiprows=102+k*5, max_rows=3)
    return amat,guess,xe,we,theta,zne,Qe,Q

class Test_feast_fortran(unittest.TestCase):

    def setUp(self):
        A = read_fortranData()[0]
        n = A.shape[0]
        self.rmin = 3.0
        self.rmax = 5.0
        self.nc = 8            # number of contour points
        self.quad = "legendre" # Choice of quadrature points
        m0 = 3                 # subspace dimension
        self.eConv = 1e-12      # residual convergence tolerance
        self.maxit = 10        # maximum FEAST iterations
        self.order = [4,3,5,2,6,1,7,0]
        
        options = {"linearSolver":"pardiso"}
        optionsDict = {"linearSystemArgs":options}
        
        Y1 = read_fortranData()[1]
        Y = []
        for i in range(m0):
            Y.append(NumpyVector(Y1[i,:], optionsDict))

        self.guess = Y
        self.mat = A

        evEigh, uvEigh = np.linalg.eigh(A)
        self.evEigh = evEigh
        self.uvEigh = uvEigh
    
    def test_legendre_pointts(self):
        fgk,fwk = read_fortranData()[2:4]
        gk,wk = quad_func(self.nc,self.quad)
        np.testing.assert_allclose(fgk,gk[self.order],rtol=1e-5,atol=0)
        np.testing.assert_allclose(fwk,wk[self.order],rtol=1e-5,atol=0)

    def test_theta(self):
        ftheta= read_fortranData()[4]
        gk = quad_func(self.nc,self.quad)[0]
        pi = np.pi
        theta = np.empty((self.nc))
        for k in range(self.nc):
            theta[k] = -(pi*0.5)*(gk[k]-1)
        np.testing.assert_allclose(ftheta,theta[self.order],rtol=1e-5,atol=0)
   
    def test_zne(self):
        fzne= read_fortranData()[5]
        efactor = 0.3
        r = abs(self.rmax-self.rmin)*0.5
        gk = quad_func(self.nc,self.quad)[0]
        pi = np.pi
        zne = np.empty((self.nc),dtype=complex)
        for k in range(self.nc):
            theta = -(pi*0.5)*(gk[k]-1)
            zne[k] = ((self.rmin+self.rmax)*0.5)+ r*math.cos(theta)+r*efactor*1.0j*math.sin(theta)
        np.testing.assert_allclose(fzne,zne[self.order],rtol=1e-5,atol=0)

    def test_Qe(self):
        typeClass = self.guess[0].__class__
        efactor = 0.3
        r = abs(self.rmax-self.rmin)*0.5
        gk,wk = quad_func(self.nc,self.quad)
        pi = np.pi
        zne = np.empty((self.nc),dtype=complex)
        n,m = len(self.guess),len(self.guess[0].array)
        Qe = np.empty((n,m),dtype=complex)
        for k in range(self.nc):
            theta = -(pi*0.5)*(gk[k]-1)
            zne[k] = ((self.rmin+self.rmax)*0.5)+ r*math.cos(theta)+r*efactor*1.0j*math.sin(theta)
        
        zne = zne[self.order]    
        for k in range(self.nc):
            fQe = read_fortranData(k)[6]
            for im0 in range(len(self.guess)):
                Qe[im0] = typeClass.solve(self.mat,self.guess[im0],zne[k]).array
            np.testing.assert_allclose(Qe,fQe,rtol=1e-5,atol=0)

    def test_Q(self):
        typeClass = self.guess[0].__class__
        efactor = 0.3
        r = abs(self.rmax-self.rmin)*0.5
        gk,wk = quad_func(self.nc,self.quad)
        pi = np.pi
        theta = np.empty((self.nc))
        zne = np.empty((self.nc),dtype=complex)
        n,m = len(self.guess),len(self.guess[0].array)
        Q = [np.nan for it in range(n)]
        for k in range(self.nc):
            theta[k] = -(pi*0.5)*(gk[k]-1)
        
        theta = theta[self.order]
        wk = wk[self.order]
        for k in range(self.nc):
            fQ = read_fortranData(k)[7]
            zne[k] = ((self.rmin+self.rmax)*0.5)+ r*math.cos(theta[k])+r*efactor*1.0j*math.sin(theta[k])
            for im0 in range(len(self.guess)):
                Qe = typeClass.solve(self.mat,self.guess[im0],zne[k])
                mult = -2.00*wk[k]*0.25*r*(efactor*math.cos(theta[k])+math.sin(theta[k])*1.00j)
                Qquad_k = typeClass.real(mult*Qe)
                #Qquad_k = calculateQuadrature(self.mat,self.guess[im0],zne[k],r,theta[k],wk[k])
                Q = updateQ(Q,im0,Qquad_k,k)
            for im0 in range(len(self.guess)):
                np.testing.assert_allclose(Q[im0].array,fQ[im0],rtol=1e-5,atol=0)

if __name__ == '__main__':
    unittest.main()

