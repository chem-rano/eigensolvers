import numpy as np
import scipy
from scipy import linalg as la
from scipy.sparse.linalg import LinearOperator
from util_funcs import find_nearest, headerBot
from util_funcs import lowdinOrtho, convert
from printUtils import _writeFile
import warnings
import time
import util
from itertools import compress

# -----------------------------------------------------
#    Inexact Lanczos with AbstractClass interface
#------------------------------------------------------

def inexactDiagonalization(H,v0,sigma,L,maxit,eConv,startTime,fout=None,fplot=None,proceed_ortho:bool = False):
    '''
    This is core function to calculate eigenvalues and eigenvectors
    with inexact Lanczos method
    Input::  H => diagonalizable input matrix or linearoperator
             sigma => eigenvalue estimate 
             v0 => eigenvector guess
             conv_tol => eigenvalue convergence tolerance

    Output:: ev as inexact Lanczos computed eigenvalues
             uv as inexact Lanczos computed eigenvectors
    '''
    
    typeClass = v0.__class__
    Ylist = [typeClass.normalize(v0)]
    ev_last = np.inf # for convergence check
    #guessEv = typeClass.matrixRepresentation(H,Ylist)[0][0]
    isConverged = False
    if not proceed_ortho: print("!!! ATTENTION: Doing inexact Lanczos in non-orthogonal basis. \n")
    nCum = 0
    _writeFile(fplot,"it","i","nCum",sep="\t",endline=False)
    _writeFile(fplot,"ev_nearest","check_ev","rel_ev","time",sep="\t")
    zpve = 9837.4069
  
  
    for it in range(maxit):
        for i in range(1,L):
            nCum += 1
            _writeFile(fout,"Lanczos iteration",it+1,"Krylov iteration",i,endline=False)
            _writeFile(fout,"Cumulative Krylov iteration",nCum)

            Ysolved = typeClass.solve(H,Ylist[i-1],sigma)
            Ylist.append(typeClass.normalize(Ysolved))
            
            S = typeClass.overlapMatrix(Ylist)            
            info, uS, idxOrtho = lowdinOrtho(S)                        
            m = uS.shape[1] 
            
            qtAq = typeClass.matrixRepresentation(H,Ylist)  
            #Hmat = qtAq
            Hmat = uS.T.conj()@qtAq@uS                      
            _writeFile(fout,"OVERLAP MATRIX")
            _writeFile(fout,S)
            _writeFile(fout,"HAMILTONIAN MATRIX")
            hamiltonianMatrix = convert(Hmat,"cm-1")
            _writeFile(fout,hamiltonianMatrix)

            #ev, uv = la.eigh(Hmat,S)                          
            ev, uv = la.eigh(Hmat)                          
            idx, ev_nearest = find_nearest(ev,sigma)
            check_ev = util.au2unit(abs(ev_nearest-ev_last),"cm-1") 

            _writeFile(fout,"Eigenvalues")
            for ivalue in range(0,len(ev),1):
                _writeFile(fout,util.au2unit(ev[ivalue],"cm-1")-zpve)
            _writeFile(fplot,it,i,nCum,sep="\t",endline=False)
            _writeFile(fplot,util.au2unit(ev_nearest,"cm-1")-zpve,sep="\t",endline=False)
            _writeFile(fplot,check_ev,check_ev/util.au2unit(ev_nearest,"cm-1"),time.time()-startTime,sep="\t")
            if i == L-1: Ylist = list(compress(Ylist, idxOrtho))
            
            if check_ev <= eConv:
                break
            ev_last = ev_nearest

        if check_ev <= eConv:
            isConverged = True
            x  = []
            for j in range(m):
                x.append(typeClass.linearCombination(Ylist,uv[:,j]))
            Ylist = x
            break
        else:
            x = [typeClass.linearCombination(Ylist,uv[:,idx])]
            Ylist = x
            _writeFile(fout,"Choosing eigevector [idx ",idx,"]: Eigenvalue ",endline=False)
            _writeFile(fout,util.au2unit(ev_nearest,"cm-1")-zpve)

        if (it == maxit-1) and (not isConverged):
            print("Alert:: Lanczos iterations is not converged!")

        _writeFile(fout)# blank line

    return ev,Ylist
# -----------------------------------------------------
if __name__ == "__main__":
    n = 100
    ev = np.linspace(1,300,n)
    np.random.seed(10)
    Q = la.qr(np.random.rand(n,n))[0]
    A = Q.T @ np.diag(ev) @ Q

    
    sigma = 30
    maxit = 4
    L  = 3
    conv_tol = 1e-08
    optionDict = {"linearSolver":"gcrotmk","linearIter":1000,"linear_tol":1e-04}

    Y0 = NumpyVector(np.random.random((n)),optionDict)

    headerBot("Inexact Lanczos")
    print("{:50} :: {: <4}".format("Sigma",sigma))
    print("{:50} :: {: <4}".format("Krylov space dimension",L+1))
    print("{:50} :: {: <4}".format("Eigenvalue convergence tolarance",conv_tol))
    print("\n")
    t1 = time.time()
    lf,xf =  inexactDiagonalization(A,Y0,sigma,L,maxit,conv_tol)
    t2 = time.time()

    print("{:50} :: {: <4}".format("Eigenvalue nearest to sigma",round(find_nearest(lf,sigma)[1],8)))
    print("{:50} :: {: <4}".format("Actual eigenvalue nearest to sigma",round(find_nearest(ev,sigma)[1],8)))
    print("{:50} :: {: <4}".format("Time taken (in sec)",round((t2-t1),2)))
    headerBot("Lanczos",yesBot=True)
