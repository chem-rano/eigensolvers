import numpy as np
from scipy import linalg as la
import sys
from scipy import special
import copy
from numpyVector import NumpyVector
import util

LINDEP_DEFAULT_VALUE = 1e-14 # Global variable

# -----------------------------------------------------
def trapezoidal(nc):
    '''
    input: nc, user-defined number of quadrature points
    output: quadrature ponits and associated weights
    '''
    points=np.zeros(nc)
    weights=np.zeros(nc)
    a=-1.0
    b=1.0
    dx=(b-a)/(nc)
    for i in range(nc):
        points[i]=a+dx*(i-1)
        weights[i]=(b-a)/(nc+1)
    return (points,weights)

# -----------------------------------------------------
# This function is copied from cross_filterdiag.py
def eigRegularized(A, B, Q, tol):
    """ Solve generalized eigenvalue problem by discarding all eigenvectors of B smaller than tol.
    :returns Eigenvalue, eigenvectors
    """
    if B is None:
        Bq = (Q.T.conj())@Q
    else:
        Bq = (Q.T.conj())@B@Q
    eBq,uBq = la.eigh(Bq)
    #eBq,uBq = la.eig(Bq)
    #    vv not abs => negative eigenvalues are even worse!
    idx = eBq > tol
    eBq = eBq[idx]
    uBq = uBq[:,idx]
    # TODO Here space is being truncated
    uBqTraf = uBq * eBq**(-0.5)
    Q_trun = Q @ uBqTraf # HRL suggested

    AqTraf = Q_trun.T.conj() @ (A @ Q_trun)
    #AqTraf = Q_trun.T.conj() @ Q_trun
    ev, uvTraf = la.eigh(AqTraf)

    uv = uBqTraf @ uvTraf
    return ev,Q@uv, Q_trun
# -----------------------------------------------------
def eigRegularized_list(Amat,B, Q, atol):
    mQ = len(Q)
    nQ = len(Q[0])
    dtype = Q[0].dtype
    qtAq = np.zeros((mQ,mQ),dtype=dtype)

    for j in range(mQ):
        Aqj = Amat @ Q[j]
        for i in range(mQ):
            qtAq[i,j] = np.vdot(Q[i],Aqj)
            qtAq[j,i] = qtAq[i,j]
    #if B is None:
    evals, uvals = la.eigh(qtAq)
    
    mu,nu = uvals.shape
    res = np.zeros((mQ,nQ))
   
    for j in range(nu):
        for k in range(mQ):
            res[j,:] += uvals[k,j] * Q[k] 
    return evals, res
# -----------------------------------------------------
#find maximum residual inside contour
def getRes(lest,x,resvecs,eps):
    '''
    Calculates maximum residual inside the contour subspace
    '''
    n,m0 = x.shape
    resnorms = np.zeros(m0)

    # step1: creates an array with residual norms (divided by x norms) 
    for i in range(m0):
        if (la.norm(x[:,i]) > 1e-14):
            resnorms[i]= la.norm(resvecs[:,i])/la.norm(x[:,i])

    # step2: checks if how many resnorms follow convergence
    s = []
    nsubspace = 0                                     # n subspace states satisfying convergence criteria
    for k in range(m0):
        if(resnorms[k] < eps):                        # residual norm is less than a specified small number; 
            nsubspace = nsubspace + 1                 # when feast subspace is identical to exact space; it is zero.
            s.append(k)
    
    # step3: if all resnorms achieved convergence, then return maximum value from them
    #         otherwise go through all residual norms and print out the maximum from them
    maxres = 0.0
    if(nsubspace == 0):
        maxres = np.max(resnorms)
    else:
        for k in range(m0):
            if(resnorms[k] < eps): 
                tmp = resnorms[k]
                if(tmp > maxres):                    # finding the maximum residual value
                    maxres = tmp

    return maxres


# -----------------------------------------------------
def print_a_range(in_arr, arr_min, arr_max):
    '''
    Prints elements of an array(in_arr) from the selected interval rr_min and arr_max
    '''
    out_arr = []
    n = len(in_arr)
    for i in range(n):
        if in_arr[i] >= arr_min and in_arr[i] <= arr_max:
            #out_arr.append(round(in_arr[i],8))
            out_arr.append(in_arr[i])
    return out_arr
# -----------------------------------------------------
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]

# -----------------------------------------------------
def nearest_degenerate(array, value):
    array = np.asarray(array)
    #Searches for duplicate element
    degen = 0
    for i in range(0, len(array)):
        for j in range(i+1, len(array)):
            #if(array[i] == array[j]):
            if(abs(array[i] - array[j]) <= 1e-6):
                degen += 1
    if degen > 0 :
        print("Got degenracy")
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]
# -----------------------------------------------------
def print_krylov(in_arr, E, L):
    '''
#    Prints L elements of an array (in_arr) near to E
#    '''
    idx, value = find_nearest(in_arr,E)
    lower_idx = idx - int(L/2) 
    upper_idx = idx + int(L/2) 
    out_arr = in_arr[lower_idx:upper_idx]
    return out_arr
# -----------------------------------------------------
#gk,wk = trapezoidal(nc)
#gk,wk = special.roots_hermite(nc)
#gk,wk = special.roots_legendre(nc)

def quad_func(nc,quad):
    if quad == "legendre":
        gk,wk = special.roots_legendre(nc)
    elif quad == "hermite":
        gk,wk = special.roots_hermite(nc)
    elif quad == "trapezoidal":
        #gk,wk == trapezoidal(nc)
        gk,wk = trapezoidal(nc)
    return gk,wk

# -----------------------------------------------------
# _qr function is copied from pyscf
def _qr(xs, dot, lindep= LINDEP_DEFAULT_VALUE):
    '''QR decomposition for a list of vectors (for linearly independent vectors only).
    xs = (r.T).dot(qs)
    '''
    nvec = len(xs)
    dtype = xs[0].dtype
    qs = np.empty((nvec,xs[0].size), dtype=dtype)
    rmat = np.empty((nvec,nvec), order='F', dtype=dtype)

    nv = 0
    for i in range(nvec):
        xi = np.array(xs[i], copy=True)
        rmat[:,nv] = 0
        rmat[nv,nv] = 1
        for j in range(nv):
            prod = dot(qs[j].conj(), xi)
            xi -= qs[j] * prod
            rmat[:,nv] -= rmat[:,j] * prod
        innerprod = dot(xi.conj(), xi).real
        norm = np.sqrt(innerprod)
        if innerprod > lindep:
            qs[nv] = xi/norm
            rmat[:nv+1,nv] /= norm
            nv += 1
    return qs[:nv], np.linalg.inv(rmat[:nv,:nv])
# -----------------------------------------------------
def headerBot(method,yesBot=False):
    nstars = 45
    if yesBot == False:
        print("*"*nstars)
        print(f"         {method}           ")
        print("*"*nstars)
        print("\n")
    elif yesBot == True:
        print("*"*nstars)
        print("  computation complete      ")
        print("*"*nstars)

# -----------------------------------------------------
def lowdinOrtho(oMat, tol= LINDEP_DEFAULT_VALUE):
    """ Extracts out linearly independent vectors from the overlap matrix `oMat` and 
    returns orthogonalized vectors (vector*S-1/2).
    :returns info (is all linear independent: True or not), vectors
    """
    evq, uvq = la.eigh(oMat)
    idx = evq > tol
    evq = evq[idx]
    uvq = uvq[:,idx]
    
    info = all(idx)
    uvqTraf = uvq * evq**(-0.5)
    return info, uvqTraf

def eigenvalueResidual(ev,prev_ev):

    m0 = len(ev)
    diff = 0.0
    prev_tot = 0.0

    for i in range(m0):
        diff += abs(prev_ev[i]-ev[i])
        prev_tot += prev_ev[i]
    res = diff/prev_tot
    return res
# -----------------------------------------------------
def calculateTarget(eigenvalues, indx, tol=1e-14):
    ''' Calculates target for the given 
    eigenvalues (here, exact eigenvalues) for a specific 
    index (indx)
    Checks if the nearest eigenvalues are not degenerate
    for tolerance, tol (default: 1e-14)'''

    ediff1 = eigenvalues[indx] - eigenvalues[indx-1] 
    ediff2 = eigenvalues[indx+1] - eigenvalues[indx] 
    assert min(ediff1,ediff2) > tol, "Got a degenerate eigenvalue"
    target = eigenvalues[indx] + min(ediff1,ediff2)*0.25
    return target

