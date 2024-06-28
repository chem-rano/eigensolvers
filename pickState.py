import numpy as np
from util_funcs import find_nearest
from numpyVector import NumpyVector
from ttnsVector import TTNSVector


def pick_maxOvlp(vectors,refVector,refIncld="True"):
    typeClass = refVector.__class__
    ovlp = [abs(refVector.vdot(vectors[i],True)) for i in range(len(vectors))]
    idxArray = np.argsort(ovlp)[::-1]
    # maximum overlap is the first one unless ref included in Ylist 
    idx = idxArray[1] if refIncld else idxArray[0]  
    return idx 

def pick_sigma(ev,sigma):
    idx = find_nearest(ev,sigma)
    return idx
