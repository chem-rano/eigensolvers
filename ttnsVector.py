""" Tree Tensor Network State (TTNS) vector wrapper"""
from __future__ import annotations # class returns class in typehint
import numpy as np
from abstractVector import AbstractVector, LINDEP_DEFAULT_VALUE
from typing import List, Optional, Dict
from numbers import Number

import warnings
import copy
from ttns2.state import TTNS
from ttns2.renormalization import AbstractRenormalization, SumOfOperators
from ttns2.sweepAlgorithms import (LinearSystem, Orthogonalization, 
                                StateFitting)
from ttns2.driver import bracket, getRenormalizedOp
from ttns2.driver import overlapMatrix as _overlapMatrix

class TTNSVector(AbstractVector):
    def __init__(self, ttns: TTNS, options:Dict[str, Dict]):
        """ TTNSVector class

        `options` should be an optional dictionary of options containing:
            `sweepAlgorithmArgs` for default options for different sweepalgorithms
            `stateFittingArgs` overwrites `sweepAlgorithmArgs` for `StateFitting`
            `orthogonalizationArgs` overwrites `sweepAlgorithmArgs` for `Orthogonalization`
            `linearSystemArgs` overwrites `sweepAlgorithmArgs` for `LinearSystem`.
        See the respective classes in `SweepAlgorithms` for the possible options.
        """
        self.ttns = ttns
        self.options = options
        # default options
        self.options["sweepAlgorithmArgs"] = options.get("sweepAlgorithmArgs", {"nSweep":1000, "convTol":1e-8})
        op = self.options["sweepAlgorithmArgs"]
        # some changed default options
        op["indent"] = op.get("indent","\t")
        self.options["stateFittingArgs"] = options.get("stateFittingArgs", self.options["sweepAlgorithmArgs"])
        self.options["orthogonalizationArgs"] = options.get("orthogonalizationArgs", self.options["sweepAlgorithmArgs"])
        self.options["linearSystemArgs"] = options.get("linearSystemArgs", self.options["sweepAlgorithmArgs"])
    
    @property
    def hasExactAddition(self):
        """
        Simplication of vector addition with its complex conjugate.
        For example, c+c* = 2c when c=(a+ib)
        This summation is true for numpy vectors
        But does not exactly same as 2c for TTNS
        """
        return False

    @property
    def dtype(self):
        # added to abstractVector
        return np.result_type(*self.ttns.dtypes())

    def __len__(self):
        raise NotImplementedError

    def __mul__(self, other: Number) -> TTNSVector:
        assert isinstance(other, Number)
        warnings.warn("This copies the TTNS. This should be avoided! use inplace")
        new = self.copy()
        new.ttns.rootNode.tens *= other
        return new
    
    def __rmul__(self,other):
        raise NotImplementedError

    def __truediv__(self, other: Number) -> TTNSVector:
        warnings.warn("This copies the TTNS. This should be avoided!")
        new = self.copy()
        new.ttns.rootNode.tens /= other
        return new

    def __imul__(self, other: Number) -> TTNSVector:
        assert isinstance(other, Number)
        self.ttns.rootNode.tens *= other
        return self

    def __itruediv__(self, other: Number) -> TTNSVector:
        assert isinstance(other, Number)
        self.ttns.rootNode.tens /= other
        return self

    def normalize(self) -> TTNSVector:
        # added this function to abstract vector
        self.ttns.normalize()
        return self

    def norm(self) -> float:
        return self.ttns.norm()

    def real(self):
        raise NotImplementedError

    def vdot(self, other: TTNSVector, conjugate=True) -> Number:
        if not conjugate:
            # need to change RenormalizedDot accordingly
            raise NotImplementedError
        return bracket(self.ttns, other.ttns)

    def copy(self) -> TTNSVector:
        return copy.deepcopy(self)

    def applyOp(self, op: AbstractRenormalization) -> TTNSVector:
        warnings.warn("TTNS call to `applyOp`. This should be avoided!")
        # Need to add operators to `StateFitting`
        raise NotImplementedError

    @staticmethod
    def linearCombination(vectors: List[TTNSVector], coeffs:Optional[List[Number]]=None) -> TTNSVector:
        # Initial guess: The one with largest coefficient.
        if coeffs is not None:
            toOpt = vectors[np.argmax(np.abs(coeffs))].copy()
        else:
            # TODO provide b
            norms = [o.norm() for o in vectors]
            toOpt = vectors[np.argmax(norms)].copy()
        solver = StateFitting([v.ttns for v in vectors], toOpt.ttns, coeffs,
                        **vectors[0].options["stateFittingArgs"])
        converged, optVal = solver.run()
        if not converged:
            warnings.warn("linearCombination: TTNS sweeps not converged!")
        return toOpt

    @staticmethod
    def orthogonalize(xs,lindep = LINDEP_DEFAULT_VALUE) -> List[TTNSVector]:
        raise NotImplementedError

    @staticmethod
    def orthogonalize_against_set(x:TTNSVector, vectors:List[TTNSVector],
                                  lindep = LINDEP_DEFAULT_VALUE) -> TTNSVector|None:
        solver = Orthogonalization(vectors, x, **x.options["orthogonalizationArgs"])
        converged, optVal = solver.run()
        if not converged:
            warnings.warn("orthogonalize_against_set: TTNS sweeps not converged!")
        if x.norm()**2 < lindep:
            return None
        else:
            return x

    @staticmethod
    def solve(H, b:TTNSVector, sigma:Number,
              x0: Optional[TTNSVector]=None,
              opType = "her") -> TTNSVector:
        if x0 is None:
            # TODO think about best options.
            #   D=1 TTNS?
            # x0=b corresponds to residual of x0 = 0 (LHS x0 - b)
            # the sign does not matter
            x0 = b.copy()
        op = getRenormalizedOp(x0.ttns, H, x0.ttns)
        if abs(sigma) > 1e-16:
            LHS = SumOfOperators([op, getRenormalizedOp(x0.ttns, -sigma, x0.ttns)])
        else:
            LHS = op
        #assert "lhsOpType" not in x0.options["linearSolverArgs"] # or just delete it in a copy of the dict
        assert "lhsOpType" not in x0.options["linearSystemArgs"] # or just delete it in a copy of the dict
        solver = LinearSystem(x0.ttns if x0 is not None else None,
                              LHS,
                              b.ttns,
                              lhsOpType = opType,
                              **x0.options["linearSystemArgs"])
                              #**x0.options["linearSolverArgs"])
        converged, val = solver.run()
        if not converged:
            warnings.warn("solve: TTNS sweeps not converged!")
        return x0

    @staticmethod
    def matrixRepresentation(operator, vectors:List[TTNSVector]):
        # vv assuming that operator has the same dtype
        dtype = np.result_type(*[v.dtype for v in vectors])
        N = len(vectors)
        M = np.empty([N,N],dtype=dtype)
        for i in range(N):
            bra = vectors[i].ttns
            for j in range(i, N):
                ket = vectors[j].ttns
                val = getRenormalizedOp(bra, operator, ket).bracket()
                M[i, j] = val
                M[j, i] = val.conj()
        return M

    @staticmethod
    def overlapMatrix(vectors:List[TTNSVector]):
        ''' Calculates overlap matrix of tensor network states'''
        return _overlapMatrix([v.ttns for v in vectors])
    
    @staticmethod
    def extendMatrixRepresentation(operator, vectors:List[TTNSVector],qtAq:ndarray):
        dtype = np.result_type(*[v.dtype for v in vectors])
        N = len(vectors)
        if N < 3:               
            qtAq = np.empty([N,N],dtype=dtype)
            for i in range(N):
                bra = vectors[i].ttns
                for j in range(i, N):
                    ket = vectors[j].ttns
                    val = getRenormalizedOp(bra, operator, ket).bracket()
                    qtAq[i, j] = val
                    qtAq[j, i] = val.conj()
        else:
            M = np.empty(N,dtype=dtype)
            bra = vectors[-1].ttns
            for i in range(1, N):
                ket = vectors[i].ttns
                M[i] = getRenormalizedOp(bra, operator, ket).bracket()
            offD = np.array([M[:-1]])
            np.concatenate((qtAq,offD.T.conj()),axis=1)
            np.vstack((qtAq,M[np.newaxis]))
        return qtAq
 
    @staticmethod
    def extendOverlapMatrix(vectors:List[TTNSVector],oMat:ndarray):
        ''' Calculates overlap elements of last tensor networks state'''
        
        dtype = np.result_type(*[v.dtype for v in vectors])
        N = len(vectors)
        elems = np.empty([N],dtype=dtype)

        for i in range(N):
            elems[i] = vectors[i].vdot(vectors[-1],True)
        offD = np.array([elems[:-1]])
        np.concatenate((oMat,offD.T.conj()),axis=1)
        np.vstack((oMat,elems[np.newaxis]))
        return oMat
