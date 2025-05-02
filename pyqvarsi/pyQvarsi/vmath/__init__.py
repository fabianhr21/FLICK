#!/usr/bin/env python
#
# pyQvarsi, math Module.
#
# Module to compute mathematical operations between
# scalar, vectorial and tensor arrays.
#
# Last rev: 18/11/2020

__VERSION__ = 3.1

# Tensor, Tensor operations
from .vector import dot, cross, outer, scaVecProd, vecTensProd, vecNorm, vecRotate
from .tensor import identity, transpose, trace, det, inverse, matmul, doubleDot, tripleDot, quatrupleDot, scaTensProd, tensVecProd, tensNorm, eigenvalues, schur, tensRotate
from .utils  import linopScaf, linopArrf, maxVal, minVal, maxArr, minArr, deltaKronecker, alternateTensor, reorder1to2
from .csr    import dok_create, csr_create, csr_tocsr, csr_unpack, csr_pack, csr_convert, csr_toarray, csr_identity, csr_transpose, csr_trace, csr_diagonal, csr_spmv, csr_spmm

del vector, tensor, utils, csr