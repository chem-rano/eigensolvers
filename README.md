# Description
Targeted eigensolvers to find an eigenstate (or a set of eigenstates) in a specified range.
The main implementation is an inexact Lanczos approach[1,3]. 
Another, preliminary and not fully tested implementation is the FEAST approach[2].
These implementations are implemented in a general way that allows both numpy arrays (NumpyVector wrapper) and tensor network states (TTNSVector wrapper). Currently, the latter it is based on in-house code, which will be released separately soon.
Note: **This is work in progress**. 

# Theoretical background
Suppose **H** is a hermitian matrix whose eigenvectors and eigenvectors are to be calculated. 
In many cases, the whole eigen spectrum is not required. Let's assume, we specifically want some eigenvectors near the  eigenvalue $\sigma$.
In this case, instead of solving the actual matrix, **H**, it is often better to solve the transformed form, **F(H)**. 
This transformation is often called spectral transform and it is chosen to broaden the gap between desired eigenvalues (i.e., $\sigma$) and other eigenvalues. 
Due to the larger separation in the eigenvalues, this part of the spectrum is then easier to converge and hence fewer iterations are needed, compared to actual **H**.
To achieve a widely separated spectrum near $\sigma$, one straightforward way is the shift-and-inveer approach, which uses $F(H)=(\sigma -H)^{-1}$.
F(**H**)**v** is calculated by iteratively and approximately by solving the linear system ($\sigma$**I**-**H**)**w** = **v**. 
The eigenvalue problem in the **w** basis is then solved to obtain eigenvalues and corresponding eigenvectors, resulting in the inexact Lanczos approach.
In the FEAST approach, the eigenvalues are computed through contour integration.
Rather than targeting a specific eigenvalue, it is aimed to find eigenvalues within a specified range.

# Prerequisites (recommended version)
1. Python (3.10.14)
2. SciPy (1.10.1)
3. NumPy (1.26.4)

# Unittest
User is recommended to run following unit tests in folder, "unittests" before NumpyVector tests
1. test_lanczos.py 
2. test_lanczosBlock.py
3. test_lanczosLINDEP.py

# Working examples
Example (driver_numpyVector.py) can be found at folder, "examples".

# Input arguments
Inexact Lanczos eigensolver
1. H  		: diagonalizable input matrix or linearoperator
2. v0 		: eigenvector guess
     		  Can be a list of `AbstractVectors`.
     		  Then, block Lanczos is performed (Krylov space on each of the guesses).
     		  Note that the guess vectors should be orthogonal.
3. sigma 		: eigenvalue estimate
4. L  		: Krylov space dimension
5. maxit 		: Maximum Lanczos iterations
6. eConv 		: relative eigenvalue convergence tolerance
7. checkFitTol 
(optional) 	: checking tolerance of fitting
8. Hsolve
 (optional) 	: As H but only used for the generation of the Lanczos vectors.
                  `H` is then used for diagonalizing the Hamiltonian matrix
9. writeOut
(optional) 	: writing file instruction
             	  default : write both iteration_lanczos.out & summary_lanczos.out
10. eShift 
(optional) 	: shift value for eigenvalues, Hmat elements
11. convertUnit 
(optional) 	: convert unit for eigenvalues, Hmat elements
12. pick 
(optional) 	: pick function for eigenstate
                  Default is get_pick_function_close_to_sigma
13. status 
(optional) 	: Additional information dictionary
                  (more details see _getStatus doc)
14. outFileName 
(optional)	: output file name
15. summaryFileName
(optional)	: summary file name

# Output files
After successful test and running of example driver file there will following output files:
1. iterations_lanczos.out (Detailed information at each cumulative iteration. Information include
parameter details in file header, overlap matrix, Hamiltonian matrix before and after diagonalization, eigenvalues. )
2. summary_lanczos.out (Summary information at each cumulative iteration along with parameter details in file header.)

# Contributors
1. Dr. Madhumita Rano
2. Prof. Henrik R. Larsson (https://github.com/hrlGroup)

# References
1. Shi-Wei Huang and Tucker Carrington Jr., “A new iterative method for calculating energy levels and
wave functions”, The Journal of Chemical Physics 112.20 (2000), pp. 8765–8771.
2. Eric Polizzi., “Density-matrix-based algorithm for solving eigenvalue problems”, Physical Review B
79.11 (2009), p. 115112.
3. Madhumita Rano and Henrik R. Larsson, Computing excited eigenstates using inexact Lanczos methods and tree tensor network states, 
arXiv preprint arXiv:2506.22574, 2025. 
