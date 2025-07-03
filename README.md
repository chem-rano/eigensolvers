# Description
Targeted eigensolvers find an eigenstate (or a set eigenstates) of the eigenvalue problem in a specified range of eigenvalue.
Here, two targeted eigensolvers, namely, inexact Lanczos [1] and FEAST [2] have been implemented.
These implementations are general with respect to eigenvector class. Types of eigenvector class are
NumpyVector (numpy arrays), TTNSVector (Tree Tensor Network), and so on. In this version, NumpyVector
class is available to work with. While TTNSVector has been used for this research work (https://arxiv.org/abs/2506.22574),
it is based on in-house TTNS code and is not accessible for testing. 
Note: FEAST implementation is under development. For further details, 
these two sources (https://github.com/certik/feast, https://github.com/brendanedwardgavin/feastjl, )are advised to check out. 

# Theoretical background
Suppose \textbf{H} is a NxN symmetric matrix whose eigenvectors and eigenvectors are to be calculated. 
In many cases, whole eigen spectra are not required. Let's assume, we specifically want some eigenvectors near eigenvalue $\sigma$.

For a higher eigenvalue spectrum with many-fold degeneracies (relevant to higher energy spectra), instead of solving the actual
matrix, \textbf{H}, it is better to solve the transformed form, \textbf{F(H)}. 
This transformation is often called spectral transform and it is chosen to broaden the gap between desired eigenvalues (i.e., $\sigma$). 
Due to the larger separation in the eigenvalues, this part of the spectrum is easy to converge and hence fewer iterations are needed as compared to actual \textbf{H}.
To achieve a widely separated spectrum near $\sigma$, one straightforward way is to convert the matrix to the desired substracted form ($\sigma \textbf{I}$ - \textbf{H}) and solve an inverted form of it (\textbf{($\sigma$I - H)}$^{-1}$).

F(\textbf{H})\textbf{v} is calculated by iteratively solving the linear system ($\sigma$\textbf{I}-\textbf{H})\textbf{w} = \textbf{v}. These vectors \textbf{w$_1$}, \textbf{w$_2$} etc. can be calculated approximately i.e., in this way, the iterative solver becomes less computationally expensive\cite{huang2000new}.
The eigenvalue problem in \textbf{w} basis is then solved to obtain eigenvalues and corresponding eigenvectors.

Another way of targeting eigenvalue $\sigma$ is through contour integration.
Rather than targeting a specific eigenvalue, it is aimed to find the eigenvalue within a specified range or the contour.
For example, $$Hx=\lambda Bx$$ is the problem to solve with $H$ is the real symmetric or Hermitian matrix and $B$ is positive semi-definite.
Then in FEAST [2] the eigenvalues within contour [$\lambda\_{min},\lambda\_{max}$] is iterativey solved.

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
A. Inexact Lanczos eigensolver
1. A 		 : 	NUMPY.NDARRAY/ LinearOperator
			Diagonalizable matrix
2. n (optional)  : 	INT
       			Dimension of diagonalizable matrix, 
			if A is not provided
3. ev (optional) :	NUMPY.NDARRAY
			Test eigenvalues, 
			if A is not provided
4. target        : 	FLOAT
			Eigenvalue target, $sigma$
3. maxit	 :	INT
			Number of Lanczos iterations
4. eConv	 :	FLOAT
			Relative eigenvalue convergence
5. optionsLinear 
(optional)       :	DICT
			Dictionary with linear solver arguments
6. status (optional)
		 :	DICT
			Print instructions

# Output files
After successful test and running of example driver file there will following output files:
1. iterations_lanczos.out (Detailed information at each cumulative iteration. Information include
parameter details in file header, overlap matrix, Hamiltonian matrix before and after diagonalization, eigenvalues. )
2. summary_lanczos.out (Summary information at each cumulative iteration along with parameter details in file header.)

# Contributors
1. Dr. Madhumita Rano
2. Prof. Henrik R. Larsson (https://github.com/hrlGroup)

# References
1. Shi-Wei Huang and Tucker Carrington Jr. “A new iterative method for calculating energy levels and
wave functions”. In: The Journal of Chemical Physics 112.20 (2000), pp. 8765–8771.
2. Eric Polizzi. “Density-matrix-based algorithm for solving eigenvalue problems”. In: Physical Review B
79.11 (2009), p. 115112.
