# Dependencies

Pkg.add("JLD") <br>
Pkg.add("Plots")  <br>
Pkg.add("StatsBase")  <br>
Pkg.add("Match")  <br>

# StochOpt
A suite of stochastic optimization methods for solving the empirical risk minimization problem.  <br>

The methods currently implemented include <br>

SVRG, the original SVRG algorithm; <br>
SVRG2, which tracks the gradients using the full Hessian. <br>
2D, which tracks the gradients using the diagonal of the Hessian. <br>
2Dsec, which tracks the gradients using the robust secant equation. <br>
SVRG2emb, which tracks the gradients using a low-rank approximation of the Hessians. <br>
CM, which tracks the gradients using the low-rank curvature matching approximation of the Hessian <br>
AM, which uses the low-rank action matching approximation of the Hessian. <br>

More details on the methods can be found in <br>

``Tracking the gradients using the Hessian: A new look at variance reducing stochastic methods'',
  Robert M. Gower, Nicolas Le Roux and Francis Bach, AISTATS 2018.

# Demo
Try  
./test/demo.jl
for a demo of the use of the package. 

# Adding more data
The package is setup so that it is easily extendable. For instance:
 
=> to test new data, download the raw data of a binary classification problem from LIBSVM and place it in the folder ./data. Then change the variable probname in demo.jl from "phishing" to the name of the newly downloaded file. 


# Adding new loss functions
to include new objective function, see load_problem.jl

# Adding new methods
to include new method X, you need to write a descent_X.jl and boot_X.jl function. See descent_grad and boot_grad for an example
 
