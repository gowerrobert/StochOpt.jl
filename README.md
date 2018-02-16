# Dependencies

```Pkg.add("JLD")
Pkg.add("Plots")
Pkg.add("StatsBase")
Pkg.add("Match")
```

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
BFGS, the standard, full memory BFGS method. <br>
BFGS_accel, an accelerated BFGS method. <br>


More details on the methods can be found in [1] and [2] <br>


# Demo
Run   ```./test/demo.jl``` for a simple demo of the use of the package.

Run
```./test/test/demo_SVRG2.jl```
for a demo of methods from [1]

Run
```./test/demo_BFGS_accel_paper_results.jl```
for a demo of methods from [2]

# Adding more data
The package is setup so that it is easily extendable. For instance:

=> to test new data, download the raw data of a binary classification problem from LIBSVM and place it in the folder ./data. Then change the variable probname in demo.jl from "phishing" to the name of the newly downloaded file.


# Adding new loss functions
to include new objective function, see load_problem.jl

# Adding new methods
to include new method X, you need to write a descent_X.jl and boot_X.jl function. See descent_grad and boot_grad for an example

# References

[1]  *Tracking the gradients using the Hessian: A new look at variance reducing stochastic methods* <br>
RMG, Nicolas Le Roux and Francis Bach.
To appear in AISTATS 2018 

[2] *Accelerated stochastic matrix inversion: general theory and speeding up BFGS rules for faster second-order optimization* <br>
RMG, Filip Hanzely, P. Richtárik and S. Stich.
arXiv:1801.05490, 2018 

For updated reference, see https://perso.telecom-paristech.fr/rgower/publications.html

# TODO
* The option "exacterror" is obsolete for now since minimizeFunc runs assuming there is and prob.fsol.
* Change organization of boot methods. Allow user to directly call boot function, with named parameters. Create different types for each method, as oppose to re-using one type with a cacophony of bizarre fields. 
