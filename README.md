**warning** package under development: it will break. But renewed shall be the code that was broke, the crashless again shall compile.

# Dependencies

```
Pkg.add("JLD")
Pkg.add("Plots")
Pkg.add("StatsBase")
Pkg.add("Match")
```

# StochOpt
A suite of stochastic optimization methods for solving the empirical risk minimization problem.  <br>


# Demo
Run   ```julia ./test/demo.jl``` for a simple demo of the use of the package.

Run
```julia ./test/test/demo_SVRG2.jl```
for a demo of methods from [1]

Run
```julia ./test/demo_BFGS.jl```
for a demo of methods from [2]

Run
```julia ./test/demo_BFGS_accel_paper_results.jl```
to re-generate the experiments from [2]

# Methods currently implemented 

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


# Adding more data
To test a new data set, download the raw data of a binary classifiction fomr LIBSVM [3] and place it in the folder ./data. 
Then replace "liver-disorders" in the code *src/load_new_LIBSVM_data.jl* and execute. In other words, run the code 

```
include("dataLoad.jl")
initDetails()

datasets = ["liver-disorders"]  
for  dataset in datasets
transformDataJLD(dataset)
X,y = loadDataset(dataset) 
showDetails(dataset)
end
```
where "liver-disorders" has been replaced with the name of the new raw data file.

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

[3]  *LIBSVM : a library for support vector machines.* <br>
Chih-Chung Chang and Chih-Jen Lin, ACM Transactions on Intelligent Systems and Technology, 2:27:1--27:27, 2011. <bf>
  Software available at http://www.csie.ntu.edu.tw/~cjlin/libsvm 

For update to date references, see https://perso.telecom-paristech.fr/rgower/publications.html

# TODO
* The option "exacterror" is obsolete for now since minimizeFunc runs assuming there is and prob.fsol.
* Change organization of boot methods. Allow user to directly call boot function, with named parameters. Create different types for each method, as oppose to re-using one type with a cacophony of bizarre fields. 
