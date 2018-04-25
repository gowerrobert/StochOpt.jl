**warning package under development:** it will break. But renewed shall be the code that was broken, the crashless again shall compile.

# Dependencies

```julia
Pkg.add("JLD")
Pkg.add("Plots")
Pkg.add("StatsBase")
Pkg.add("Match")
```

# StochOpt
A suite of stochastic optimization methods for solving the empirical risk minimization problem.  <br>


# Demo
For a simple demo of the use of the package
```julia
julia ./test/demo.jl
```

For a demo of the methods from [1]
```julia
julia ./test/test/demo_SVRG2.jl
```
For a demo of the methods from [2]
```julia
julia ./test/demo_BFGS.jl
```

# Repeating paper results

To re-generate all of the experiments from [1]
```julia
julia ./repeat_paper_experiments/repeat_SVRG2_paper_experiments.jl
```

To re-generate all of the experiments from [2]
```julia
julia ./repeat_paper_experiments/repeat_BFGS_accel_paper_results.jl
```


To re-generate the experiments from Section 6.1 of [4]
```julia
julia ./repeat_paper_experiments/test_optimal_minibatch_SAGA_nice.jl
```



# Methods implemented

SVRG, the original SVRG algorithm; <br>
SVRG2, which tracks the gradients using the full Hessian. <br>
2D, which tracks the gradients using the diagonal of the Hessian. <br>
2Dsec, which tracks the gradients using the robust secant equation. <br>
SVRG2emb, which tracks the gradients using a low-rank approximation of the Hessians. <br>
CM, which tracks the gradients using the low-rank curvature matching approximation of the Hessian <br>
AM, which uses the low-rank action matching approximation of the Hessian. <br>
BFGS, the standard, full memory BFGS method. <br>
BFGS_accel, an accelerated BFGS method. <br>
SAGA, stochastic average gradient descent, with several options of samplings (including optimal probabilities) <br>

More details on the methods can be found in [1] and [2] <br>


# Adding more data
To test a new data set, download the raw data of a binary classifiction fomr LIBSVM [3] and place it in the folder ./data.
Then replace "liver-disorders" in the code *src/load_new_LIBSVM_data.jl* and execute. In other words, run the code

```julia
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
to include new objective function, see load_logistic.jl and copy the same structure

# Adding new methods
to include a new method X, you need to write a descent_X.jl and boot_X.jl function. See descent_grad and boot_grad for an example. I also recommend writing your type and including it in StochOpt or using one of the types there defined already.

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

[4] *Stochastic Quasi-Gradient Methods:
Variance Reduction via Jacobian Sketching* <br>
RMG, Peter Richtárik, Francis Bach


For up-to-date references see https://perso.telecom-paristech.fr/rgower/publications.html

# TODO
* The option "exacterror" is obsolete for now since minimizeFunc runs assuming that prob.fsol has been save and calculated.
* Implement the calculation of the Jacobian.
* The code for SVRG2 type methods (DFPprev, DFPgauss, CMprev, CMgauss) should have their own type. Right now they are definied using the generic Method type, which is why the code for these functions is illegible.
* Change organization of how methods are booted. Allow user to directly call initiation function, with named parameters inputs. Create different types for each method, stop re-using one generic Method type with a cacophony of bizarre unintelligible fields.
