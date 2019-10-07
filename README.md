**warning package under development:** it will break. But renewed shall be the code that was broken, the crashless again shall compile.

# Download the datasets
```bash
bash download_datasets
```
If it crashes, data can be downloaded manually [here](https://partage.mines-telecom.fr/index.php/s/9MreP5y6evFWyJP). Then, datasets `.jld` files should be placed in the directory `./data`.


# Dependencies
Launch the  Julia REPL and press `]` to enter the package REPL mode, then install the required packages with
```bash
(v1.0) pkg> add JLD
(v1.0) pkg> add Plots
(v1.0) pkg> add StatsBase
(v1.0) pkg> add Match
(v1.0) pkg> add Combinatorics
(v1.0) pkg> add Formatting
(v1.0) pkg> add LaTeXStrings
(v1.0) pkg> add PyPlot
(v1.0) pkg> add Distributions
...
(v1.0) pkg> add Distributed
```
if any problem with PyPlot, try the following manipulation
```bash
$ julia
julia> ENV["PYTHON"]=""
julia> ]
(v1.0) pkg> build PyCall
```

# StochOpt
A suite of stochastic optimization methods for solving the empirical risk minimization problem. <br>


# Demo
For a simple demo of the use of the package
```julia
julia ./test/demo.jl
```

For a demo of the SVRG2 type methods from [1]
```julia
julia ./test/test/demo_SVRG2.jl
```
For a demo of the BFGS and accelerated BFGS methods from [2]
```julia
julia ./test/demo_BFGS.jl
```

For a demo of SAGA with optimized probabilities from [4]
```julia
julia ./test/demo_SAGA.jl
```

For a demo of SAGA nice from [5]
```julia
julia ./test/demo_SAGA_nice.jl
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
julia ./repeat_paper_experiments/compare_SAGA_importance_opt_Lis.jl
```

To re-generate the experiments from Section 6.1 of [4]
```julia
julia ./repeat_paper_experiments/test_optimal_minibatch_SAGA_nice.jl
```

To re-generate the experiments from Section 5.1 & 5.2 of [5] (~1h 30min)
```julia
julia -p <number_of_processors_to_add> ./repeat_paper_experiments/repeat_optimal_minibatch_step_sizes_SAGA_paper_experiment_1_and_2_parallel.jl all_problems
```
setting `all_problems` to `false` to run the code only on the first problem, unscaled *uniform* synthetic dataset with $\lambda =10^{-1}$, (~XXXXXmin) or to `true` to run it on all of them (~XXXXXXXXh XXXXXmin).

To re-generate experiments from Section 5.3 of [5]
```julia
julia -p <number_of_processors_to_add> ./repeat_paper_experiments/repeat_optimal_minibatch_step_sizes_SAGA_paper_experiment_3_parallel.jl all_problems
```
setting `all_problems` to `false` to run the code only on the first problem, scaled *ijcnn1_full* with $\lambda =10^{-1}$, (~1min) or to `true` to run it on all of them (~1h 30min).

To re-generate the experiments from Section 5.4 of [5]
```julia
julia -p <number_of_processors_to_add> ./repeat_paper_experiments/repeat_optimal_minibatch_step_sizes_SAGA_paper_experiment_4_parallel.jl all_problems
```
setting `all_problems` to `false` to run the code only on the first problem, scaled *ijcnn1_full* with $\lambda =10^{-1}$, (~2min) or to `true` to run it on all of them (~XXh XXmin).


# Methods implemented
SVRG, the original SVRG algorithm. <br>
SVRG2, which tracks the gradients using the full Hessian. <br>
2D, which tracks the gradients using the diagonal of the Hessian. <br>
2Dsec, which tracks the gradients using the robust secant equation. <br>
SVRG2emb, which tracks the gradients using a low-rank approximation of the Hessians. <br>
CM, which tracks the gradients using the low-rank curvature matching approximation of the Hessian. <br>
AM, which uses the low-rank action matching approximation of the Hessian. <br>
BFGS, the standard, full memory BFGS method. <br>
BFGS_accel, an accelerated BFGS method. <br>
SAGA, stochastic average gradient descent, with several options of samplings (including optimal probabilities). <br>
SAGA nice, mini-batch version of SAGA with nice sampling. <br>

More details on the methods can be found in [1], [2], [4] and [5] <br>


# Code Philosophy
To provide en environment where competing stochastic methods can be compared on equal footing. This is why all methods are called by the same wrapper function "minimizeFunc" (or it's extension minimizeFunc_grid_stepsize). All performance measures such as time taken, test error or epochs are calculated by these wrapper functions. Each new method need only supply a *stepmethod* and a *bootmethod*. The stepmethod returns an update vector d which is then added to x_k to give the next iterate x_{k+1}. The bootmethod is called once to initialize the method.


# Adding more data
To test a new data set, download the raw data of a binary classification from LIBSVM [3] and place it in the folder ./data.
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
Robert M. Gower, Nicolas Le Roux and Francis Bach.
To appear in AISTATS 2018

[2] *Accelerated stochastic matrix inversion: general theory and speeding up BFGS rules for faster second-order optimization* <br>
Robert M. Gower, Filip Hanzely, P. Richtárik and S. Stich.
arXiv:1801.05490, 2018

[3]  *LIBSVM : a library for support vector machines.* <br>
Chih-Chung Chang and Chih-Jen Lin, ACM Transactions on Intelligent Systems and Technology, 2:27:1--27:27, 2011. <bf>
  Software available at http://www.csie.ntu.edu.tw/~cjlin/libsvm

[4] *Stochastic Quasi-Gradient Methods:
Variance Reduction via Jacobian Sketching* <br>
Robert M. Gower, Peter Richtárik, Francis Bach

[5] *Optimal mini-batch and step sizes for SAGA* <br>
Nidham Gazagnadou, Robert M. Gower and Joseph Salmon.
arXiv:1902.00071, 2019

[6] *Towards closing the gap between the theory and practice of SVRG* <br>
Othmane Sebbouh, Nidham Gazagnadou, Samy Jelassi, Francis Bach, Robert M. Gower. arXiv:1908.02725, 2019

For up-to-date references see https://perso.telecom-paristech.fr/rgower/publications.html


# TODO
* change output type to have a testerrors field. Think where best to load a test problem. Probably outside of minimizeFunc. Where best to place code for test_error ?  Probably best to start a new src file for error calculations? or testing related things?
* Implement the calculation of the Jacobian.
* The code for SVRG2 type methods (AMprev, AMgauss, CMprev, CMgauss) should have their own type. Right now they are definied using the generic Method type, which is why the code for these functions is illegible.
