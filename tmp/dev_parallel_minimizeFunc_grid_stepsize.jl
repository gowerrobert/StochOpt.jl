using Distributed

addprocs(3)

@everywhere begin # this part will be available on all CPUs
    using JLD
    using Plots
    using StatsBase
    using Match
    using Combinatorics
    using Random
    using Printf
    using LinearAlgebra
    using Statistics
    using Base64
    
    using SharedArrays

    include("./src/StochOpt.jl") # Be carefull about the path here
end

########################################### Exploring parallelization ###########################################
#region
# @everywhere include("/home/nidham/phd/moving2julia7/StochOpt.jl/tmp/count_heads.jl")
# a = @spawn count_heads(100000000)
# println("a: ", a)
# b = @spawn count_heads(100000000)
# println("b: ", b)
# println("Sum: ", fetch(a)+fetch(b))

# nheads = @distributed (+) for i = 1:20
#     println("The i of this iteration in $i")
#     Int(rand(Bool))
# end
# println("@distributed loop, number of heads: ", nheads)

# @everywhere using SharedArrays

# a = SharedArray{Float64}(10)
# println("Before the @distributed loop, a: ", a)
# @sync @distributed for i = 1:10
#     a[i] = i
#     # println(a[i])
# end

# println("After the @distributed loop, a:  ", fetch(a))

## Compute the singular values of several large random matrices in parallel
# using LinearAlgebra
# M = Matrix{Float64}[rand(1000,1000) for i = 1:10];
# pmap(svdvals, M)

#endregion
#################################################################################################################


############################################## Parallel grid search #############################################
Nprocs = nprocs()
print("Number of processors ", Nprocs,".\n")

### LOADING DATA ###
println("--- Loading data ---");
datasets = readlines("./data/available_datasets.txt");
# idx = 4; # australian
idx = 14; # news20.binary
data = datasets[idx];

@time X, y = loadDataset(data);

# varinfo(r"(X|y|prob)")

### SETTING UP THE PROBLEM ###
println("\n--- Setting up the selected problem ---");
options = set_options(tol=10.0^(-1), max_iter=10^8, max_time=10.0^2, max_epocs=10^8, 
                      regularizor_parameter = "normalized", initial_point="zeros", force_continue=false);

@time prob = load_logistic_from_matrices(X, y, data, options, lambda=-1, scaling="none");

options = set_options(tol=10.0^(-16.0), skip_error_calculation=10^2, exacterror=false, max_iter=10^8, 
                      max_time=60.0*20.0, max_epocs=500, repeat_stepsize_calculation=true, rep_number=2);
options.batchsize = 100;
method_input = "SVRG";

@everywhere include("./tmp/parallel_minimizeFunc_grid_stepsize.jl")

# parallel_toy_grid_search(prob, method_input, options);

@time output = parallel_minimizeFunc_grid_stepsize(prob, method_input, options);

# minimizeFunc_grid_stepsize(prob, method_input, options);

#################################################################################################################
