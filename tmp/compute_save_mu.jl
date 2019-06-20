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
using Formatting
using SharedArrays

# path = "./"
# include("./src/StochOpt.jl")
path = "/cal/homes/ngazagnadou/StochOpt.jl/"
include("$(path)src/StochOpt.jl")

problems = [3, 4, 7, 8, 9, 10] # Only ridge problems

datasets = ["ijcnn1_full", "ijcnn1_full",                       # scaled,   n = 141,691, d =     22
            "YearPredictionMSD_full", "YearPredictionMSD_full", # scaled,   n = 515,345, d =     90
            "covtype_binary", "covtype_binary",                 # scaled,   n = 581,012, d =     54
            "slice", "slice",                                   # scaled,   n =  53,500, d =    384
            "slice", "slice",                                   # unscaled, n =  53,500, d =    384
            "real-sim", "real-sim"];                            # unscaled, n =  72,309, d = 20,958

scalings = ["column-scaling", "column-scaling",
            "column-scaling", "column-scaling",
            "column-scaling", "column-scaling",
            "column-scaling", "column-scaling",
            "none", "none",
            "none", "none"];

lambdas = [10^(-1), 10^(-3),
           10^(-1), 10^(-3),
           10^(-1), 10^(-3),
           10^(-1), 10^(-3),
           10^(-1), 10^(-3),
           10^(-1), 10^(-3)];

## In the following table, set smaller values for finer estimations (yet, longer simulations)
skip_multipliers = [0.01,        # 2min 20s avec 0.01
                    0.01,        # 4 min avec 0.01
                    0.01,        # 11 min avec 0.01
                    0.01,        # 10 min avec 0.01
                    0.05,        # 36 min avec 0.05 / 1h 30min avec 0.01 / 5h30 avec 0.001
                    1.0,         # 1h 47min avec 1.0 / 2h 36min avec 0.1
                    0.1,         # 22 min avec 1.0 / 43 min avec 0.1
                    1.0,         # 52 min avec 1.0
                    0.1,         # 12 min avec 1.0 / 21 min avec 0.1
                    1.0,         # plus de 7h (max_time reached for b=2^14) avec 1.0 / 6h30 avec 10.0 (pas assez précis)
                    0.1,         # 1h 27min avec 1.0 (pas assez précis) / 2h 12 min avec 0.1
                    0.1];        # 6h 15min avec 1.0 / more than 6h avec 0.1

precision = 10.0^(-4)

for idx_prob in problems
# idx_prob = 1
    @time begin
        data = datasets[idx_prob];
        scaling = scalings[idx_prob];
        lambda = lambdas[idx_prob];
        println("EXPERIMENT : ", idx_prob, " over ", length(problems));
        @printf "Inputs: %s + %s + %1.1e \n" data scaling lambda;

        Random.seed!(1);

        ## Loading the data
        println("--- Loading data ---");
        data_path = "$(path)data/";
        X, y = loadDataset(data_path, data);

        ## Setting up the problem
        println("\n--- Setting up the selected problem ---");
        options = set_options(tol=precision, max_iter=10^8, max_epocs=10^8,
                              max_time=60.0*60.0*10.0,
                              skip_error_calculation=10^4,
                              batchsize=1,
                              regularizor_parameter = "normalized",
                              initial_point="zeros", # is fixed not to add more randomness
                              force_continue=false); # force continue if diverging or if tolerance reached
        u = unique(y);
        if length(u) < 2
            error("Wrong number of possible outputs");
        elseif length(u) == 2
            println("Binary output detected: the problem is set to logistic regression")
            prob = load_logistic_from_matrices(X, y, data, options, lambda=lambda, scaling=scaling);
        else
            println("More than three modalities in the outputs: the problem is set to ridge regression")
            prob = load_ridge_regression(X, y, data, options, lambda=lambda, scaling=scaling);
        end

        X = nothing;
        y = nothing;

        n = prob.numdata;
        d = prob.numfeatures;
        mu = prob.mu;
        L = prob.L;

        ## Save strong-convexity parameter
        mufilename = get_mu_filename(prob);
        save("$(mufilename)_new.jld", "mu", mu)

        ## Computing mini-batch and step sizes
        b_practical = round(Int, 1 + (mu*(n-1))/(4*L)) # ERROR for pb 11? b_practical cannot be lager than ~ n/4 ?

        @printf "\nL = %e and mu = %e\n" L mu
        @printf "Condition number = %e\n" L/mu
        println("Practical optimal mini-batch = ", b_practical, "\n")
    end
end

