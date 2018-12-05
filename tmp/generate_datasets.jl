println("--- Loading packages ---")
using JLD
using Plots
using StatsBase
using Match
using Combinatorics
using Random
using Printf # julia 0.7
using LinearAlgebra # julia 0.7

include("../src/StochOpt.jl") # Be carefull about the path here


println("--- Getting seed ---")
Random.seed!(1234);
seed = string(reinterpret(Int32, Random.GLOBAL_RNG.seed[1]));
seed = string("_seed-", seed);

### LOADING DATA ###
# data = "gaussian";
# data = "diagonal";
data = "alone_eig_val";

numdata = 5;
numfeatures = 3; # useless for gen_diag_data

println("--- Generating data ---")
if(data == "gaussian")
    X, y, probname = gen_gauss_data(numfeatures, numdata, lambda=0.0, err=0.001);
elseif(data == "diagonal")
    X, y, probname = gen_diag_data(numdata, lambda=0.0, Lmax=100);
elseif(data == "alone_eig_val")
    X, y, probname = gen_diag_alone_eig_data(numfeatures, numdata, lambda=0.0, a=100, err=0.001);
else
    error("unkown generation scheme.");
end
println(X,y)

## Rotate diagonal matrices


## Saving the generated datasets with the corresponding seed
println("--- Saving the data set ---")
println("Saved in ", "$(default_path)$(probname).jld")
probname = string(probname, seed);
default_path = "./data/";
save("$(default_path)$(probname).jld", "X", X, "y", y);
