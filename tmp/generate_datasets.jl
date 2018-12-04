srand(1234);

using JLD
using Plots
using StatsBase
using Match
using Combinatorics

include("./src/StochOpt.jl") # Be carefull about the path here


default_path = "./data/";
seed = string(reinterpret(Int32, Base.Random.GLOBAL_RNG.seed[1]));
seed = string("_seed-", seed);

### LOADING DATA ###
data = "gaussian";
# data = "diagonal";
# data = "alone_eig_val";

numdata = 100;
numfeatures = 12; # useless for gen_diag_data

if(data == "gaussian")
    X, y, probname = gen_gauss_data(numfeatures, numdata, lambda=0.0, err=0.001);
elseif(data == "diagonal")
    X, y, probname = gen_diag_data(numdata, lambda=0.0, Lmax=100);
elseif(data == "lone_eig_val")
    X, y, probname = gen_diag_lone_eig_data(numfeatures, numdata, lambda=0.0, a=100, err=0.001);
else
    error("unkown generation scheme.");
end

## Rotate diagonal matrices


## Saving the generated datasets with the corresponding seed
probname = string(probname, seed);
# save("$(default_path)$(probname).jld", "X", X, "y", y);
