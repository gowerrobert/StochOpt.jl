# HUGE PROBLEM IN THIS: we generate data before checking if a corresponding file already exists.
println("--- Loading packages ---")
using JLD
using Plots
using StatsBase
using Match
using Combinatorics
using Random
using Printf # julia 0.7
using LinearAlgebra # julia 0.7

include("./src/StochOpt.jl") # Be carefull about the path here


println("--- Getting seed ---")
Random.seed!(1);
seed = string(reinterpret(Int32, Random.GLOBAL_RNG.seed[1]));
seed = string("_seed-", seed);

### LOADING DATA ###
# data = "gaussian";
data = "diagonal";
# data = "alone_eig_val";

numdata = 10;
numfeatures = 50; # useless for gen_diag_*

rotate = true; # keeping same eigenvalues, but removing the diagonal structure of X

println("--- Generating data ---")
if(data == "gaussian")
    X, y, probname = gen_gauss_data(numfeatures, numdata, lambda=0.0, err=0.001);
elseif(data == "diagonal")
    X, y, probname = gen_diag_data(numdata, lambda=0.0, Lmax=100, rotate=rotate);
elseif(data == "alone_eig_val")
    X, y, probname = gen_diag_alone_eig_data(numfeatures, numdata, lambda=0.0, a=100, err=0.001, rotate=rotate);
else
    error("unkown generation scheme.");
end

## Saving the generated datasets with the corresponding seed
println("--- Saving the data set ---")
probname = string(probname, seed);
default_path = "./data/";

try
    dataset = load("$(default_path)$(probname).jld");
    println("The following data set already exists in: $(default_path)$(probname).jld");
    lines = readlines("$(default_path)available_datasets.txt");
    if !("$(probname)" in lines)
        open("$(default_path)available_datasets.txt", "a") do file
            write(file, "$(probname)\n");
        end
    end
catch loaderror
    println(loaderror);
    println("Saving the new datat set: $(default_path)$(probname).jld");
    save("$(default_path)$(probname).jld", "X", X, "y", y);
    lines = readlines("$(default_path)available_datasets.txt");
    if !("$(probname)" in lines)
        open("$(default_path)available_datasets.txt", "a") do file
            write(file, "$(probname)\n");
        end
    end
end


### DRAFT ###
## Managaing text files with julia ##
# lines = readlines("$(default_path)available_datasets.txt");
# lines
# !("a" in lines)
# "d" in lines

# open("$(default_path)available_datasets.txt", "r") do file
#     lines = readlines(file);
# end



# open("$(default_path)available_datasets.txt") do file
#     for ln in eachline(file)
#         println("$(length(ln)), $(ln)")
#     end
# end


# open("$(default_path)available_datasets.txt", "r") do file
#     for ln in eachline(file)
#         println("$(length(ln)), $(ln)")
#     end
# end