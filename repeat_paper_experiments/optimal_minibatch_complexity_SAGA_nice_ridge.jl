using JLD
using Plots
using StatsBase
using Match
using Combinatorics
include("../src/StochOpt.jl") # Be carefull about the path here


### LOADING DATA ###
probname = "australian"; # "artificial" for artificaly generated data

# If probname="artificial", precise the number of features and data
srand(1234) # fixing the seed
numfeatures = 20;
numdata = 50;

println("--- Loading data ---");
probnames = ["australian", "covtype"]; #, "letter_scale", "heart", "phishing", "madelon", "a9a", "mushrooms", "phishing", "w8a", "gisette_scale", 
if(probname == "artificial")
    ## Load artificial data
    X, y, probname = gen_gauss_data(numfeatures, numdata, lambda=0.0);
elseif(probname in probnames)
    ## Load truncated LIBSVM data
    X, y = loadDataset(probname);
    X = X';
    numfeatures = size(X)[1];
    numdata = size(X)[2];
    y = convert(Array{Float64}, 1:1:size(X)[2]) ; # 14 data taken from X instead of 690
else
    error("unkown problem name.");
end

println(numfeatures)
println(numdata)

### SETTING UP THE PROBLEM ###
println("\n--- Setting up the ridge regression problem ---");
options = set_options(max_iter=10^8, max_time=0.02, max_epocs=3000, repeat_stepsize_calculation=true, skip_error_calculation=51);
prob = load_ridge_regression(X, y, probname, options, lambda=-1);  # Loads logisitc problem


### CREATING THE MINI-BATCH SIZE SEQUENCE ###
# tauseq = collect(1:14);
# tauseq = [1, 2, 3, 4, 5, 14];
# tauseq = cat(1, 1:4, 14);
# tauseq = [1, 2, 3, 3, 3, 4, 5, 5, 14]; # test uniqueness
# tauseq = [1, 5, 4, 3, 2, 14]; # test sorting

# tauseq = [1, 2, 3, numdata];
tauseq = [1, 2, 3, 4, 5];

# Sanity checks
tauseq = unique(sort(tauseq));
n = prob.numdata;
numtau = length(tauseq);
if(minimum(tauseq) < 1 || maximum(tauseq) > n)
    error("values of tauseq are out of range.");
end
println("\n--- Mini-batch sequence ---");
println(tauseq);


### COMPUTE SAGA-NICE COMPLEXITIES ###
println("\n--- Compute SAGA-nice complexities (iteration and total) ---");
default_path = "./data/"; savename = replace(replace(prob.name, r"[\/]", "-"), ".", "_");
savenamecomp = string(savename,"-complexities-nidham");
itercomp = 0.0; Lsides = 0.0; Rsides = 0.0;
try
    itercomp, Lsides, Rsides = load("$(default_path)$(savenamecomp).jld", "itercomp", "Lsides", "Rsides");
    println("found ", "$(default_path)$(savenamecomp).jld with itercomp\n", itercomp);
catch loaderror   # Calculate iteration complexity for all minibatchsizes
    println(loaderror);
    itercomp, Lsides, Rsides = calculate_complex_SAGA_nice(prob, options, tauseq);
    # L = eigmax(prob.X*prob.X')/prob.numdata+prob.lambda;
    # save("$(default_path)$(savenamecomp).jld", "itercomp", itercomp, "Lsides", Lsides, "Rsides", Rsides);
end

println("Mini-batch size sequence:\n", tauseq)

## Total complexity equals the iteration complexity times the size of the batch
# totcomp = (itercomp').*(1:prob.numdata);
totcomp = (itercomp').*tauseq;

### PLOTING ###
println("\n--- Ploting complexities ??? ---");
pyplot() # pyplot
fontsmll = 8; fontmed = 14; fontbig = 14;
plot(tauseq, [totcomp itercomp'], label=["total complex" "iter complex"],
    linestyle=:auto, xlabel="batchsize", ylabel="complexity", tickfont=font(fontsmll), guidefont=font(fontbig), legendfont=font(fontmed),
    markersize=6, linewidth=4, marker=:auto, grid=false, ylim=(0, maximum(totcomp)+minimum(itercomp)), xticks=tauseq)
    minimum(itercomp)
# savefig("./figures/$(savenamecomp).pdf");

# Comparing only the iteration complexities
plot(tauseq, Lsides', ylabel="expected smoothness", xlabel="batchsize", tickfont=font(fontsmll), guidefont=font(fontbig),
    markersize=6, linewidth=4, marker=:auto, grid=false, legend=false, xticks=tauseq)
savenameexpsmooth = string(savenamecomp, "-expsmooth");
# savefig("./figures/$(savenameexpsmooth).pdf");