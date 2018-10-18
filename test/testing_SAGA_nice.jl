using JLD
using Plots
using StatsBase
using Match
using Combinatorics

include("./src/StochOpt.jl") # Be carefull about the path here
# srand(1234) # fixing the seed


### LOADING DATA ###
probname = "abalone"; # libsvm regression dataset | "gaussian", "diagonal" or "lone_eig_val" for artificaly generated data

# If probname="artificial", precise the number of features and data
numdata = 24;
numfeatures = 12; # useless for gen_diag_data

println("--- Loading data ---");
probnames = ["abalone", "housing"];
#, "letter_scale", "heart", "phishing", "madelon", "a9a",
# "mushrooms", "phishing", "w8a", "gisette_scale",
if(probname == "gaussian")
    ## Load artificial data
    X, y, probname = gen_gauss_data(numfeatures, numdata, lambda=0.0, err=0.001);
elseif(probname == "diagonal")
    X, y, probname = gen_diag_data(numdata, lambda=0.0, Lmax=100);
elseif(probname == "lone_eig_val")
    X, y, probname = gen_diag_lone_eig_data(numfeatures, numdata, lambda=0.0, a=100, err=0.001);
elseif(probname in probnames)
    ## Load truncated LIBSVM data
    X, y = loadDataset(probname);
    # X = X';
    # numfeatures = size(X)[1];
    # numdata = size(X)[2];
    # y = convert(Array{Float64}, 1:1:size(X)[2]) ; # 14 data taken from X instead of 690
else
    error("unkown problem name.");
end

### SETTING UP THE PROBLEM ###
println("\n--- Setting up the ridge regression problem ---");
options = set_options(max_iter=10^8, max_time=10.0, max_epocs=1000, repeat_stepsize_calculation=true, skip_error_calculation=51,
                      force_continue=true, initial_point="randn");
prob = load_ridge_regression(X, y, probname, options, lambda=-1, scaling="none");  # Disabling scaling
# QUESTION: how is lambda selected?
n = prob.numdata;
d = prob.numfeatures;


### COMPUTING THE SMOOTHNESS CONSTANTS ###
# Compute the smoothness constants L, L_max, \cL, \bar{L}
datathreshold = 24;
if(n > datathreshold) # if n is too large we do not compute the exact expected smoothness constant nor its relative quantities
    println("The number of data is to large to compute the expected smoothness constant exactly");
    computeexpsmooth = false;
else
    computeexpsmooth = true;
end

mu = get_mu_str_conv(prob);
L = get_LC(prob, collect(1:n));
Li_s = get_Li(prob);
Lmax = maximum(Li_s);
Lbar = mean(Li_s);

######################################## EMPIRICAL OPTIMAL MINIBATCH SIZE ########################################

default_path = "./data/"; savename = replace(replace(prob.name, r"[\/]", "-"), ".", "_");
savenamecomp = string(savename);
fontsmll = 8; fontmed = 14; fontbig = 14;


## Select the list of mini-batch sizes 
# taulist = [1, 5];
# taulist = [5, 1];
# taulist = 5:-1:1;
taulist = [5];

srand(1234);

tic();
numsimu = 1; # number of runs of mini-batch SAGA for averaging the empirical complexity
tolerance = 10.0^(-2); # epsilon for which: (f(x)-fsol)/(f0-fsol) < epsilon
options = set_options(tol=tolerance, max_iter=10^8, max_time=10000.0, max_epocs=30,
                      initial_point="zeros", # fix initial point to zeros for a maybe fairer comparison? -> YES
                #   repeat_stepsize_calculation=true,
                      skip_error_calculation=1, # What is this option?
                      force_continue=false); # force continue if diverging or if tolerance reached
itercomplex = zeros(length(taulist), 1); # List of saved outputs
OUTPUTS = [];
for idxtau in 1:length(taulist) # 1:n
    tau = taulist[idxtau];
    println("\nCurrent mini-batch size: ", tau);
    options.batchsize = tau;
    for i=1:numsimu
        sg = initiate_SAGA(prob, options, minibatch_type="nice");
        println("STEPSIZE OF sg: ", sg.stepsize);
        output = minimizeFunc(prob, sg, options);
        itercomplex[idxtau] += output.iterations;
        OUTPUTS = [OUTPUTS; output];
    end
end
fails = [OUTPUTS[i].fail for i=1:length(taulist)];
itercomplex = itercomplex ./ numsimu;
toc();


## Plotting the average behaviour of each mini-batch size
rel_loss_avg = [];
for i=1:length(taulist)
    rel_loss_array = [];
    for j=1:numsimu
        # println("idx:", (i-1)*numsimu+j);
        # println(OUTPUTS[(i-1)*numsimu+j].fs[1]);
        rel_loss_array = [rel_loss_array; [(OUTPUTS[(i-1)*numsimu+j].fs'.-prob.fsol)./(OUTPUTS[(i-1)*numsimu+j].fs[1].-prob.fsol)]];
    end

    maxlength = maximum([length(rel_loss_array[j]) for j=1:numsimu]);
    tmp = similar(rel_loss_array[1], maxlength, 0);
    for j=1:numsimu
        # resize vector Maybe 0 or NA instead of tolerance
        tmp = hcat(tmp, vcat(rel_loss_array[j], fill(tolerance, maxlength-length(rel_loss_array[j]), 1)));
    end
    tmp = mean(tmp, 2);
    rel_loss_avg = [rel_loss_avg; [tmp]];
end

output = OUTPUTS[1];
epocsperiters = [OUTPUTS[i].epocsperiter for i=1:numsimu:length(OUTPUTS)];
lfs = [length(rel_loss_avg[i]) for i=1:length(rel_loss_avg)];
iterations = lfs.-1;
datapassbnds = iterations.*epocsperiters;
x_val = datapassbnds.*([collect(1:lfs[i]) for i=1:length(taulist)])./lfs;
pyplot()
p = plot(x_val[1], rel_loss_avg[1],
        xlabel="epochs", ylabel="residual", yscale=:log10, label=output.name,
        linestyle=:auto, tickfont=font(fontsmll), guidefont=font(fontbig), legendfont=font(fontmed), 
        markersize=6, linewidth=4, marker=:auto, grid=false);
for i=2:length(taulist)
    println(i);
    output = OUTPUTS[1+(i-1)*numsimu];
    plot!(p, x_val[i], rel_loss_avg[i],
        xlabel="epochs", ylabel="residual", yscale=:log10, label=output.name,
        linestyle=:auto, tickfont=font(fontsmll), guidefont=font(fontbig), legendfont=font(fontmed), 
        markersize=6, linewidth=4, marker=:auto, grid=false)
end
display(p)
# savenameempcomplex = string(savenamecomp, "epoc-rel-loss-$(numsimu)-avg");
# savefig("./figures/$(savenameempcomplex).pdf");