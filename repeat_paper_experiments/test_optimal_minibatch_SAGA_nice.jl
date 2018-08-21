using JLD
using Plots
using StatsBase
using Match
using Combinatorics
include("../src/StochOpt.jl")


### LOADING DATA ###
println("--- Loading data ---");
## Load LIBSVM data
# # # probname = "australian";  #australian, letter_scale, heart
# # # X, y = loadDataset(probname);
# # # X = X';
# # # length(X)
# # # y = convert(Array{Float64}, 1:1:size(X)[2]); # 14 data taken from X instead of 690

## Load artificial data
numfeatures = 20;
numdata = 15;
X, y, probname = gen_gauss_data(numfeatures, numdata, lambda=0.0);


### SETTING UP THE PROBLEM ###
println("\n--- Setting up the problem ---");
options = set_options(max_iter=10^8, max_time=0.02, max_epocs=3000, repeat_stepsize_calculation=true, skip_error_calculation=51);
prob = load_ridge_regression(X, y, probname, options, lambda=-1);  # Loads logisitc problem


### COMPUTE SAGA-NICE COMPLEXITIES ###
println("\n--- Compute SAGA-nice complexities (iteration, total and iteration by Hofman et al.) ---");
default_path = "./data/"; savename = replace(replace(prob.name, r"[\/]", "-"), ".", "_");
savenamesmooth = string(savename,"-smoothnesconsts");
itercomp = 0.0; Lsides = 0.0; Rsides = 0.0;
try
    itercomp, Lsides, Rsides = load("$(default_path)$(savenamesmooth).jld", "itercomp", "Lsides", "Rsides");
    println("found ", "$(default_path)$(savenamesmooth).jld with itercomp\n", itercomp);
catch loaderror   # Calculate iteration complexity for all minibatchsizes
    println(loaderror);
    itercomp, Lsides, Rsides = calculate_complex_SAGA_nice(prob, options);
    # L = eigmax(prob.X*prob.X')/prob.numdata+prob.lambda;
    save("$(default_path)$(savenamesmooth).jld", "itercomp", itercomp, "Lsides", Lsides, "Rsides", Rsides);
end

## Total complexity equals the iteration complexity times the size of the batch
totcomp = (itercomp').*(1:prob.numdata);

## Calculate iteration complexity from T. Hofmann, A. Lucchi, S. Lacoste-Julien &  B. C. McWilliams, Variance reduced stochastic gradient descent with neighbors. NIPS 28, 2015, 2305-2313
itercomphoff = calculate_complex_Hofmann(prob, options);


### PLOTING ###
println("\n--- Ploting complexities ??? ---");
pgfplots() # pyplot
fontsmll = 8; fontmed = 14; fontbig = 14;
plot([totcomp itercomp' itercomphoff'], label=["our total complex" "our iter complex" "Hofmann et al. iter complex"],
    linestyle=:auto, xlabel="batchsize", tickfont=font(fontsmll), guidefont=font(fontbig), legendfont=font(fontmed),
    markersize=6, linewidth=4, marker=:auto, grid=false)
ylims!((minimum(itercomp), maximum(totcomp)+minimum(itercomp)))
savefig("./figures/$(savenamesmooth).pdf");

# Comparing only the iteration complexities
plot(Lsides', ylabel="expected smoothness", label=false, xlabel="batchsize", tickfont=font(fontsmll), guidefont=font(fontbig),
    markersize=6, linewidth=4, marker=:auto, grid=false)
savenamesmoothhof = string(savenamesmooth, "-exp");
savefig("./figures/$(savenamesmoothhof).pdf");

plot([itercomp' itercomphoff'], label=["our iter complex" "Hofmann et al."], linestyle=:auto, xlabel="batchsize", tickfont=font(fontsmll),
    guidefont=font(fontbig), legendfont=font(fontmed), markersize=6, linewidth=4, marker=:auto, grid=false)
savenamesmoothhof = string(savenamesmooth, "-hof");
savefig("./figures/$(savenamesmoothhof).pdf");


# # # n= 100;
# # # batchsizes = 1:n;
# # # rhos = (n./batchsizes).*(n.-batchsizes)./(n-1);
# # # plot(rhos, label=false, linestyle=:auto,  xlabel="batchsize", ylabel="sketchresidual",  tickfont=font(fontsmll), guidefont=font(fontbig), legendfont =font(fontmed), markersize=6, linewidth =4, marker =:auto,  grid=false)
# # # savefig("./figures/sketchresidualmini.pdf");


# # ## Running methods
# # # OUTPUTS = [];  # List of saved outputs
# # # ######
# # # ## Basic parameters and options for solvers# =100, ,skip_error_calculation =5
# # # options.batchsize= indmin(totcomp); #Optimal minibatch size
# # # SAGA = intiate_SAGA(prob , options, minibatch_type="nice")
# # # output= minimizeFunc(prob, SAGA, options);
# # # OUTPUTS = [OUTPUTS ; output];
# # # # # # #####
# # # options.batchsize= 1; #Optimal minibatch size
# # # SAGA = intiate_SAGA(prob , options, minibatch_type="nice")
# # # output= minimizeFunc(prob, SAGA, options);
# # # OUTPUTS = [OUTPUTS ; output];
# # # # ####
# # # # SAGA = intiate_SAGA(prob , options, minibatch_type="partition", probability_type= "uni")
# # # # output= minimizeFunc(prob, SAGA, options);
# # # # OUTPUTS = [OUTPUTS ; output];
# # # # # #
# # # save("$(default_path)$(savename)-opt-minibatch.jld", "OUTPUTS",OUTPUTS);
# # #
# # # pyplot()# gr() pyplot() # pgfplots() #plotly()
# # # plot_outputs_Plots(OUTPUTS,prob,options) # Plot and save output
