srand(1);

using JLD
using Plots
using StatsBase
using Match
using Combinatorics

include("../src/StochOpt.jl") # Be carefull about the path here


### LOADING DATA ###
data = "YearPredictionMSD"; # libsvm regression dataset | "gaussian", "diagonal" or "lone_eig_val" for artificaly generated data

# If probname="artificial", precise the number of features and data
numdata = 500;
numfeatures = 12; # useless for gen_diag_data

println("--- Loading data ---");
datasets = ["YearPredictionMSD", "abalone", "housing"];
#, "letter_scale", "heart", "phishing", "madelon", "a9a",
# "mushrooms", "phishing", "w8a", "gisette_scale",
if(data == "gaussian")
    ## Load artificial data
    X, y, probname = gen_gauss_data(numfeatures, numdata, lambda=0.0, err=0.001);
elseif(data == "diagonal")
    X, y, probname = gen_diag_data(numdata, lambda=0.0, Lmax=100);
elseif(data == "lone_eig_val")
    X, y, probname = gen_diag_alone_eig_data(numfeatures, numdata, lambda=0.0, a=100, err=0.001);
elseif(data in datasets)
    probname = data;
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
                      force_continue=true, initial_point="randn", batchsize=0);
prob = load_ridge_regression(X, y, probname, options, lambda=-1, scaling="none");  # Disabling scaling
# QUESTION: how is lambda selected?
n = prob.numdata;
d = prob.numfeatures;

### PLOTTING SETTINGS ###
println("\n--- Plotting upper bounds of the expected smoothness constant ---");
default_path = "./data/"; savename = replace(replace(prob.name, r"[\/]", "-"), ".", "_");
savenamecomp = string(savename);
fontsmll = 8; fontmed = 14; fontbig = 14;

### COMPUTING THE SMOOTHNESS CONSTANTS ###
# Compute the smoothness constants L, L_max, \cL, \bar{L}
datathreshold = 24; # if n is too large we do not compute the exact expected smoothness constant nor its relative quantities

mu = get_mu_str_conv(X, n, d, prob.lambda); # mu = minimum(sum(prob.X.^2, 1)) + prob.lambda;
L = get_LC(prob.X, prob.lambda, collect(1:n)); # L = eigmax(prob.X*prob.X')/n + prob.lambda;
Li_s = get_Li(prob.X, prob.lambda);
Lmax = maximum(Li_s); # Lmax = maximum(sum(prob.X.^2, 1)) + prob.lambda;
Lbar = mean(Li_s);


########################### EMPIRICAL UPPER BOUNDS OF THE EXPECTED SMOOTHNESS CONSTANT ###########################
#region
### COMPUTING THE BOUNDS
simplebound, bernsteinbound, heuristicbound, expsmoothcst = get_expected_smoothness_bounds(prob);

### PLOTTING ###
println("\n--- Plotting upper bounds ---");
# PROBLEM: there is still a problem of ticking non integer on the xaxis
pyplot()
plot_expected_smoothness_bounds(prob, simplebound, bernsteinbound, heuristicbound, expsmoothcst);

# heuristic equals true expected smoothness constant for tau=1 and n as expected, else it is above as hoped
if(n<=datathreshold)
    println("Heuristic - expected smoothness gap", heuristicbound - expsmoothcst)
    println("Simple - heuristic gap", simplebound[end] - heuristicbound[end])
    println("Bernstein - simple gap", bernsteinbound[end] - simplebound[end])
end
#endregion
##################################################################################################################


##################################### EMPIRICAL UPPER BOUNDS OF THE STEPSIZES ####################################
#region
# TO BE DONE: implement grid-search for the stepsizes, i.e.
# 1) set a grid of stepsizes around 1/(4Lmax)
# 2) run several SAGA_nice on the same problem with different stepsize (average?)
# 3) pick the 'best' stepsize

### COMPUTING THE UPPER-BOUNDS OF THE STEPSIZES ###
simplestepsize, bernsteinstepsize, heuristicstepsize, expsmoothstepsize = get_stepsize_bounds(prob, simplebound, bernsteinbound, heuristicbound, expsmoothcst);

### PLOTTING ###
println("\n--- Plotting stepsizes ---");
# PROBLEM: there is still a problem of ticking non integer on the xaxis
pyplot()
plot_stepsize_bounds(prob, simplestepsize, bernsteinstepsize, heuristicstepsize, expsmoothstepsize);
#endregion
##################################################################################################################


###################################### THEORETICAL OPTIMAL MINI-BATCH SIZES ######################################
#region
## Compute optimal mini-batch size
if typeof(expsmoothcst)==Array{Float64,2}
    LHS = 4*(1:n).*(expsmoothcst+prob.lambda)./mu;
    RHS = n + (n-(1:n)) .* (4*(Lmax+prob.lambda)/((n-1)*mu));
    exacttotalcplx = max.(LHS, RHS);
    _, opt_minibatch_exact = findmin(exacttotalcplx);
else
    opt_minibatch_exact = nothing;
end
## WARNING: Verify computations : should we add lambda????
opt_minibatch_simple = round(Int, 1 + (mu*(n-1))/(4*Lbar)); # One should not add again lambda since it is already taken into account in Lbar
opt_minibatch_bernstein = max(1, round(Int, 1 + (mu*(n-1))/(8*L) - (4/3)*log(d)*((n-1)/n)*(Lmax/(2*L)) )); ## WARNING: Verify computations : should we add lambda????
opt_minibatch_heuristic = round(Int, 1 + (mu*(n-1))/(4*L));
#endregion
##################################################################################################################


########################################### SAVNG RESULTS ########################################################
save_SAGA_nice_constants(prob, data, simplebound, bernsteinbound, heuristicbound, expsmoothcst,
                         simplestepsize, bernsteinstepsize, heuristicstepsize, expsmoothstepsize,
                         opt_minibatch_simple, opt_minibatch_bernstein, opt_minibatch_heuristic,
                         opt_minibatch_exact);
##################################################################################################################


######################################### PRINTING CONSTANTS AND RESULTS #########################################
println("\nPROBLEM DIMENSIONS:");
println("   Number of datapoints = ", n); # n in the paper notation
println("   Number of features = ", d); # d in the paper notation

println("\nSimple optimal tau = ", opt_minibatch_simple);
println("Bernstein optimal tau = ", opt_minibatch_bernstein);
println("Heuristic optimal tau = ", opt_minibatch_heuristic);

println("\nSMOOTHNESS CONSTANTS:");
println("   Lmax = ", Lmax);
println("   L = ", L);
println("Li_s = ", Li_s);
println("   Lbar = ", Lbar);
##################################################################################################################














## DRAFTS ##

############################################ THEORETICAL COMPLEXITIES ############################################
#region

#===
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

### COMPUTE SAGA-NICE THEORETICAL COMPLEXITIES ###
println("\n--- Compute SAGA-nice theoretical complexities (iteration and total) ---");
default_path = "./data/"; savename = replace(replace(prob.name, r"[\/]", "-"), ".", "_");
savenamecompperso = string(savename,"-complexities");
itercomp = 0.0; Lsides = 0.0; Rsides = 0.0;
try
    itercomp, Lsides, Rsides = load("$(default_path)$(savenamecompperso).jld", "itercomp", "Lsides", "Rsides");
    println("found ", "$(default_path)$(savenamecompperso).jld with itercomp\n", itercomp);
catch loaderror   # Calculate iteration complexity for all minibatchsizes
    println(loaderror);
    itercomp, Lsides, Rsides = calculate_complex_SAGA_nice(prob, options, tauseq);
    # L = eigmax(prob.X*prob.X')/prob.numdata+prob.lambda;
    # save("$(default_path)$(savenamecompperso).jld", "itercomp", itercomp, "Lsides", Lsides, "Rsides", Rsides);
end

println("Mini-batch size sequence:\n", tauseq)

## Total complexity equals the iteration complexity times the size of the batch
# totcomp = (itercomp').*(1:prob.numdata);
totcomp = (itercomp').*tauseq;

### PLOTTING ###
println("\n--- Plotting complexities ??? ---");
pyplot() # pyplot
fontsmll = 8; fontmed = 14; fontbig = 14;
plot(tauseq, [totcomp itercomp'], label=["total complex" "iter complex"],
    linestyle=:auto, xlabel="batchsize", ylabel="complexity", tickfont=font(fontsmll),
    guidefont=font(fontbig), legendfont=font(fontmed), markersize=6, linewidth=4, marker=:auto,
    grid=false, ylim=(0, maximum(totcomp)+minimum(itercomp)), xticks=tauseq)
   ylim=(minimum(itercomp), maximum(totcomp)+minimum(itercomp))
# savefig("./figures/$(savenamecompperso).pdf");

# Comparing only the iteration complexities
## WARNING: Lsides is not exactly the expected smoothness cosntant but 4*\cL/mu !!
plot(tauseq, Lsides', ylabel="expected smoothness", xlabel="batchsize", tickfont=font(fontsmll),
    guidefont=font(fontbig), markersize=6, linewidth=4, marker=:auto, grid=false, legend=false,
    xticks=tauseq)
savenameexpsmooth = string(savenamecompperso, "-expsmooth");
# savefig("./figures/$(savenameexpsmooth).pdf");
===#
#endregion
##################################################################################################################