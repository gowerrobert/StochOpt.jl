using JLD
using Plots
using StatsBase
using Match
using Combinatorics
include("../src/StochOpt.jl") # Be carefull about the path here

### SAVING AND PRINTING OPTIONS ###
default_path = "./data/"; savename = replace(replace(prob.name, r"[\/]", "-"), ".", "_");
savenamecomp = string(savename);
fontsmll = 8; fontmed = 14; fontbig = 14;


### LOADING DATA ###
probname = "diagonal"; # libsvm regression dataset | "gaussian", "diagonal" or "lone_eig_val" for artificaly generated data

# If probname="artificial", precise the number of features and data
numdata = 10;
numfeatures = 50; # useless for gen_diag_data

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

mu = get_mu_str_conv(prob); # TO BE VERIFIED
L = get_LC(prob, collect(1:n));
Li_s = get_Li(prob);
Lmax = maximum(Li_s);
Lbar = mean(Li_s);


################################################ EMPIRICAL BOUNDS ################################################

### COMPUTING THE UPPER-BOUNDS OF THE EXPECTED SMOOTHNESS CONSTANT ###
if(computeexpsmooth)
    expsmoothcst = zeros(n, 1);
end
simplebound = zeros(n, 1);
heuristicbound = zeros(n, 1);
concentrationbound = zeros(n, 1);
for tau = 1:n
    print("Calculating bounds for tau = ", tau, "\n");
    if(computeexpsmooth)
        tic();
        expsmoothcst[tau] = get_expected_smoothness_cst(prob, tau);
    end
    leftcoeff = (n*(tau-1))/(tau*(n-1));
    rightcoeff = (n-tau)/(tau*(n-1));
    simplebound[tau] = leftcoeff*Lbar + rightcoeff*Lmax;
    heuristicbound[tau] = leftcoeff*L + rightcoeff*Lmax;
    concentrationbound[tau] = ((2*n*(tau-1))/(tau*(n-1)))*L + (1/tau)*(((n-tau)/(n-1)) + (4*log(d))/3)*Lmax;
    concentrationbound[tau] = 2*leftcoeff*L + (rightcoeff + (4*log(d))/(3*tau))*Lmax; 
    if(computeexpsmooth) toc(); end
end

### PLOTING ###
println("\n--- Ploting upper bounds of the expected smoothness constant ---");
# plotly()
pyplot()
# PROBLEM: there is still a problem of ticking non integer on the xaxis
if(computeexpsmooth)
    plot(1:n, [heuristicbound simplebound concentrationbound expsmoothcst], label=["heuristic" "simple" "concentration" "true"],
    linestyle=:auto, xlabel="batchsize", ylabel="smoothness constant",tickfont=font(fontsmll), # xticks=1:n, 
    guidefont=font(fontbig), legendfont=font(fontmed), markersize=6, linewidth=4, marker=:auto, grid=false, 
    ylim=(0, max(maximum(simplebound),maximum(concentrationbound),maximum(heuristicbound))+minimum(expsmoothcst)),
    title=string("Pb: ", probname, ", n=", string(n), ", d=", string(d)))
else
    plot(1:n, [heuristicbound simplebound concentrationbound], label=["heuristic" "simple" "concentration"],
    linestyle=:auto, xlabel="batchsize", ylabel="smoothness constant",tickfont=font(fontsmll), # xticks=1:n, 
    guidefont=font(fontbig), legendfont=font(fontmed), linewidth=4, grid=false, 
    ylim=(0, max(maximum(simplebound),maximum(concentrationbound),maximum(heuristicbound))+minimum(heuristicbound)),
    title=string("Pb: ", probname, ", n=", string(n), ", d=", string(d)))
end
savenameexpsmooth = string(savenamecomp, "-expsmoothbounds");
savefig("./figures/$(savenameexpsmooth).pdf");

# Zoom
if(computeexpsmooth)
    plot(1:n, [heuristicbound simplebound concentrationbound expsmoothcst], label=["heuristic" "simple" "concentration" "true"],
    linestyle=:auto, xlabel="batchsize", ylabel="smoothness constant", tickfont=font(fontsmll), #xticks=1:n, 
    guidefont=font(fontbig), legendfont=font(fontmed), markersize=6, linewidth=4, marker=:auto, grid=false, 
    ylim=(0.85*minimum(expsmoothcst), 1.2*max(maximum(simplebound), maximum(heuristicbound))),
    title=string("Pb: ", probname, ", n=", string(n), ", d=", string(d)," zoom"))
else
    plot(1:n, [heuristicbound simplebound concentrationbound], label=["heuristic" "simple" "concentration"],
    linestyle=:auto, xlabel="batchsize", ylabel="smoothness constant", tickfont=font(fontsmll), #xticks=1:n, 
    guidefont=font(fontbig), legendfont=font(fontmed), linewidth=4, grid=false,  #marker=:auto,
    # ylim=(0.85*minimum(heuristicbound), 1.2*max(maximum(simplebound), maximum(heuristicbound))),
    ylim=(0.85*minimum(heuristicbound), 10.0*minimum(heuristicbound)),
    title=string("Pb: ", probname, ", n=", string(n), ", d=", string(d)," zoom"))
end
savenameexpsmoothzoom = string(savenameexpsmooth, "-zoom");
savefig("./figures/$(savenameexpsmoothzoom).pdf");


##################################################################################################################

## Compute optimal tau
tautheory = round(Int, 1 + (mu*(n-1))/(4*Lbar)) # One should not add again lambda since it is already taken into account in Lbar
tauheuristic = round(Int, 1 + (mu*(n-1))/(4*L))

println("\nPROBLEM DIMENSIONS:");
println("   Number of datapoints", n); # n in the paper notation
println("   Number of features", d); # d in the paper notation

println("\nSMOOTHNESS CONSTANTS:");
println("   Lmax : ", Lmax);
println("   L : ", L);
println("   Lbar : ", Lbar);

println("\nTheoretical optimal tau = ", tautheory);
println("Heuristic optimal tau = ", tauheuristic);

savenamecsts = string(savenamecomp, "-constants");
if(computeexpsmooth)
save("./figures/$(savenamecsts)-with-true-expected-smoothness-cst.jld", "n", n, "d", d, "mu", mu, "L", L, "Lmax", Lmax, "Lbar", Lbar, "Li_s", Li_s,
    "tautheory", tautheory, "tauheuristic", tauheuristic, "expsmoothcst", expsmoothcst);
else
    save("./figures/$(savenamecsts).jld", "n", n, "d", d, "mu", mu, "L", L, "Lmax", Lmax, "Lbar", Lbar, "Li_s", Li_s);
end
