using JLD
using Plots
using StatsBase
using Match
using Combinatorics

tic();

include("../src/StochOpt.jl") # Be carefull about the path here
srand(1234) # fixing the seed


### LOADING DATA ###
probname = "gaussian"; # libsvm regression dataset | "gaussian", "diagonal" or "lone_eig_val" for artificaly generated data

# If probname="artificial", precise the number of features and data
numdata = 500;
numfeatures = 20; # useless for gen_diag_data

println("--- Loading data ---");
probnames = ["abalone", "housing"]; 
#, "letter_scale", "heart", "phishing", "madelon", "a9a", 
# "mushrooms", "phishing", "w8a", "gisette_scale", 
if(probname == "gaussian")
    ## Load artificial data
    X, y, probname = gen_gauss_data(numfeatures, numdata, lambda=0.0, err=0.001);
elseif(probname == "diagonal")
    X, y, probname = gen_diag_data(numdata, lambda=0.0, Lmax=2);
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
datathreshold = 20;
if(n > datathreshold) # if n is too large we do not compute the exact expected smoothness constant nor its relative quantities
    println("The number of data is to large to compute the expected smoothness constant exactly");
    computeexpsmooth = false;
else
    computeexpsmooth = true;
end

# Lmax = maximum(sum(prob.X.^2, 1)) + prob.lambda; # Algebraic formula
# Lbis =  eigmax(prob.X*prob.X')/n + prob.lambda;
# mu = minimum(sum(prob.X.^2, 1)) + prob.lambda # TO BE VERIFIED
mu = get_mu_str_conv(prob); # TO BE VERIFIED
L = get_LC(prob, collect(1:n));
Li_s = get_Li(prob);
Lmax = maximum(Li_s);
Lbar = mean(Li_s);


############################################ THEORETICAL COMPLEXITIES ############################################

# ### CREATING THE MINI-BATCH SIZE SEQUENCE ###
# # tauseq = collect(1:14);
# # tauseq = [1, 2, 3, 4, 5, 14];
# # tauseq = cat(1, 1:4, 14);
# # tauseq = [1, 2, 3, 3, 3, 4, 5, 5, 14]; # test uniqueness
# # tauseq = [1, 5, 4, 3, 2, 14]; # test sorting

# # tauseq = [1, 2, 3, numdata];
# tauseq = [1, 2, 3, 4, 5];

# # Sanity checks
# tauseq = unique(sort(tauseq));
# n = prob.numdata;
# numtau = length(tauseq);
# if(minimum(tauseq) < 1 || maximum(tauseq) > n)
#     error("values of tauseq are out of range.");
# end
# println("\n--- Mini-batch sequence ---");
# println(tauseq);


# ### COMPUTE SAGA-NICE THEORETIDCAL COMPLEXITIES ###
# println("\n--- Compute SAGA-nice theoretical complexities (iteration and total) ---");
# default_path = "./data/"; savename = replace(replace(prob.name, r"[\/]", "-"), ".", "_");
# savenamecomp = string(savename,"-complexities-nidham");
# itercomp = 0.0; Lsides = 0.0; Rsides = 0.0;
# try
#     itercomp, Lsides, Rsides = load("$(default_path)$(savenamecomp).jld", "itercomp", "Lsides", "Rsides");
#     println("found ", "$(default_path)$(savenamecomp).jld with itercomp\n", itercomp);
# catch loaderror   # Calculate iteration complexity for all minibatchsizes
#     println(loaderror);
#     itercomp, Lsides, Rsides = calculate_complex_SAGA_nice(prob, options, tauseq);
#     # L = eigmax(prob.X*prob.X')/prob.numdata+prob.lambda;
#     # save("$(default_path)$(savenamecomp).jld", "itercomp", itercomp, "Lsides", Lsides, "Rsides", Rsides);
# end

# println("Mini-batch size sequence:\n", tauseq)

# ## Total complexity equals the iteration complexity times the size of the batch
# # totcomp = (itercomp').*(1:prob.numdata);
# totcomp = (itercomp').*tauseq;

# ### PLOTING ###
# println("\n--- Ploting complexities ??? ---");
# pyplot() # pyplot
# fontsmll = 8; fontmed = 14; fontbig = 14;
# plot(tauseq, [totcomp itercomp'], label=["total complex" "iter complex"],
#     linestyle=:auto, xlabel="batchsize", ylabel="complexity", tickfont=font(fontsmll),
#     guidefont=font(fontbig), legendfont=font(fontmed), markersize=6, linewidth=4, marker=:auto,
#     grid=false, ylim=(0, maximum(totcomp)+minimum(itercomp)), xticks=tauseq)
#    ylim=(minimum(itercomp), maximum(totcomp)+minimum(itercomp))
# # savefig("./figures/$(savenamecomp).pdf");

# # Comparing only the iteration complexities
# ## WARNING: Lsides is not exactly the expected smoothness cosntant but 4*\cL/mu !!
# plot(tauseq, Lsides', ylabel="expected smoothness", xlabel="batchsize", tickfont=font(fontsmll),
#     guidefont=font(fontbig), markersize=6, linewidth=4, marker=:auto, grid=false, legend=false, 
#     xticks=tauseq)
# savenameexpsmooth = string(savenamecomp, "-expsmooth");
# # savefig("./figures/$(savenameexpsmooth).pdf");

##################################################################################################################


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
default_path = "./data/"; savename = replace(replace(prob.name, r"[\/]", "-"), ".", "_");
savenamecomp = string(savename,"-nidham");
fontsmll = 8; fontmed = 14; fontbig = 14;
# # plotly()
# pyplot()
# # PROBLEM: there is still a problem of ticking non integer on the xaxis
# if(computeexpsmooth)
#     plot(1:n, [simplebound concentrationbound heuristicbound expsmoothcst], label=["simple" "concentration" "heuristic" "true"],
#     linestyle=:auto, xlabel="batchsize", ylabel="smoothness constant",tickfont=font(fontsmll), # xticks=1:n, 
#     guidefont=font(fontbig), legendfont=font(fontmed), markersize=6, linewidth=4, marker=:auto, grid=false, 
#     ylim=(0, max(maximum(simplebound),maximum(concentrationbound),maximum(heuristicbound))+minimum(expsmoothcst)),
#     title=string("Pb: ", probname, ", n=", string(n), ", d=", string(d)))
# else
#     plot(1:n, [simplebound concentrationbound heuristicbound], label=["simple" "concentration" "heuristic"],
#     linestyle=:auto, xlabel="batchsize", ylabel="smoothness constant",tickfont=font(fontsmll), # xticks=1:n, 
#     guidefont=font(fontbig), legendfont=font(fontmed), linewidth=4, grid=false, 
#     ylim=(0, max(maximum(simplebound),maximum(concentrationbound),maximum(heuristicbound))+minimum(heuristicbound)),
#     title=string("Pb: ", probname, ", n=", string(n), ", d=", string(d)))
# end
savenameexpsmooth = string(savenamecomp, "-expsmoothbounds");
# savefig("./figures/$(savenameexpsmooth).pdf");

# Zoom
# if(computeexpsmooth)
#     plot(1:n, [simplebound concentrationbound heuristicbound expsmoothcst], label=["simple" "concentration" "heuristic" "true"],
#     linestyle=:auto, xlabel="batchsize", ylabel="smoothness constant", tickfont=font(fontsmll), #xticks=1:n, 
#     guidefont=font(fontbig), legendfont=font(fontmed), markersize=6, linewidth=4, marker=:auto, grid=false, 
#     ylim=(0.85*minimum(expsmoothcst), 1.2*max(maximum(simplebound), maximum(heuristicbound))),
#     title=string("Pb: ", probname, ", n=", string(n), ", d=", string(d)," zoom"))
# else
#     plot(1:n, [simplebound concentrationbound heuristicbound], label=["simple" "concentration" "heuristic"],
#     linestyle=:auto, xlabel="batchsize", ylabel="smoothness constant", tickfont=font(fontsmll), #xticks=1:n, 
#     guidefont=font(fontbig), legendfont=font(fontmed), linewidth=4, grid=false,  #marker=:auto,
#     # ylim=(0.85*minimum(heuristicbound), 1.2*max(maximum(simplebound), maximum(heuristicbound))),
#     ylim=(0.85*minimum(heuristicbound), 10.0*minimum(heuristicbound)),
#     title=string("Pb: ", probname, ", n=", string(n), ", d=", string(d)," zoom"))
# end
savenameexpsmoothzoom = string(savenamecomp, "-expsmoothbounds-zoom");
# savefig("./figures/$(savenameexpsmoothzoom).pdf");

# heuristic equals true expected smoothness constant for tau=1 and n as expected, else it is above as hoped
# heuristicbound .== expsmoothcst
# heuristicbound .> expsmoothcst

simplebound[end] - heuristicbound[end]
concentrationbound[end] - simplebound[end]

######################################## EMPIRICAL BOUNDS OF THE STEPSIZES #######################################

#TO BE DONE : need to implement line-search (does is make sense for SGD?) to see how the computed stepsizes are far from our lower bounds

### COMPUTING THE UPPER-BOUNDS OF THE STEPSIZES ###
rho = ((n-(1:n) )./((1:n).*(n-1)));
rightterm = (rho/n)*Lmax + (mu*n)/(4*(1:n)); # Right-hand side term in the max
if(computeexpsmooth)
    truestepsize = (1/4).* (1./max.(expsmoothcst, rightterm));
end
simplestepsize = (1/4).* (1./max.(simplebound, rightterm));
concentrationstepsize = (1/4).* (1./max.(concentrationbound, rightterm));
heuristicstepsize = (1/4).* (1./max.(heuristicbound, rightterm));

### PLOTING ###
println("\n--- Ploting stepsizes ---");
default_path = "./data/"; savename = replace(replace(prob.name, r"[\/]", "-"), ".", "_");
savenamecomp = string(savename,"-nidham");
# # plotly()
# pyplot()
# # PROBLEM: there is still a problem of ticking non integer on the xaxis
# if(computeexpsmooth)
#     plot(1:n, [simplestepsize concentrationstepsize heuristicstepsize truestepsize], label=["simple" "concentration" "heuristic" "true"],
#     linestyle=:auto, xlabel="batchsize", ylabel="step size",tickfont=font(fontsmll), # xticks=1:n, 
#     guidefont=font(fontbig), legendfont=font(fontmed), markersize=6, linewidth=4, marker=:auto, grid=false, 
#     ylim=(0, maximum(truestepsize)+minimum(concentrationstepsize)),
#     title=string("Pb: ", probname, ", n=", string(n), ", d=", string(d)))
# else
#     plot(1:n, [simplestepsize concentrationstepsize heuristicstepsize], label=["simple" "concentration" "heuristic"],
#     linestyle=:auto, xlabel="batchsize", ylabel="step size",tickfont=font(fontsmll), # xticks=1:n, 
#     guidefont=font(fontbig), legendfont=font(fontmed), markersize=6, linewidth=4, grid=false, #marker=:auto, 
#     ylim=(0, maximum(heuristicstepsize)+minimum(concentrationstepsize)),
#     title=string("Pb: ", probname, ", n=", string(n), ", d=", string(d)))
# end

# savenamestepsize = string(savenamecomp, "-stepsizes");
# savefig("./figures/$(savenamestepsize).pdf");

## Empirical stepsizes returned by optimal mini-batch SAGA with line searchs
# WORK IN PROGRESS

######################################## EMPIRICAL OPTIMAL MINIBATCH SIZE ########################################

## Compute optimal tau
tautheory = round(Int, 1 + (mu*(n-1))/(4*Lbar)); # One should not add again lambda since it is already taken into account in Lbar
tauheuristic = round(Int, 1 + (mu*(n-1))/(4*L));
# sleep(3);

## Empirical stepsizes returned by optimal mini-batch SAGa with line searchs
if(n <= datathreshold)
    taulist = 1:n;
else
    taulist = [1; tautheory; n]#[collect(1:(tautheory+1)); n];
end

# taulist = collect(1:10);


# options = set_options(max_iter=10^8, max_time=10.0, max_epocs=100, repeat_stepsize_calculation=true, skip_error_calculation=51,
#                       force_continue=false, initial_point="rand");
tolerance = 10.0^(-1.0); # epsilon for which: (f(x)-fsol)/(f0-fsol) < epsilon
options = set_options(tol=tolerance, max_iter=10^8, max_time=1000.0, max_epocs=3000, initial_point="rand",
                    #   repeat_stepsize_calculation=true,
                      skip_error_calculation=51,
                      force_continue=false); # fix initial point to zeros for a maybe fairer comparison?

numsimu = 10;
itercomplex = zeros(length(taulist), 1); # List of saved outputs
for idxtau in 1:length(taulist) # 1:n
    tau = taulist[idxtau];
    println("\nCurrent mini-batch size: ", tau);
    options.batchsize = tau;
    for i=1:numsimu
        sg = initiate_SAGA(prob, options, minibatch_type="nice");
        output = minimizeFunc(prob, sg, options);
        itercomplex[idxtau] += output.iterations;
    end
end
itercomplex = itercomplex ./ numsimu;

# gr()
# plot_outputs_Plots(OUTPUTS, prob, options); # Plot and save output

## Computing the empirical complexity
# empcomplex = taulist.*[OUTPUTS[i].iterations for i=1:length(OUTPUTS)] # tau times number of iterations
empcomplex = taulist.*itercomplex;
# # plotly()
# pyplot()
# plot(taulist, empcomplex, linestyle=:solid, xlabel="batchsize", ylabel="empirical complexity",
#     # ylim=(0, maximum(empcomplex)+minimum(empcomplex)),
#     xticks = taulist,
#     # xticks=(taulist, ["1\n= tau_theory" "2" "3" "4" "n"]),
#     legend=false, guidefont=font(fontbig), linewidth=4, grid=false, #marker=:auto,
#     title=string("Pb: ", probname, ", n=", string(n), ", d=", string(d)))

# # plotly()
# pyplot()
# plot(taulist[1:end-1], empcomplex[1:end-1], linestyle=:solid, xlabel="batchsize", ylabel="empirical complexity",
#     # ylim=(0, maximum(empcomplex)+minimum(empcomplex)),
#     xticks = taulist, 
#     # xticks=(taulist, ["1\n= tau_theory" "2" "3" "4" "n"]),
#     legend=false, guidefont=font(fontbig), linewidth=4, grid=false, #marker=:auto,
#     title=string("Pb: ", probname, ", n=", string(n), ", d=", string(d)))

println("\nPROBLEM DIMENSIONS:");
println("   Number of datapoints", n); # n in the paper notation
println("   Number of features", d); # d in the paper notation

println("\nSMOOTHNESS CONSTANTS:");
println("   Lmax : ", Lmax);
println("   L : ", L);
# println("Li_s : ", Li_s);
println("   Lbar : ", Lbar);
# sleep(3);

println("\nTheoretical optimal tau = ", tautheory);
println("Heuristic optimal tau = ", tauheuristic);

println("List of mini-batch sizes = ", taulist);
println("\nEmpirical complexity = ", empcomplex);

toc();