"""
    plot_expected_smoothness_bounds(prob::Prob, simplebound::Array{Float64}, bernsteinbound::Array{Float64}, heuristicbound::Array{Float64}, expsmoothcst)

Plots two upper-bounds of the expected smoothness constant (simple and Bernstein), 
a heuristic estimation of it and its exact value (if there are few data points).

#INPUTS:\\
    - **Prob** prob: considered problem, i.e. logistic regression, ridge ression... (see src/StochOpt.jl)\\
    - nx1 **Array{Float64,2}** simplebound: simple bound of the expected smoothness constant for each mini-batch size ``τ`` from 1 to n\\
    - nx1 **Array{Float64,2}** bernsteinbound: Bernstein bound of the expected smoothness constant for each mini-batch size ``τ`` from 1 to n\\
    - nx1 **Array{Float64,2}** heuristic: heuristic estimation of the expected smoothness constant for each mini-batch size ``τ`` from 1 to n\\
    - nx1 **Array{Float64,2}** or **Void** expsmoothcst: exact expected smoothness constant for each mini-batch size ``τ`` from 1 to n\\
#OUTPUTS:\\
    - None
"""
function plot_expected_smoothness_bounds(prob::Prob, simplebound::Array{Float64}, bernsteinbound::Array{Float64}, heuristicbound::Array{Float64}, expsmoothcst)
    # PROBLEM: there is still a problem of ticking non integer on the xaxis

    probname = replace(replace(prob.name, r"[\/]", "-"), ".", "_");
    default_path = "./figures/";
    fontsmll = 8;
    fontmed = 12;
    fontbig = 14;
    xlabeltxt = "batchsize";

    n = prob.numdata;
    d = prob.numfeatures;

    if typeof(expsmoothcst)==Array{Float64,2}
        plot(1:n, [heuristicbound simplebound bernsteinbound expsmoothcst], label=["heuristic" "simple" "bernstein" "true"],
             linestyle=:auto, xlabel=xlabeltxt, ylabel="smoothness constant", tickfont=font(fontsmll), # xticks=1:n,
             guidefont=font(fontbig), legendfont=font(fontmed), markersize=6, linewidth=4, marker=:auto, grid=false,
             ylim=(0, max(maximum(simplebound),maximum(bernsteinbound),maximum(heuristicbound))+minimum(expsmoothcst)),
             title=string(probname, ", n=", string(n), ", d=", string(d)));
    elseif typeof(expsmoothcst)==Void
        plot(1:n, [heuristicbound simplebound bernsteinbound], label=["heuristic" "simple" "bernstein"],
             linestyle=:auto, xlabel=xlabeltxt, ylabel="smoothness constant", tickfont=font(fontsmll), # xticks=1:n,
             guidefont=font(fontbig), legendfont=font(fontmed), linewidth=4, grid=false,
             ylim=(0, max(maximum(simplebound),maximum(bernsteinbound),maximum(heuristicbound))+minimum(heuristicbound)),
             title=string(probname, ", n=", string(n), ", d=", string(d)));
    else
        error("Wrong type of expsmoothcst");
    end
    savename = "-expsmoothbounds";
    savefig("$(default_path)$(probname)$(savename).pdf");

    # Zoom
    if typeof(expsmoothcst)==Array{Float64,2}
        plot(1:n, [heuristicbound simplebound bernsteinbound expsmoothcst], label=["heuristic" "simple" "bernstein" "true"],
             linestyle=:auto, xlabel=xlabeltxt, ylabel="smoothness constant", tickfont=font(fontsmll), #xticks=1:n,
             guidefont=font(fontbig), legendfont=font(fontmed), markersize=6, linewidth=4, marker=:auto, grid=false,
             ylim=(0.85*minimum(expsmoothcst), 1.2*max(maximum(simplebound), maximum(heuristicbound))),
             title=string(probname, ", n=", string(n), ", d=", string(d)," zoom"));
    elseif typeof(expsmoothcst)==Void
        plot(1:n, [heuristicbound simplebound bernsteinbound], label=["heuristic" "simple" "bernstein"],
             linestyle=:auto, xlabel=xlabeltxt, ylabel="smoothness constant", tickfont=font(fontsmll), #xticks=1:n,
             guidefont=font(fontbig), legendfont=font(fontmed), linewidth=4, grid=false,  #marker=:auto,
             # ylim=(0.85*minimum(heuristicbound), 1.2*max(maximum(simplebound), maximum(heuristicbound))),
             ylim=(0.85*minimum(heuristicbound), 1.5*minimum(heuristicbound)),
             title=string(probname, ", n=", string(n), ", d=", string(d)," zoom"));
    else
        error("Wrong type of expsmoothcst");
    end
    savenamezoom = string(savename, "-zoom");
    savefig("$(default_path)$(probname)$(savenamezoom).pdf");
end


"""
    plot_stepsize_bounds(prob::Prob, simplestepsize::Array{Float64}, bernsteinstepsize::Array{Float64}, heuristicstepsize::Array{Float64}, expsmoothstepsize)

Plots upper bounds of the stepsizes corresponding to the simple and Bernstein upper bounds, 
heuristic estimation and the exact expected smoothness constant (if there are few data points).

#INPUTS:\\
    - **Prob** prob: considered problem, i.e. logistic regression, ridge ression... (see src/StochOpt.jl)\\
    - nx1 **Array{Float64,2}** simplestepsize: simple bound stepsizes\\
    - nx1 **Array{Float64,2}** bernsteinstepsize: Bernstein bound stepsizes\\
    - nx1 **Array{Float64,2}** heuristicstepsize: heuristic estimation stepsizes\\
    - nx1 **Array{Float64,2}** or **Void** expsmoothstepsize: exact expected smoothness constant stepsize upper bound\\
#OUTPUTS:\\
    - None
"""
function plot_stepsize_bounds(prob::Prob, simplestepsize::Array{Float64}, bernsteinstepsize::Array{Float64}, heuristicstepsize::Array{Float64}, expsmoothstepsize)
    # PROBLEM: there is still a problem of ticking non integer on the xaxis

    probname = replace(replace(prob.name, r"[\/]", "-"), ".", "_");
    default_path = "./figures/";
    fontsmll = 8;
    fontmed = 12;
    fontbig = 14;
    xlabeltxt = "batchsize";
    
    n = prob.numdata;
    d = prob.numfeatures;

    if typeof(expsmoothstepsize)==Array{Float64,2}
        plot(1:n, [heuristicstepsize simplestepsize bernsteinstepsize expsmoothstepsize], label=["heuristic" "simple" "bernstein" "true"],
             linestyle=:auto, xlabel=xlabeltxt, ylabel="step size",tickfont=font(fontsmll), # xticks=1:n,
             guidefont=font(fontbig), legendfont=font(fontmed), markersize=6, linewidth=4, marker=:auto, grid=false,
             ylim=(0, maximum(expsmoothstepsize)+minimum(bernsteinstepsize)),
             # legend=:bottomright,
             title=string(probname, ", n=", string(n), ", d=", string(d)));
    elseif typeof(expsmoothstepsize)==Void
        plot(1:n, [heuristicstepsize simplestepsize bernsteinstepsize], label=["heuristic" "simple" "bernstein"],
             linestyle=:auto, xlabel=xlabeltxt, ylabel="step size",tickfont=font(fontsmll), # xticks=1:n,
             guidefont=font(fontbig), legendfont=font(fontmed), markersize=6, linewidth=4, grid=false, #marker=:auto,
             ylim=(0, maximum(heuristicstepsize)+minimum(bernsteinstepsize)),
             title=string(probname, ", n=", string(n), ", d=", string(d)));
    else
        error("Wrong type of expsmoothstepsize");
    end
    savename = "-stepsizes";
    savefig("$(default_path)$(probname)$(savename).pdf");
end

"""
    plot_empirical_complexity(prob::Prob, minibatchlist::Array{Int64,1}, empcomplex::Array{Float64,1}, 
                              opt_minibatch_simple::Int64,
                              opt_minibatch_bernstein::Int64,
                              opt_minibatch_heuristic::Int64,
                              opt_minibatch_emp::Int64)

Saves the plot of the empirical total complexity. 

#INPUTS:\\
    - **Prob** prob: considered problem, i.e. logistic regression, ridge ression... (see src/StochOpt.jl)\\
    - **Array{Int64,1}** minibatchlist: list of the different mini-batch sizes\\
    - **Array{Float64,1}** empcomplex: average total complexity (tau*iteration complexity) 
      for each of the mini-batch size (tau) over numsimu samples\\
    - **Int64** opt_minibatch_simple: simple bound optimal mini-batch size\\
    - **Int64** opt_minibatch_bernstein: Bernstein bound optimal mini-batch size\\
    - **Int64** opt_minibatch_heuristic: heuristic optimal mini-batch size\\
    - **Int64** opt_minibatch_emp: empirical optimal mini-batch size\\
#OUTPUTS:\\
    - None
"""
function plot_empirical_complexity(prob::Prob, minibatchlist::Array{Int64,1}, empcomplex::Array{Float64,1}, 
                                   opt_minibatch_simple::Int64,
                                   opt_minibatch_bernstein::Int64,
                                   opt_minibatch_heuristic::Int64,
                                   opt_minibatch_emp::Int64)

    probname = replace(replace(prob.name, r"[\/]", "-"), ".", "_");
    default_path = "./figures/";
    fontsmll = 8;
    fontmed = 12;
    fontbig = 14;
    xlabeltxt = "batchsize";
    ylabeltxt = "empirical total complexity";

    n = prob.numdata;
    d = prob.numfeatures;
    
    plot(minibatchlist, empcomplex, linestyle=:solid, 
         xlabel=xlabeltxt, ylabel=ylabeltxt, label="",
         xticks=(minibatchlist, minibatchlist),
         guidefont=font(fontbig), linewidth=4, grid=false, #marker=:auto,
         title=string("Pb: ", probname, ", n=", string(n), ", d=", string(d)));
    vline!([opt_minibatch_emp+0.04], line=(:auto, 3), color=1, label="empirical");
    vline!([opt_minibatch_simple-0.04], line=(:auto, 3), color=:red, label="simple", legendtitle="Optimal mini-batch size");
    vline!([opt_minibatch_bernstein-0.02], line=(:auto, 3), color=:black, label="bernstein");
    vline!([opt_minibatch_heuristic+0.02], line=(:auto, 3), color=:purple, label="heurstic");
    savename = "-empcomplex-$(numsimu)-avg";
    savefig("$(default_path)$(probname)$(savename).pdf");
end