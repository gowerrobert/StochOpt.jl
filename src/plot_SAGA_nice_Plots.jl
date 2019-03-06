"""
    plot_expected_smoothness_bounds(prob::Prob, simplebound::Array{Float64}, bernsteinbound::Array{Float64}, heuristicbound::Array{Float64}, expsmoothcst ; showlegend::Bool=false)

Plots two upper-bounds of the expected smoothness constant (simple and Bernstein),
a heuristic estimation of it and its exact value (if there are few data points).

#INPUTS:\\
    - **Prob** prob: considered problem, i.e. logistic regression, ridge ression... (see src/StochOpt.jl)\\
    - nx1 **Array{Float64,2}** simplebound: simple bound of the expected smoothness constant for each mini-batch size ``τ`` from 1 to n\\
    - nx1 **Array{Float64,2}** bernsteinbound: Bernstein bound of the expected smoothness constant for each mini-batch size ``τ`` from 1 to n\\
    - nx1 **Array{Float64,2}** heuristic: heuristic estimation of the expected smoothness constant for each mini-batch size ``τ`` from 1 to n\\
    - nx1 **Array{Float64,2}** or **Nothing** expsmoothcst: exact expected smoothness constant for each mini-batch size ``τ`` from 1 to n\\
#OUTPUTS:\\
    - None
"""
function plot_expected_smoothness_bounds(prob::Prob, simplebound::Array{Float64}, bernsteinbound::Array{Float64}, heuristicbound::Array{Float64}, expsmoothcst ; showlegend::Bool=false)
    # PROBLEM: there is still a problem of ticking non integer on the xaxis

    probname = replace(replace(prob.name, r"[\/]" => "-"), "." => "_");
    default_path = "./figures/";
    fontsmll = 8;
    fontmed = 12;
    fontlegend = 13;
    fontbig = 14;
    xlabeltxt = "mini-batch size";

    labellist = [L"$L_\mathrm{practical}$" L"$L_\mathrm{simple}$" L"$L_\mathrm{Bernstein}$" L"$\mathcal{L}_1$"];
    colorlist = [:blue :orange :green :purple :black];
    markerlist = [:rect :circle :star5 :diamond :utriangle];
    linestylelist = [:solid :dash :dot :dashdot :auto];

    n = prob.numdata;
    d = prob.numfeatures;

    if typeof(expsmoothcst)==Array{Float64,2}
        plot(1:n, [heuristicbound simplebound bernsteinbound expsmoothcst],
             legend=showlegend,
             label=labellist,
             xlabel=xlabeltxt, ylabel="smoothness constant", tickfont=font(fontsmll), # xticks=1:n,
             guidefont=font(fontbig), legendfont=font(fontlegend), markersize=6, linewidth=3, grid=false,
             color=reshape(colorlist[1:4], 1, :),
            #  marker=reshape(markerlist[1:4], 1, :),
            #  linestyle=:auto,
             linestyle=reshape(linestylelist[1:4], 1, :)
            #  ,
             )
            #  ylim=(0, max(maximum(simplebound),maximum(bernsteinbound),maximum(heuristicbound))+minimum(expsmoothcst)))
            #  title=string(probname, ", n=", string(n), ", d=", string(d)))
    elseif typeof(expsmoothcst)==Nothing
        plot(1:n, [heuristicbound simplebound bernsteinbound],
             legend=showlegend,
             label=reshape(labellist[1:3], 1, :),
            #  yscale=:log10, # bug in julia 0.7
             xlabel=xlabeltxt, ylabel="smoothness constant", tickfont=font(fontsmll), # xticks=1:n,
             guidefont=font(fontbig), legendfont=font(fontlegend), linewidth=3, grid=false,
             color=reshape(colorlist[1:3], 1, :),
            #  marker=reshape(markerlist[1:3], 1, :),
            #  linestyle=:auto,
             linestyle=reshape(linestylelist[1:3], 1, :)
            #  ,
            )
            #  ylim=(0, max(maximum(simplebound),maximum(bernsteinbound),maximum(heuristicbound))+minimum(heuristicbound)))
            #  title=string(probname, ", n=", string(n), ", d=", string(d)));
    else
        error("Wrong type of expsmoothcst");
    end
    savename = "-exp1-expsmoothbounds";
    savefig("$(default_path)$(probname)$(savename).pdf");

    # Zoom
    if typeof(expsmoothcst)==Array{Float64,2}
        plot(1:n, [heuristicbound simplebound bernsteinbound expsmoothcst],
             legend=showlegend,
             label=labellist,
             xlabel=xlabeltxt, ylabel="smoothness constant", tickfont=font(fontsmll), #xticks=1:n,
             guidefont=font(fontbig), legendfont=font(fontlegend), markersize=6, linewidth=3, grid=false, # marker=:auto,
             color=reshape(colorlist[1:4], 1, :),
            #  marker=reshape(markerlist[1:4], 1, :),
            #  linestyle=:auto,
             linestyle=reshape(linestylelist[1:4], 1, :)
             ,
            # )
             ylim=(0.85*min(minimum(expsmoothcst), minimum(heuristicbound)), 1.5*max(simplebound[end], bernsteinbound[end], heuristicbound[end])));
            #  title=string(probname, ", n=", string(n), ", d=", string(d)," zoom"));
    elseif typeof(expsmoothcst)==Nothing
        plot(1:n, [heuristicbound simplebound bernsteinbound],
             legend=showlegend,
             label=reshape(labellist[1:3], 1, :),
             xlabel=xlabeltxt, ylabel="smoothness constant", tickfont=font(fontsmll), #xticks=1:n,
             guidefont=font(fontbig), legendfont=font(fontlegend), linewidth=3, grid=false,  #marker=:auto,
             color=reshape(colorlist[1:3], 1, :),
            #  marker=reshape(markerlist[1:3], 1, :),
            #  linestyle=:auto,
             linestyle=reshape(linestylelist[1:3], 1, :)
             ,
            # )
            #  ylim=(0.85*minimum(heuristicbound), 1.5*minimum(heuristicbound)),
             ylim=(0.85*minimum(heuristicbound), 1.5*max(simplebound[end], bernsteinbound[end], heuristicbound[end])));
            #  title=string(probname, ", n=", string(n), ", d=", string(d)," zoom"));
    else
        error("Wrong type of expsmoothcst");
    end
    savenamezoom = string(savename, "-zoom");
    savefig("$(default_path)$(probname)$(savenamezoom).pdf");
end


"""
    plot_stepsize_bounds(prob::Prob, simplestepsize::Array{Float64}, bernsteinstepsize::Array{Float64}, heuristicstepsize::Array{Float64}, expsmoothstepsize ; showlegend::Bool=false)

Plots upper bounds of the stepsizes corresponding to the simple and Bernstein upper bounds,
heuristic estimation and the exact expected smoothness constant (if there are few data points).

#INPUTS:\\
    - **Prob** prob: considered problem, i.e. logistic regression, ridge ression... (see src/StochOpt.jl)\\
    - nx1 **Array{Float64,2}** simplestepsize: simple bound stepsizes\\
    - nx1 **Array{Float64,2}** bernsteinstepsize: Bernstein bound stepsizes\\
    - nx1 **Array{Float64,2}** heuristicstepsize: heuristic estimation stepsizes\\
    - nx1 **Array{Float64,2}** hofmannstepsize: optimal step size given by Hofmann et. al. 2015\\
    - nx1 **Array{Float64,2}** or **Nothing** expsmoothstepsize: exact expected smoothness constant stepsize upper bound\\
#OUTPUTS:\\
    - None
"""
function plot_stepsize_bounds(prob::Prob, simplestepsize::Array{Float64}, bernsteinstepsize::Array{Float64},
                              heuristicstepsize::Array{Float64}, hofmannstepsize::Array{Float64}, expsmoothstepsize ; showlegend::Bool=false)
    # PROBLEM: there is still a problem of ticking non integer on the xaxis

    probname = replace(replace(prob.name, r"[\/]" => "-"), "." => "_");
    default_path = "./figures/";
    fontsmll = 8;
    fontmed = 12;
    fontlegend = 13;
    fontbig = 14;
    xlabeltxt = "mini-batch size";

    labellist = [L"$\gamma_\mathrm{practical}$" L"$\gamma_\mathrm{simple}$" L"$\gamma_\mathrm{Bernstein}$" L"$\gamma_{\mathcal{L}_1}$"];
    colorlist = [:blue :orange :green :purple :black];
    markerlist = [:rect :circle :star5 :diamond :utriangle];
    linestylelist = [:solid :dash :dot :dashdot :solid];
    seriestypelist = [:line :line :line :line :scatter];

    n = prob.numdata;
    d = prob.numfeatures;

    if n > 24
        freq = round(Int, n/20); # 20 points
    else
        freq = 1;
    end

    ymaxval = maximum(heuristicstepsize)+minimum(bernsteinstepsize);
    yminval = minimum(vcat(heuristicstepsize, simplestepsize, bernsteinstepsize, hofmannstepsize));
    ydelta = ymaxval - yminval;

    if typeof(expsmoothstepsize)==Array{Float64,2}
        p = plot(1:n, [heuristicstepsize simplestepsize bernsteinstepsize expsmoothstepsize],
                 legend=showlegend,
                 label=labellist,
                 xlabel=xlabeltxt, ylabel="step size", tickfont=font(fontsmll), # xticks=1:n,
                 guidefont=font(fontbig),
                 legendfont=font(fontlegend),
                 markersize=6,
                 linewidth=3,
                 grid=false, #'marker=:auto,'
                 color=reshape(colorlist[1:4], 1, :),
                #  marker=markerlist,
                #  linestyle=:auto,
                 linestyle=reshape(linestylelist[1:4], 1, :),
                 seriestype=reshape(seriestypelist[1:4], 1, :),
                 markerstrokecolor=:black
                )
                #  ,
                #  ylim=(-10*yminval, ymaxval));
        plot!(p, 1:freq:n, hofmannstepsize[1:freq:n],
              legend=showlegend,
              label=L"$\gamma_\mathrm{Hofmann}$",
              color=colorlist[end],
              linestyle=linestylelist[end],
              seriestype=seriestypelist[end])
             #  title=string(probname, ", n=", string(n), ", d=", string(d)));
    elseif typeof(expsmoothstepsize)==Nothing
        p = plot(1:n, [heuristicstepsize simplestepsize bernsteinstepsize],
                 legend=showlegend,
                 label=reshape(labellist[1:3], 1, :),
                 xlabel=xlabeltxt, ylabel="step size", tickfont=font(fontsmll), # xticks=1:n,
                 guidefont=font(fontbig), legendfont=font(fontlegend), markersize=6, linewidth=3, grid=false, #marker=:auto,
                #  color=reshape([colorlist[1:3]; colorlist[end]], 1, :),
                 color=reshape(colorlist[1:3], 1, :),
                #  marker=reshape([markerlist[1:3]; markerlist[end]], 1, :),
                #  linestyle=:auto,
                #  linestyle=reshape([linestylelist[1:3]; linestylelist[end]], 1, :),
                #  seriestype=reshape([seriestypelist[1:3]; seriestypelist[end]], 1, :),
                 linestyle=reshape(linestylelist[1:3], 1, :),
                 seriestype=reshape(seriestypelist[1:3], 1, :)
                #  ,
                )
                #  ylim=(-10*yminval, ymaxval))
                #  title=string(probname, ", n=", string(n), ", d=", string(d)));
        plot!(p, 1:freq:n, hofmannstepsize[1:freq:n],
              legend=showlegend,
              label=L"$\gamma_\mathrm{Hofmann}$",
              color=colorlist[end],
              linestyle=linestylelist[end],
              seriestype=seriestypelist[end]);
    else
        error("Wrong type of expsmoothstepsize");
    end
    savename = "-exp2-stepsizes";
    savefig(p, "$(default_path)$(probname)$(savename).pdf");
end

"""
    plot_empirical_complexity(prob::Prob, minibatchgrid::Array{Int64,1}, empcomplex::Array{Float64,1},
                              b_practical::Int64, b_empirical::Int64 ; path::AbstractString="./")

Saves the plot of the empirical total complexity.

#INPUTS:\\
    - **Prob** prob: considered problem, i.e. logistic regression, ridge ression... (see src/StochOpt.jl)\\
    - **Array{Int64,1}** minibatchgrid: list of the different mini-batch sizes\\
    - **Array{Float64,1}** empcomplex: average total complexity (tau*iteration complexity)
      for each of the mini-batch size (tau) over numsimu samples\\
    - **Int64** b_practical: heuristic optimal mini-batch size\\
    - **Int64** b_empirical: empirical optimal mini-batch size\\
    - **AbstractString** path: path to the folder where the plots are saved\\
#OUTPUTS:\\
    - None
"""
function plot_empirical_complexity(prob::Prob, minibatchgrid::Array{Int64,1}, empcomplex::Array{Float64,1},
                                   b_practical::Int64, b_empirical::Int64 ; path::AbstractString="./")
    numsimu = 1

    probname = replace(replace(prob.name, r"[\/]" => "-"), "." => "_")
    default_path = "./figures/"

    fontmed = 12
    fontbig = 15
    xlabeltxt = "mini-batch size"
    ylabeltxt = "empirical total complexity"

    n = prob.numdata
    d = prob.numfeatures

    labellist = [latexstring("\$b_\\mathrm{empirical} = $b_empirical\$"),
                 latexstring("\$b_\\mathrm{practical} \\; = $b_practical\$")]

    plot(minibatchgrid, empcomplex, linestyle=:solid, color=:black,
         xaxis=:log, yaxis=:log,
         xlabel=xlabeltxt, ylabel=ylabeltxt, label="",
         xticks=(minibatchgrid, minibatchgrid),
         xrotation = 45,
         tickfont=font(fontmed),
         guidefont=font(fontbig), linewidth=3, grid=false)
        #  title=string("Pb: ", probname, ", n=", string(n), ", d=", string(d)))
    vline!([b_empirical], line=(:dash, 3), color=:blue, label=labellist[1],
           legendfont=font(fontbig), legend=:best) #:legend
    #legendtitle="Optimal mini-batch size")
    vline!([b_practical], line=(:dot, 3), color=:red, label=labellist[2])
    savename = "-exp4-empcomplex-$(numsimu)-avg"
    savefig("$(default_path)$(probname)$(savename).pdf")
end