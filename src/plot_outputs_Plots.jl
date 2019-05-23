# Front end for plotting the execution in time and in flops of the outputs recorded in OUTPUTS.
function plot_outputs_Plots(OUTPUTS, prob::Prob, options ; datapassbnd::Int64=0, methodname::AbstractString="", suffix::AbstractString="", path::AbstractString="./", legendpos::Symbol=:best, legendtitle::AbstractString="", legendfont::Int64=12, legendabove::Bool=false, nolegend::Bool=false) #, datapassbnd::Int64
    ## Now in epocs X function values

    # New version with method name
    probname = string(replace(prob.name, r"[\/]." => "-"));
    if length(methodname) > 0
        probname = string(probname, "-", methodname);
    end
    if length(suffix) > 0
        probname = string(probname, suffix);
    end
    probname = replace(probname, "." => "_");

    if(options.precondition)
        probname = string(probname, "-precon")
    end
    data_path = "data/";  # savename= string(replace(prob.name, r"[\/]" => "-"),"-", options.batchsize);
    save("$(path)$(data_path)$(probname).jld", "OUTPUTS", OUTPUTS);
    fontsmll = 8;
    fontmed = 12;
    fontbig = 14;
    xlabeltxt = "epochs";

    ## Plotting epochs
    output = OUTPUTS[1];

    if output.epochs == [0.0]
        number_epochs = output.iterations*output.epocsperiter
        lf_all = length(output.fs)
        epochs = (number_epochs/(lf_all-1)).*(0:(lf_all-1))
    else
        number_epochs = output.epochs[end]
        epochs = output.epochs
        # println("\nIn plot_outputs_Plots output.epochs: ", output.epochs, "\n")
    end

    # Select the desired number of epochs if "datapassbnd" is gvien
    truncatefigure = true
    if datapassbnd == 0 # Setting the datapassbnd to the number of datapasses available
        truncatefigure = false
        datapassbnd = number_epochs
    end

    rel_loss = (output.fs.-prob.fsol)./(output.fs[1].-prob.fsol) # the relative loss might be negative if we reach a better solution
    fs = output.fs[rel_loss.>0]
    lf = length(fs)
    # bnd = convert(Int64, min(ceil(datapassbnd*lf/(output.iterations*output.epocsperiter)), lf))
    bnd = convert(Int64, min(ceil(datapassbnd*lf/number_epochs), lf))

    plt = plot(epochs[1:bnd], (fs[1:bnd].-prob.fsol)./(fs[1].-prob.fsol),
               xlabel=xlabeltxt, ylabel="residual", yscale=:log10, label=output.name,
               linestyle=:auto, tickfont=font(fontsmll), guidefont=font(fontbig),
               legendfont=font(legendfont), legend=legendpos, legendtitle=legendtitle,
               markersize=6, linewidth=4, marker=:auto, grid=false)
    min_value = minimum((fs[1:bnd].-prob.fsol)./(fs[1].-prob.fsol))
    for j=2:length(OUTPUTS)
        output = OUTPUTS[j];

        if output.epochs == [0.0]
            number_epochs = output.iterations*output.epocsperiter
            lf_all = length(output.fs)
            epochs = (number_epochs/(lf_all-1)).*(0:(lf_all-1))
        else
            number_epochs = output.epochs[end]
            epochs = output.epochs
            # println("\nIn plot_outputs_Plots output.epochs: ", output.epochs, "\n")
        end

        if !truncatefigure # Setting the datapassbnd to the maximum number of epochs if no truncation is given
            datapassbnd = number_epochs
        end

        rel_loss = (output.fs.-prob.fsol)./(output.fs[1].-prob.fsol) # the relative loss might be negative if we reach a better solution
        fs = output.fs[rel_loss.>0]
        lf = length(fs)
        # bnd = convert(Int64, min(ceil(datapassbnd*lf/(output.iterations*output.epocsperiter)), lf))
        bnd = convert(Int64, min(ceil(datapassbnd*lf/number_epochs), lf))

        plot!(plt, epochs[1:bnd], (fs[1:bnd].-prob.fsol)./(fs[1].-prob.fsol),
              xlabel=xlabeltxt, ylabel="residual", yscale=:log10, label=output.name, linestyle=:auto, tickfont=font(fontsmll),
              guidefont=font(fontbig), legendfont=font(legendfont), markersize=6, linewidth=4, marker=:auto, grid=false)
        if min_value > minimum((fs[1:bnd].-prob.fsol)./(fs[1].-prob.fsol))
            min_value = minimum((fs[1:bnd].-prob.fsol)./(fs[1].-prob.fsol))
        end
    end
    if legendabove
        ylims!(min_value - 0.5*min_value , 300)
    end
    if nolegend
        plot!(legend = nothing)
    end
    println("$(path)figures/$(probname)-epoc.pdf")
    savefig(plt, "$(path)figures/$(probname)-epoc.pdf")

    ## Plotting times
    output = OUTPUTS[1]

    if output.epochs == [0.0]
        number_epochs = output.iterations*output.epocsperiter
    else
        number_epochs = output.epochs[end]
    end
    datapassbnd = number_epochs # no truncation option available for time
    rel_loss = (output.fs.-prob.fsol)./(output.fs[1].-prob.fsol);
    fs = output.fs[rel_loss.>0];
    lf = length(fs);
    bnd = convert(Int64, min(ceil(datapassbnd*lf/number_epochs), lf));
    plot(output.times[1:bnd], (fs[1:bnd].-prob.fsol)./(fs[1].-prob.fsol), xlabel="time", ylabel="residual", yscale=:log10, label=output.name,
         linestyle=:auto, tickfont=font(fontsmll), guidefont=font(fontbig), legendfont=font(legendfont),
         legend=legendpos, legendtitle=legendtitle, markersize=6, linewidth=4, marker=:auto, grid=false)
    println(output.name, ": 2^", log(2, output.stepsize_multiplier))
    min_value = minimum((fs[1:bnd].-prob.fsol)./(fs[1].-prob.fsol))
    for jiter=2:length(OUTPUTS)
        output = OUTPUTS[jiter]

        if output.epochs == [0.0]
            number_epochs = output.iterations*output.epocsperiter
        else
            number_epochs = output.epochs[end]
        end
        datapassbnd = number_epochs
        rel_loss = (output.fs.-prob.fsol)./(output.fs[1].-prob.fsol);
        fs = output.fs[rel_loss.>0];
        lf = length(fs);
        bnd = convert(Int64, min(ceil(datapassbnd*lf/number_epochs), lf));
        plot!(output.times[1:bnd], (fs[1:bnd].-prob.fsol)./(fs[1].-prob.fsol), xlabel="time", ylabel="residual", yscale=:log10, label=output.name, linestyle=:auto, tickfont=font(fontsmll),
            guidefont=font(fontbig), legendfont=font(legendfont), markersize=6, linewidth=4, marker=:auto, grid=false)
        println(output.name,": 2^", log(2, output.stepsize_multiplier))
        if min_value > minimum((fs[1:bnd].-prob.fsol)./(fs[1].-prob.fsol))
            min_value = minimum((fs[1:bnd].-prob.fsol)./(fs[1].-prob.fsol))
        end
    end
    if legendabove
        ylims!(min_value - 0.5*min_value , 300)
    end
    if nolegend
        plot!(legend = nothing)
    end
    println("$(path)figures/$(probname)-time.pdf")
    savefig("$(path)figures/$(probname)-time.pdf")

    ## Plot test error as well
    if(!isempty(OUTPUTS[1].testerrors))
        output = OUTPUTS[1]

        if output.epochs == [0.0]
            number_epochs = output.iterations*output.epocsperiter
        else
            number_epochs = output.epochs[end]
        end
        datapassbnd = number_epochs
        lf = length(output.testerrors);
        bnd = convert(Int64, min(ceil(datapassbnd*lf/number_epochs), lf));
        plot(output.times[1:bnd], output.testerrors[1:bnd], xlabel="time", ylabel="residual", label=string(output.name, "-t"), linestyle=:auto,
             tickfont=font(fontsmll), guidefont=font(fontbig), legendfont=font(legendfont), legend=legendpos, legendtitle=legendtitle, markersize=6, linewidth=4, marker=:auto, grid=false)
        println(output.name,": 2^", log(2, output.stepsize_multiplier))
        println(output.testerrors[1:bnd])
        min_value = minimum((fs[1:bnd].-prob.fsol)./(fs[1].-prob.fsol))
        for jiter=2:length(OUTPUTS)
            output = OUTPUTS[jiter]

            if output.epochs == [0.0]
                number_epochs = output.iterations*output.epocsperiter
            else
                number_epochs = output.epochs[end]
            end
            datapassbnd = number_epochs
            lf = length(output.testerrors);
            bnd = convert(Int64, min(ceil(datapassbnd*lf/number_epochs), lf));
            plot!(output.times[1:bnd], output.testerrors[1:bnd], xlabel="time", ylabel="residual", label=string(output.name, "-t"), linestyle=:auto, tickfont=font(fontsmll), guidefont=font(fontbig),
                legendfont=font(legendfont), markersize=6, linewidth=4, marker=:auto, grid=false)
            println(output.testerrors[1:bnd])
            println(output.name, ": 2^", log(2, output.stepsize_multiplier))
            if min_value > minimum((fs[1:bnd].-prob.fsol)./(fs[1].-prob.fsol))
                min_value = minimum((fs[1:bnd].-prob.fsol)./(fs[1].-prob.fsol))
            end
        end
        if legendabove
            ylims!(min_value - 0.5*min_value , 300)
        end
        if nolegend
            plot!(legend = nothing)
        end
        println(probname)
        savefig("$(path)figures/$(probname)-t-time.pdf")
    end

    open("$(path)figures/$(probname)-stepsizes.txt", "w") do f
        write(f, "$(probname) stepsize_multiplier \n")
        for i=1:length(OUTPUTS)
            output = OUTPUTS[i];
            loofstep = log(2, output.stepsize_multiplier);
            outname = output.name;
            write(f, "$(outname) : 2^ $(loofstep)\n")
        end
    end
end


# plotting gradient computations per iteration
# output = OUTPUTS[1];
# lf = length(output.fs);
# bnd = convert(Int64,min(ceil(datapassbnd*lf/(output.iterations*output.epocsperiter)),lf));
# plot(output.gradsperiter*(1:bnd)*(output.iterations/lf),(output.fs[1:bnd].-prob.fsol)./(output.fs[1].-prob.fsol),
# xlabel = "grads",
# ylabel = "residual",
# yscale = :log10,
# label  = output.name,
# linestyle=:auto,   tickfont=font(12), guidefont=font(18), legendfont =font(12),  markersize = 8, linewidth =4,
# marker =:auto,  grid = false)
# println(output.name,": 2^", log(2,output.stepsize_multiplier))
# for jiter =2:length(OUTPUTS)
#   output = OUTPUTS[jiter];
#   lf = length(output.fs);
#   bnd = convert(Int64,min(ceil(datapassbnd*lf/(output.iterations*output.epocsperiter)),lf));
#   plot!(output.gradsperiter*(1:bnd)*(output.iterations/lf),(output.fs[1:bnd].-prob.fsol)./(output.fs[1].-prob.fsol), yscale = :log10,
#   label  = output.name, linestyle=:auto, marker =:auto, grid = false, markersize = 8, linewidth =4)
#   println(output.name,": 2^", log(2,output.stepsize_multiplier))
# end
# println(probname)
# savefig("./figures/$(probname)-grads.pdf");
