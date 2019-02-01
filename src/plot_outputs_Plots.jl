# Front end for plotting the execution in time and in flops of the outputs recorded in OUTPUTS.
function plot_outputs_Plots(OUTPUTS, prob::Prob, options ; datapassbnd::Int64=0, suffix::AbstractString="") #, datapassbnd::Int64
    #Now in epocs X function values
    if length(suffix) > 0
        probname = string(replace(prob.name, r"[\/]." => "-"), suffix);
    elseif options.batchsize > 0
        probname = string(replace(prob.name, r"[\/]." => "-"), "-", options.batchsize);
    else
        probname = string(replace(prob.name, r"[\/]." => "-"));
    end
    probname = replace(probname, "." => "_")
    if(options.precondition)
        probname = string(probname, "-precon")
    end
    default_path = "./data/";  # savename= string(replace(prob.name, r"[\/]" => "-"),"-", options.batchsize);
    save("$(default_path)$(probname).jld", "OUTPUTS", OUTPUTS);
    fontsmll = 8;
    fontmed = 12;
    fontbig = 14;
    xlabeltxt = "epochs";

    # plotting epocs per iteration
    output = OUTPUTS[1];

    # Select the desired number of epochs if "datapassbnd" is gvien
    truncatefigure = true;
    if (datapassbnd == 0) # Setting the datapassbnd to the number of datapasses available
        truncatefigure = false;
        datapassbnd = output.iterations*output.epocsperiter;
    end

    numepochs = output.iterations*output.epocsperiter;
    lf_all = length(output.fs);
    epochs = (numepochs/(lf_all-1)).*(0:(lf_all-1));

    rel_loss = (output.fs.-prob.fsol)./(output.fs[1].-prob.fsol); # the relative loss might be negative if we reach a better solution
    fs = output.fs[rel_loss.>0];
    lf = length(fs);
    bnd = convert(Int64, min(ceil(datapassbnd*lf/(output.iterations*output.epocsperiter)), lf));
    plt = plot(epochs[1:bnd], (fs[1:bnd].-prob.fsol)./(fs[1].-prob.fsol),
               xlabel=xlabeltxt, ylabel="residual", yscale=:log10, label=output.name,
               linestyle=:auto, tickfont=font(fontsmll), guidefont=font(fontbig), legendfont=font(fontmed),
               markersize=6, linewidth=4, marker=:auto, grid=false)
    for j=2:length(OUTPUTS)
        output = OUTPUTS[j];
        if (!truncatefigure) # Setting the datapassbnd to the maximum number of epochs if no truncation is given
            datapassbnd = output.iterations*output.epocsperiter;
        end

        numepochs = output.iterations*output.epocsperiter;
        lf_all = length(output.fs);
        epochs = (numepochs/(lf_all-1)).*(0:(lf_all-1));

        rel_loss = (output.fs.-prob.fsol)./(output.fs[1].-prob.fsol); # the relative loss might be negative if we reach a better solution
        fs = output.fs[rel_loss.>0];
        lf = length(fs);
        bnd = convert(Int64, min(ceil(datapassbnd*lf/(output.iterations*output.epocsperiter)), lf));
        plot!(plt, epochs[1:bnd], (fs[1:bnd].-prob.fsol)./(fs[1].-prob.fsol),
              xlabel=xlabeltxt, ylabel="residual", yscale=:log10, label=output.name, linestyle=:auto, tickfont=font(fontsmll),
              guidefont=font(fontbig), legendfont=font(fontmed), markersize=6, linewidth=4, marker=:auto,  grid=false)
    end
    println("./figures/$(probname)-epoc.pdf");
    savefig(plt, "./figures/$(probname)-epoc.pdf");


    # plotting times
    output = OUTPUTS[1];
    datapassbnd = output.iterations*output.epocsperiter; # no truncation option available for time
    rel_loss = (output.fs.-prob.fsol)./(output.fs[1].-prob.fsol);
    fs = output.fs[rel_loss.>0];
    lf = length(fs);
    bnd = convert(Int64, min(ceil(datapassbnd*lf/(output.iterations*output.epocsperiter)), lf));
    plot(output.times[1:bnd], (fs[1:bnd].-prob.fsol)./(fs[1].-prob.fsol), xlabel="time", ylabel="residual", yscale=:log10, label=output.name, linestyle=:auto, tickfont=font(fontsmll),
        guidefont=font(fontbig), legendfont=font(fontmed), markersize=6, linewidth=4, marker=:auto, grid=false)
    println(output.name, ": 2^", log(2, output.stepsize_multiplier))
    for jiter=2:length(OUTPUTS)
        output = OUTPUTS[jiter];
        datapassbnd = output.iterations*output.epocsperiter;
        rel_loss = (output.fs.-prob.fsol)./(output.fs[1].-prob.fsol);
        fs = output.fs[rel_loss.>0];
        lf = length(fs);
        bnd = convert(Int64, min(ceil(datapassbnd*lf/(output.iterations*output.epocsperiter)), lf));
        plot!(output.times[1:bnd], (fs[1:bnd].-prob.fsol)./(fs[1].-prob.fsol), xlabel="time", ylabel="residual", yscale=:log10, label=output.name, linestyle=:auto, tickfont=font(fontsmll),
            guidefont=font(fontbig), legendfont=font(fontmed), markersize=6, linewidth=4, marker=:auto, grid=false)
        println(output.name,": 2^", log(2,output.stepsize_multiplier))
    end
    println("./figures/$(probname)-time.pdf");
    savefig("./figures/$(probname)-time.pdf");


    if(!isempty(OUTPUTS[1].testerrors)) # plot test error as well
        output = OUTPUTS[1];
        datapassbnd = output.iterations*output.epocsperiter;
        lf = length(output.testerrors);
        bnd = convert(Int64, min(ceil(datapassbnd*lf/(output.iterations*output.epocsperiter)), lf));
        plot(output.times[1:bnd], output.testerrors[1:bnd], xlabel="time", ylabel="residual", label=string(output.name, "-t"), linestyle=:auto, tickfont=font(fontsmll), guidefont=font(fontbig),
            legendfont=font(fontmed), markersize=6, linewidth=4, marker=:auto, grid=false)
        println(output.name,": 2^", log(2, output.stepsize_multiplier))
        println(output.testerrors[1:bnd])
        for jiter=2:length(OUTPUTS)
            output = OUTPUTS[jiter];
            datapassbnd = output.iterations*output.epocsperiter;
            lf = length(output.testerrors);
            bnd = convert(Int64, min(ceil(datapassbnd*lf/(output.iterations*output.epocsperiter)), lf));
            plot!(output.times[1:bnd], output.testerrors[1:bnd], xlabel="time", ylabel="residual", label=string(output.name, "-t"), linestyle=:auto, tickfont=font(fontsmll), guidefont=font(fontbig),
                legendfont=font(fontmed), markersize=6, linewidth=4, marker=:auto, grid=false)
            println(output.testerrors[1:bnd])
            println(output.name, ": 2^", log(2, output.stepsize_multiplier))
        end
        println(probname)
        savefig("./figures/$(probname)-t-time.pdf");
    end

    open("./figures/$(probname)-stepsizes.txt", "w") do f
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
