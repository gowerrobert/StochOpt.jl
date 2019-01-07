function get_saved_stepsize(probname::AbstractString, method_name::AbstractString, options)
    default_path = "./data/";
    savename = replace(probname, r"[\/]" => "-");
    if(method_name!="SVRG2")
        method_name_temp = replace(method_name, r"6" => "");
    else
        method_name_temp = method_name;
    end
    if(occursin("CM", method_name_temp) || occursin("AM", method_name_temp))
        savename = string(savename, '-', method_name_temp, "-", options.batchsize, "-", options.embeddim, "-stepsize") ;
    else
        # savename = string(savename, '-', method_name_temp, "-", options.batchsize, "-stepsize") ;
        savename = string(savename, '-', method_name_temp, "-stepsize");
    end
    beststep = 0.0;
    #repeat =1 means we should repeat all calculations even if there is a saved output already
    try
        output = load("$(default_path)$(savename).jld", "output");
        println("found ", "$(default_path)$(savename).jld with stepsize ", output.stepsize_multiplier)
        beststep = output.stepsize_multiplier;
    catch loaderror
        println(loaderror)
        # println("Calculating best stepsize for ", method_name, " on ", probname, " with batchsize ", options.batchsize)
        beststep = 0.0
    end
    return beststep, savename
end