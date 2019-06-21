function minimizeFunc_grid_stepsize(prob::Prob, method_input, options::MyOptions; testprob=nothing, grid::Array{Float64,1}=[0.0])
    default_path = "./data/";
    # savename = string(replace(prob.name, r"[\/]", "-"),'-',method_name_temp,"-",options.batchsize,"-stepsize") ;
    # savename = string(savename,'-',method_name_temp,"-",options.batchsize,"-stepsize") ;

    if typeof(method_input) == String
        method_name = method_input;
    else
        method_name = method_input.name;
    end

    beststep, savename = get_saved_stepsize(prob.name, method_name, options);
    if beststep == 0.0 || options.repeat_stepsize_calculation == true
        options.force_continue = false;
        if grid == [0.0]
            stepsizes = [2.0^(21), 2.0^(17), 2.0^(15), 2.0^(13), 2.0^(11), 2.0^(9), 2.0^(7), 2.0^(5),
                         2.0^(3), 2.0^(1), 2.0^(-1), 2.0^(-3), 2.0^(-5), 2.0^(-7), 2.0^(-9), 2.0^(-11)];
        else
            stepsizes = grid;
        end
        bestindx = length(stepsizes);
        beststeps_found = zeros(options.rep_number);
        start_step = 1;
        for expnum = 1:options.rep_number
            println("\n---------------------------------------------> Experiment # ", expnum);
            # minfval = 1.0; # Makes no sense, completely arabitrary value..
            minfval = Inf;
            thelastonebetter = 0;
            beststep = 0.0; #iteratesp = 1;
            for stepind = start_step:length(stepsizes)
                println("--------------------------------> minfval = ", minfval);
                recentmethods = [SVRG_vanilla_method, Free_SVRG_method, SAGA_nice_method] # reset if one of these
                if typeof(method_input) in recentmethods
                    # println("\n\nMONITORING STEPSIZE BEFORE RESET => ", method_input.stepsize, "\n\n")
                    # method_input = method_input.reset(prob, method_input, options); # old implementation
                    method_input.reset(prob, method_input, options); # /!\ new implementation with mutating fucntion. TO CHECK IF IT WORKS!!!
                    # println("\n\nMONITORING STEPSIZE AFTER RESET  => ", method_input.stepsize, "\n\n")
                end
                step = stepsizes[stepind];
                println("\nTrying stepsize ", step);
                options.stepsize_multiplier = step;


                output = minimizeFunc(prob, method_input, options);
                println("---> Fail = ", output.fail);
                if((output.fs[end] < minfval) && (output.fail == "max_time" || output.fail == "max_epocs" || output.fail == "tol-reached"))
                    println("found a better stepsize: ", beststep, " because ", minfval, " > ", output.fs[end])
                    minfval = output.fs[end];
                    beststep = step;
                    bestindx = something(findfirst(isequal(step), stepsizes), 0);
                    thelastonebetter = 1.0;
                elseif(thelastonebetter == 1.0)
                    println("the last one was better because ", minfval, " < ", output.fs[end], " = output.fs[end]")
                    break; #It's only getting worst by decreasing the stepsize.
                else
                    println("not better because ",minfval, " < ", output.fs[end], " = output.fs[end]" )
                    thelastonebetter = 0.0;
                end
                #iteratesp = iteratesp+1;
            end
            beststeps_found[expnum] = beststep;
            start_step = max(bestindx - 4, 1);
        end
        # Get the median over experiments as the best step size
        println("\nbest steps:")
        println(beststeps_found)
        beststep = mode(beststeps_found);
        # println("mode best step:")
        # println(beststep)
    end
    options.force_continue = true;
    options.stepsize_multiplier = beststep;

    options.max_epocs *= 2;
    options.max_time *= 3.0; # To adjust

    outputfirst = minimizeFunc(prob, method_input, options, testprob=testprob);
    # for expnum =2: options.rep_number
    #   outputfirst= minimizeFunc(prob, method_name, options); # Repeat a few times account for Julia just intime compiling
    # end
    println("\n=> Best step: ", beststep);

    save("$(default_path)$(savename).jld", "output", outputfirst);

    return outputfirst
end
