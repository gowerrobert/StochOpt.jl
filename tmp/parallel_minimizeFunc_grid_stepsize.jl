
function parallel_toy_1(n::Int64, m::Int64)
    println("i,j")
    array_val = SharedArray{Float64}(n, m);
    @sync @distributed for idx = 1:n*m
    # for idx = 1:n*m
        i, j = divrem(idx-1, m) .+ (1,1);
        println(i, ",", j, "  ");
        sleep(1);
        array_val[i,j] = idx;
    end
end

## Usefull trick to avoid having two nested for loop 
function parallel_toy_2(n::Int64, m::Int64)
    println("i,j")
    @sync @distributed for ii = 1:n*m
    # for ii = 1:n*m
        i, j = divrem(ii-1, m) .+ (1,1);
        println(i, ",", j, "  ")
    end
end


function parallel_toy_grid_search(prob::Prob, method_input, options::MyOptions; testprob=nothing)

    stepsizes = [2.0^(9), 2.0^(7), 2.0^(5), 2.0^(3), 2.0^(1), 2.0^(-1), 2.0^(-3), 2.0^(-5), 2.0^(-7), 2.0^(-9)];
    array_val = SharedArray{Float64}(options.rep_number, length(stepsizes));

    @sync @distributed for expnum = 1:options.rep_number
        @sync @distributed for stepind = 1:length(stepsizes)
            step = stepsizes[stepind];
            # println("\nBefore options.stepsize_multiplier: ", options.stepsize_multiplier);
            options.stepsize_multiplier *= 2;
            # println("\nAfter options.stepsize_multiplier: ", options.stepsize_multiplier);
            println("\nTrying stepsize ", step);
            # val = 1/step + rand();
            val = step;
            # println("Val ", val);
            array_val[expnum, stepind] = val;
        end
    end
    println("\nGrid of stepsizes\n", stepsizes)
    println("\nArray of values\n", array_val)
    # println("\nArray of medians\n", median(array_val, dims=1))

    ## Get the median over experiments as the best step size
    minval, bestidx = findmin(median(array_val, dims=1)[1,:])
    beststep = stepsizes[bestidx]
    
    # @printf "\nMinimum %5.5f reached for step = %f (idx=%d)\n\n" minval beststep bestidx
end


function parallel_minimizeFunc_grid_stepsize(prob::Prob, method_input, options::MyOptions; testprob=nothing)
    options.printiters = false;

    default_path = "./data/";

    if typeof(method_input) == String
        method_name = method_input;
    else
        method_name = method_input.name;
    end
    
    beststep, savename = get_saved_stepsize(prob.name, method_name, options);

    if beststep == 0.0 || options.repeat_stepsize_calculation == true
        options.force_continue = false;

        stepsizes = [2.0^(21), 2.0^(17), 2.0^(15), 2.0^(13), 2.0^(11), 2.0^(9), 2.0^(7), 2.0^(5), 
                     2.0^(3), 2.0^(1), 2.0^(-1), 2.0^(-3), 2.0^(-5), 2.0^(-7), 2.0^(-9), 2.0^(-11)];
        array_minval = SharedArray{Float64}(options.rep_number, length(stepsizes));

        @sync @distributed for expnum = 1:options.rep_number
            @sync @distributed for stepind = 1:length(stepsizes)
                step = stepsizes[stepind];
                println("\nTrying stepsize ", step);
                
                options.stepsize_multiplier = step;
                output = minimizeFunc(prob, method_input, options);

                array_minval[expnum, stepind] = output.fs[end];
            end
        end

        println("\nGrid of stepsizes\n", stepsizes)
        println("Array of values\n")
        for i=1:options.rep_number
            println(array_minval[i,:])
        end
        println("Array of medians\n", median(array_minval, dims=1))
    
        ## Get the median over experiments as the best step size
        minval, bestidx = findmin(median(array_minval, dims=1)[1,:])
        beststep = stepsizes[bestidx]
        
        @printf "\nMinimum %5.5f reached for step = %f (idx=%d)\n\n" minval beststep bestidx
    end

    options.printiters = true;
    options.force_continue = true;
    options.stepsize_multiplier = beststep;
    # options.max_time = 60.0*60.0;

    println("Best step: ", beststep);
    outputfirst = minimizeFunc(prob, method_input, options, testprob=testprob);
    
    # save("$(default_path)$(savename).jld", "output", outputfirst)
    
    return outputfirst
end
