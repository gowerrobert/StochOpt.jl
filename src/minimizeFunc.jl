# A wrapper function for testing and timing iterative methods for
# solving the empirical risk minimization problem - 2018 - Robert M. Gower
# StochOpt Copyright (C) 2018, Robert Gower

function minimizeFunc(prob::Prob, method_input, options::MyOptions; testprob=nothing, stop_at_tol::Bool=false, skip_decrease::Bool=false)

    if options.initial_point == "randn" # set initial point
        x = randn(prob.numfeatures);
    elseif(options.initial_point == "rand")
        x = rand(prob.numfeatures); #
    elseif(options.initial_point == "ones")
        x = ones(prob.numfeatures); #
    else
        x = zeros(prob.numfeatures); #
    end
    # println("---> Initial point set")

    if typeof(method_input) == String
        method = boot_method(method_input, prob, options);
        if method=="METHOD DOES NOT EXIST"
            println("FAIL: unknown method name:")
            return
        end
    else
        # println("\n---Method is not a String---\n") # To try if this else is for SAGA_nice or SVRG_nice
        method = method_input;
        # method = method.bootmethod(prob, method, options); # SAGA_nice
        method.bootmethod(prob, method, options); # SVRG_nice
        # method = method.bootmethod(prob, method, options, x); # ??
    end
    println(method.name);
    # println("---> Method set")

    times = [0];
    ##
    if options.exacterror == false
        prob.fsol = 0.0; # Using suboptimality as a measure of error
    else
        load_fsol!(options, prob); # already loaded in load_logistic.jl
    end
    # println("---> fsol set (", prob.fsol, ")")

    # Initiate the epochs vector
    output_epochs = Float64[0.0]

    # load a pre-calculated best  solution
    # println("size of X: ", size(X), " ", "prob.numdata ",prob.numdata, " length(1:prob.numdata): ",length(1:prob.numdata) )
    f0 = prob.f_eval(x, 1:prob.numdata);
    fs = [f0];
    if testprob != nothing  # calculating the test error
        testerrors = [testerror(testprob, x)];
    else
        testerrors = [];
    end
    d = zeros(prob.numfeatures); # Search direction vector
    # local tickcounter = 1; # What is this for?
    local timeaccum = 0;
    iterations = 0;
    fail = "failed";

    skip_error_calculation = options.skip_error_calculation;
    ## Print head
    # if(options.printiters)
    #     println("------------------------------------------------------------------")
    #     println("      It    | 100*(f(x)-fsol)/(f0-fsol) |   epochs  |     Time   |")
    #     println("------------------------------------------------------------------")
    # end
    if(options.printiters)
        if(options.exacterror == false)
            println("----------------------------------------------------------------------")
            println("      It    |                 f(x)          |   epochs  |     Time   |")
            println("----------------------------------------------------------------------")
        else
            println("--------------------------------------------------------------------------------------")
            println("      It    |           100*(f(x)-fsol)/(f0-fsol)           |   epochs  |     Time   |")
            println("--------------------------------------------------------------------------------------")
        end
    end
    for iter = 1:options.max_iter
        if iter == 1
            ## "Warm up" to avoid error on the elpased time at the first iteration of the first launch: does not work
            # println("Warm up")
            time_elapsed = @elapsed method.stepmethod(x, prob, options, method, iter, d);

            ## Resetting back the method
            if typeof(method_input) == String
                method = boot_method(method_input, prob, options);
                if method=="METHOD DOES NOT EXIST"
                    println("FAIL: unknown method name:")
                    return
                end
            else
                method = method_input
                println("\nIn minimizeFunc number_computed_gradients: ", method.number_computed_gradients)
                method.reset(prob, method, options)
                method.bootmethod(prob, method, options) # SAGA_nice and SVRG implementations
                println("In minimizeFunc number_computed_gradients: ", method.number_computed_gradients,"\n")
            end
            d = zeros(prob.numfeatures); # Search direction vector
            fail = "failed";
        end

        ## Taking a step
        time_elapsed = @elapsed method.stepmethod(x, prob, options, method, iter, d) # mutating function
        x[:] = x + method.stepsize * d

        # println("method.stepsize ", method.stepsize); # Monitoring the stepsize value (for later implementation of line search)
        # println("method.stepsize ", method.stepsize, ", norm(d): ", norm(d));

        timeaccum += time_elapsed # Keeps track of time accumulated at every iteration

        # Monitor the number of computed gradient in an attribute of the method
        # if clause to compute the correct number of epochs
        if method.epocsperiter == 0
            epochs = method.number_computed_gradients[end]/prob.numdata
        else
            epochs = iter*method.epocsperiter
        end

        if mod(iter, skip_error_calculation) == 0
            output_epochs = [output_epochs epochs] # saves the epochs at which the error is computed
            fs = [fs prob.f_eval(x, 1:prob.numdata)]
            # println("fs[end] = ", fs[end]);
            if(testprob != nothing) # calculating the test error
                testerrors = [testerrors testerror(testprob, x)];
            end
            times = [times timeaccum];
            # println("fsol : ", prob.fsol)
            # if(options.printiters)   ## printing iterations info
            #     @printf "  %8.0d  |           %5.2f           |  %7.2f  |  %8.4f  |\n" iter 100*(fs[end]-prob.fsol)/(f0-prob.fsol) iter*method.epocsperiter times[end];
            # end
            if options.printiters
                if options.exacterror == false
                    @printf "  %8.0d  |           %5.20f           |  %7.2f  |  %8.4f  |\n" iter fs[end] epochs times[end]
                else
                    @printf "  %8.0d  |           %5.20f           |  %7.2f  |  %8.4f  |\n" iter 100*(fs[end]-prob.fsol)/(f0-prob.fsol) epochs times[end]
                end
            end
            if options.force_continue == false || stop_at_tol # CHANGE ALL "if var == false" with "if !var"
                if((fs[end]-prob.fsol)/(f0-prob.fsol) < options.tol)
                    fail = "tol-reached";
                    iterations = iter;
                    break;
                end
            end
            if(options.force_continue == false)
                # if((fs[end]-prob.fsol)/(f0-prob.fsol) < options.tol)
                #     fail = "tol-reached";
                #     iterations = iter;
                #     break;
                # end
                # testing if method gone wild
                # if(fs[end]/f0 > 1.10 || (fs[end]-prob.fsol)/(f0-prob.fsol) > 1.1 || fs[end]/fs[end-1] > 1.5 || (fs[end]-prob.fsol)/(fs[end-1]-prob.fsol) > 1.15) # testing if method gone wild
                #     println("DIV 0");
                #     fail = "diverging";
                #     iterations = iter;
                #     break;
                # end
                if(fs[end]/f0 > 1.10)
                    println("DIV 1");
                    fail = "diverging"; iterations = iter;
                    break;
                elseif((fs[end]-prob.fsol)/(f0-prob.fsol) > 1.1)
                    println("DIV 2");
                    fail = "diverging"; iterations = iter;
                    break;
                elseif(fs[end]/fs[end-1] > 1.5)
                    println("DIV 3");
                    fail = "diverging"; iterations = iter;
                    break;
                elseif((fs[end]-prob.fsol)/(fs[end-1]-prob.fsol) > 10^2) # previous threshold at 1.15
                    println("DIV 4");
                    fail = "diverging"; iterations = iter;
                    break;
                end
            end
            if(isnan(sum(x)) || isnan(fs[end]) || isinf(fs[end]))
                println("DIV NAN");
                fail = "nan";  iterations = iter;
                break;
            end

            if ((fs[end]-prob.fsol)/(f0-prob.fsol) < 2*options.tol) && skip_decrease
                println("Decreasing the skip_error parameter");
                println("--------------- Before: ", skip_error_calculation);
                skip_error_calculation = round(Int, skip_error_calculation/2);
                println("--------------- After: ", skip_error_calculation);
                skip_decrease = false;
            end
        end # End printing and function evaluation if

        if timeaccum > options.max_time || epochs > options.max_epocs
            if(timeaccum > options.max_time)
                fail = "max_time";
            elseif epochs > options.max_epocs
                fail = "max_epocs"
            end
            iterations = iter
            output_epochs = [output_epochs epochs] # saves the epochs at which the error is computed
            fs = [fs prob.f_eval(x, 1:prob.numdata)]
            if(testprob != nothing)   # calculating the test error
                testerrors = [testerrors testerror(testprob, x)]
            end
            times = [times timeaccum];
            # if(options.printiters)
            #     @printf "  %8.0d  |           %5.2f           |  %7.2f  |  %8.4f  |\n" iter 100*(fs[end]-prob.fsol)/(f0-prob.fsol) iter*method.epocsperiter times[end]
            # end
            if options.printiters
                println("\n-------------------------------- End of minimization ---------------------------------")
                if options.exacterror == false
                    @printf "  %8.0d  |           %5.20f           |  %7.2f  |  %8.4f  |\n" iter fs[end] epochs times[end]
                else
                    @printf "  %8.0d  |           %5.20f           |  %7.2f  |  %8.4f  |\n" iter 100*(fs[end]-prob.fsol)/(f0-prob.fsol) epochs times[end]
                end
            end
            break
        end
    end # End of For loop
    outputname = method.name
    # output = Output(outputname, iterations, method.epocsperiter, method.gradsperiter, times, fs, testerrors, x, fail, options.stepsize_multiplier); #./(f0.-prob.fsol) # old
    # output = Output(outputname, iterations, method.epocsperiter, method.gradsperiter, times, fs, testerrors, x, fail, method.stepsize); # taking step size from method not options

    if occursin("Leap-SVRG", method.name)
        # reseting method step size to stochastic step size because the gradient step size is always 1/L
        method.stepsize = method.stochastic_stepsize
    end
    output = Output(outputname, iterations, method.epocsperiter, method.gradsperiter, output_epochs, times, fs, testerrors, x, fail, method.stepsize) # Adding the number of computed gradients

    return output
end