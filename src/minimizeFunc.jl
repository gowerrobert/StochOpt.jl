# A wrapper function for testing and timing iterative methods for
# solving the empirical risk minimization problem - 2018 - Robert M. Gower
# StochOpt Copyright (C) 2018, Robert Gower
function minimizeFunc(prob::Prob, method_input, options::MyOptions; testprob=nothing, stop_at_tol::Bool=false)

    if(options.initial_point == "randn") # set initial point
        x = randn(prob.numfeatures);
    elseif(options.initial_point == "rand")
        x = rand(prob.numfeatures); #
    elseif(options.initial_point == "ones")
        x = ones(prob.numfeatures); #
    else
        x = zeros(prob.numfeatures); #
    end
    # println("---> Initial point set")

    if(typeof(method_input) == String)
        method = boot_method(method_input, prob, options);
        if(method=="METHOD DOES NOT EXIST")
            println("FAIL: unknown method name:")
            return
        end
    else
        # println("\n---Method is not a String---\n") # To try if this else is for SAGA
        method = method_input;
        method = method.bootmethod(prob, method, options); # Previous code
        # method = method.bootmethod(prob, method, options, x);
    end
    println(method.name);
    # println("---> Method set")

    times = [0];
    ##
    if(options.exacterror == false)
        prob.fsol = 0.0; # Using suboptimality as a measure of error
    else
        load_fsol!(options, prob); # already loaded in load_logistic.jl
    end
    # println("---> fsol set (", prob.fsol, ")")

    # load a pre-calculated best  solution
    # println("size of X: ", size(X), " ", "prob.numdata ",prob.numdata, " length(1:prob.numdata): ",length(1:prob.numdata) )
    f0 = prob.f_eval(x, 1:prob.numdata)
    fs = [f0];
    if(testprob != nothing)   # calculating the test error
        testerrors = [testerror(testprob, x)];
    else
        testerrors = [];
    end
    d = zeros(prob.numfeatures); # Search direction vector
    local tickcounter = 1;
    local timeaccum = 0;
    iterations = 0;
    fail = "failed";
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
        time_elapsed = @elapsed method.stepmethod(x, prob, options, method, iter, d);
        x[:] = x + method.stepsize * d;
        # println("method.stepsize ", method.stepsize); # Monitoring the stepsize value (for later implementation of line search)
        # println("method.stepsize ", method.stepsize, ", norm(d): ", norm(d));
        timeaccum = timeaccum + time_elapsed; # Keeps track of time accumulated at every iteration
        if(mod(iter, options.skip_error_calculation) == 0)
            fs = [fs prob.f_eval(x, 1:prob.numdata)];
            # println("fs[end] = ", fs[end]);
            if(testprob != nothing) # calculating the test error
                testerrors = [testerrors testerror(testprob, x)];
            end
            times = [times timeaccum];
            # println("fsol : ", prob.fsol)
            # if(options.printiters)   ## printing iterations info
            #     @printf "  %8.0d  |           %5.2f           |  %7.2f  |  %8.4f  |\n" iter 100*(fs[end]-prob.fsol)/(f0-prob.fsol) iter*method.epocsperiter times[end];
            # end
            if(options.printiters)
                if(options.exacterror == false)
                    @printf "  %8.0d  |           %5.20f           |  %7.2f  |  %8.4f  |\n" iter fs[end] iter*method.epocsperiter times[end];
                else
                    @printf "  %8.0d  |           %5.20f           |  %7.2f  |  %8.4f  |\n" iter 100*(fs[end]-prob.fsol)/(f0-prob.fsol) iter*method.epocsperiter times[end];
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
                fail = "nan";  iterations = iter;
                break;
            end
        end # End printing and function evaluation if
        if(timeaccum > options.max_time || iter*method.epocsperiter > options.max_epocs)
            if(timeaccum > options.max_time)
                fail = "max_time";
            elseif(iter*method.epocsperiter > options.max_epocs)
                fail = "max_epocs"
            end
            iterations = iter;
            fs = [fs prob.f_eval(x, 1:prob.numdata)];
            if(testprob != nothing)   # calculating the test error
                testerrors = [testerrors testerror(testprob, x)];
            end
            times = [times timeaccum];
            # if(options.printiters)
            #     @printf "  %8.0d  |           %5.2f           |  %7.2f  |  %8.4f  |\n" iter 100*(fs[end]-prob.fsol)/(f0-prob.fsol) iter*method.epocsperiter times[end];
            # end
            if(options.printiters)
                if(options.exacterror == false)
                    @printf "  %8.0d  |           %5.20f           |  %7.2f  |  %8.4f  |\n" iter fs[end] iter*method.epocsperiter times[end];
                else
                    @printf "  %8.0d  |           %5.20f           |  %7.2f  |  %8.4f  |\n" iter 100*(fs[end]-prob.fsol)/(f0-prob.fsol) iter*method.epocsperiter times[end];
                end
            end
            break;
        end
    end # End of For loop
    outputname = method.name;
    output = Output(outputname, iterations, method.epocsperiter, method.gradsperiter, times, fs, testerrors, x, fail, options.stepsize_multiplier); #./(f0.-prob.fsol)
    return output
end