# A wrapper function for testing and timing iterative methods for
# solving the empirical risk minimization problem - 2018 - Robert M. Gower
# StochOpt Copyright (C) 2018, Robert Gower
function  minimizeFunc(prob::Prob, method_name::AbstractString, options::MyOptions )
  method = boot_method(method_name,prob,options);
  if(method=="METHOD DOES NOT EXIST")
    println("FAIL: unknown method name:")
    return
  end
  println(method.name);
  times= [0];
  x =  zeros(prob.numfeatures); # initial point
  # println("size of X: ", size(X), " ", "prob.numdata ",prob.numdata, " length(1:prob.numdata): ",length(1:prob.numdata) )
  f0= prob.f_eval(x,1:prob.numdata)
  fs = [f0];
  d = zeros(prob.numfeatures); # Search direction vector
  local tickcounter =1;
  local timeaccum=0;
  iterations =0;
  fail = "failed";
  ## Print heard
  if(options.printiters)
    println("-------------------")
    println("It   | (f(x)-fsol)/(f0-fsol)  |  datap  | Time   ")
    println("-------------------")
  end
  for iter= 1:options.maxiter
    tic();
    method.stepmethod(x,prob,options,method,iter,d);
    x[:] = x + method.stepsize * d;
    #  println("method.stepsize ", method.stepsize, "norm(d): ", norm(d) );
    timeaccum= timeaccum +  toq(); # Keeps track of time accumulated at every iteration

    if(mod(iter,options.skip_error_calculation)==0 )
      if(options.exacterror)
        #println(options.exacterror);# if f* is given do something
      end
      fs= [fs prob.f_eval(x,1:prob.numdata) ];
      times = [ times   timeaccum];
      if(options.printiters)
        ## printing iterations info
        @printf "%3.0d  | %3.2f  |  %3.2f  | %3.4f |\n" iter 100*(fs[end]-prob.fsol)/(f0-prob.fsol) iter*method.epocsperiter times[end] ;
      end
      if((fs[end]-prob.fsol)/(f0-prob.fsol) < options.tol)
        fail ="tol-reached"; iterations =iter;
        break;
      end
      if(options.force_continue== false)
        if(fs[end]/f0 >1.10 || (fs[end]-prob.fsol)/(f0-prob.fsol) >1.1 || fs[end]/fs[end-1] >1.5 || (fs[end]-prob.fsol)/(fs[end-1]-prob.fsol) >1.15) # testing if method gone wild
          fail ="diverging"; iterations =iter;
          break;
        end
      end
      if(isnan(sum(x)) || isnan(fs[end])  || isinf(fs[end]) )
        fail = "nan";  iterations = iter;
        break;
      end
    end # End printing and function evaluation if
    if(timeaccum >options.max_time || iter*method.epocsperiter > options.max_epocs )
      if(timeaccum >options.max_time ) fail ="max_time";
      elseif(iter*method.epocsperiter > options.max_epocs ) fail = "max_epocs"
      end
      iterations = iter;
      if(options.exacterror)
        #println(options.exacterror)
      end
      fs= [fs prob.f_eval(x,1:prob.numdata) ];
      times = [ times   timeaccum];
      if(options.printiters)
        @printf "%3.0d  | %3.2f  |  %3.2f  | %3.4f \n" iter 100*(fs[end]-prob.fsol)/(f0-prob.fsol) iter*method.epocsperiter times[end] ;
      end
      break;
    end

  end # End of For loop

  outputname = method.name;
  output = Output(outputname,iterations,method.epocsperiter, method.gradsperiter, times, fs, x,fail,  options.stepsize_multiplier); #./(f0.-prob.fsol)
  return output

end
