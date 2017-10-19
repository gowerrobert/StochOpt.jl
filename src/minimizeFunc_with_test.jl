# A wrapper function for testing and timing iterative methods for
# solving the empirical risk minimization problem - 2016 - Robert M. Gower
# PseudoInvRand Copyright (C) 2016, Robert Gower
function  minimizeFunc_with_test(prob::Prob, testprob::Prob, method_name::AbstractString, options::MyOptions )
  method = boot_method(method_name,prob,options);
  if(method=="METHOD DOES NOT EXIST")
    println("FAIL: unknown method name:")
    return
  end
  println(method.name);
  times= [0];
  x =  zeros(prob.numfeatures); # initial point
  # println("size of X: ", size(X), " ", "prob.numdata ",prob.numdata, " length(1:prob.numdata): ",length(1:prob.numdata) )
  f0= test_error(testprob, x);
  fs = [f0];
  d = zeros(prob.numfeatures); # Search direction vector
  local tickcounter =1;
  local timeaccum=0;
  iterations =0;
  fail = "failed";
  ## Print heard
  if(options.printiters)
    println("-------------------")
    println("It   | testerror  |  datap  | Time   ")
    println("-------------------")
  end
  for iter= 1:options.maxiter
    tic();
    method.stepmethod(x,prob,options,method,iter,d);
    x[:] = x + method.stepsize * d;
    #  println("method.stepsize ", method.stepsize, "norm(d): ", norm(d) );
    timeaccum= timeaccum +  toq(); # Keeps track of time accumulated at every iteration

    if(mod(iter,options.skip_error_calculation)==0 )
      fs= [fs test_error(testprob, x); ];
      times = [ times   timeaccum];
      if(options.printiters)
        ## printing iterations info
        @printf "%3.0d  | %3.2f  |  %3.2f  | %3.4f |\n" iter 100*fs[end] iter*method.epocsperiter times[end] ;
      end
      if( fs[end] < options.tol)
        fail ="tol-reached"; iterations =iter;
        break;
      end
      if(fs[end]/f0 >1.10 ) # testing if method gone wild
        fail ="diverging"; iterations =iter;
        break;
      end
      if(isnan(sum(x)) || isnan(fs[end]) || fs[end]/f0 >1000  )
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
      fs= [fs test_error(testprob, x) ];
      times = [ times   timeaccum];
      if(options.printiters)
        @printf "%3.0d  | %3.2f  |  %3.2f  | %3.4f \n" iter 100*fs[end] iter*method.epocsperiter times[end] ;
      end
      break;
    end

  end # End of For loop
  outputname = method.name;
  output = Output(outputname,iterations,method.epocsperiter, [method.gradsperiter], fs.*f0, x,fail,  options.stepsize_multiplier); #./(f0.-prob.fsol)
  return output

end
