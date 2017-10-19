function  minimizeFunc_grid_stepsize(prob::Prob, method_name::AbstractString, options::MyOptions , repeat::Bool )
  default_path = "./data/";
  method_name_temp =   method_name;
  if(options.precondition)
    method_name_temp = string(method_name,"-qN");
  end
#  savename = string(replace(prob.name, r"[\/]", "-"),'-',method_name_temp,"-",options.batchsize,"-stepsize") ;
 #savename = string(savename,'-',method_name_temp,"-",options.batchsize,"-stepsize") ;
  beststep, savename = get_saved_stepsize(prob.name, method_name_temp,options);
  if(beststep ==0.0 ||  repeat==1)
    stepsizes = [2.0^(15) 2.0^(14) 2.0^(13) 2.0^(12) 2.0^(11) 2.0^(10) 2.0^(9) 2.0^(8) 2.0^(7) 2.0^(6) 2.0^(5) 2.0^(3) 2.0^(1) 2.0^(-1) 2.0^(-3)  2.0^(-5) 2.0^(-7) 2.0^(-9)  2.0^(-11) ];
    bestindx = length(stepsizes);
    beststeps_found = zeros(options.rep_number);
    start_step =1;
    for expnum =1: options.rep_number
      minfval = 1.0;  thelastonebetter =0;
      beststep = 0.0; #iteratesp=1;
      for stepind = start_step:length(stepsizes)
      #  println("Trying stepsize ", step);
        step = stepsizes[stepind];
        options.stepsize_multiplier =step
        output= minimizeFunc(prob, method_name, options);
        if(output.fs[end] < minfval && (output.fail =="max_time" || output.fail =="max_epocs" || output.fail =="tol-reached"))
          #println("found a better stepsize: ",beststep, " because ",minfval, " > ", output.fs[end], )
          minfval =output.fs[end];
          beststep = step;
          bestindx = findfirst(stepsizes,step);
          thelastonebetter=1.0;
        elseif(thelastonebetter==1.0)
        #  println("the last one was better because ",minfval, " < ", output.fs[end], "= output.fs[end]" )
          break; #It's only getting worst by decreasing the stepsize.
        else
        #  println("not better because ",minfval, " < ", output.fs[end], "= output.fs[end]" )
          thelastonebetter= 0.0;
        end
        #iteratesp = iteratesp+1;
      end
      beststeps_found[expnum] = beststep;
      start_step = max(bestindx-4,1);
    end
    # Get the median over experiments as the best step size
    println("best steps:")
    println(beststeps_found)
    beststep = mode(beststeps_found);
    println("median best step:")
    println(beststep)
   end
  options.force_continue = true;
  options.stepsize_multiplier =beststep;
  outputfirst= minimizeFunc(prob, method_name, options);
#  for expnum =1: options.rep_number
  outputfirst= minimizeFunc(prob, method_name, options); # Repeat twice to account for Julia just intime compiling
#  end
  save("$(default_path)$(savename).jld", "output",outputfirst)
  options.force_continue = false;
  #PyPlot PGFPlots Plotly GR
  # gr()# gr() pyplot() # pgfplots() #plotly()
  # plot_outputs_Plots([output],prob)
  return outputfirst
end
