function descent_CMcoord(x::Array{Float64},prob::Prob,options::MyOptions,mth::Method, N::Int64,d::Array{Float64})
# SVRGkemb: A 2nd variant of SVRG that uses a embed Hessian matrix mth.S'*H*mth.S
  embeddim = Int64(options.aux);

  if(N%mth.numinneriters ==0)# Reset reference point, grad estimate and Hessian estimate
    mth.prevx[:]  = x; # Stores previous x
    mth.S[:] =  sample(1:prob.numfeatures,embeddim,replace=false);
    C = convert(Array{Int64}, mth.S);
    prob.Hess_C!(x,1:prob.numdata, C, mth.grad, mth.HS);
    mth.H[:]  = pinv(mth.HS[C,:]);
    d[:]= -mth.grad;
  elseif(N>mth.numinneriters)     #SVRGkemb inner step
    sample!(1:prob.numdata,mth.ind,replace=false);
    mth.diffpnt[:] = x-mth.prevx;
    C = convert(Array{Int64}, mth.S);
    prob.Hess_C!(mth.prevx,mth.ind, C, d, mth.HSi);
    mth.SHS[:] = mth.HSi[C,:];
    mth.aux[:]=mth.HS'* mth.diffpnt;
    d[:] = d+ mth.HS*mth.H*(mth.SHS*mth.aux-mth.aux);
    d[:] = d-prob.g_eval(x,mth.ind)-mth.grad; # prob.g_eval(mth.prevx,mth.ind) has already been added on
  #  y  = S^\top H (\theta_t - \bar{\theta}_k)
  #  d  = HS (S^\top H S)^{\dagger} \left(S^\top H_i(\bar{\theta})Sy -  y\right)
  else # SVRG
    sample!(1:prob.numdata,mth.ind,replace=false);
    d[:] =-prob.g_eval(x,mth.ind) + prob.g_eval(mth.prevx,mth.ind) -mth.grad;
  end
  # if(options.precondition && N>mth.numinneriters)
  #    Id = 10.0^(-10)*eye(size(mth.H)[1],size(mth.H)[1]);
  #    d[:] = d[:] - mth.Sold*inv(inv(mth.H+Id)+ mth.Sold'*mth.Sold+Id)*mth.Sold'*d;
  # end
end
