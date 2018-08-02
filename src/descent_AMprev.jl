function descent_AMprev(x::Array{Float64},prob::Prob,options::MyOptions,mth::Method, N::Int64,d::Array{Float64})
  # AMprev: A 2nd variant of SVRG that uses an action matching Hessian estimate
  embeddim = Int64(options.aux);
  reminnerloop =N%mth.numinneriters;
  bucksize =  convert(Int64,floor(mth.numinneriters/embeddim))
  bucketnum =  convert(Int64,ceil(reminnerloop/bucksize)); #(reminnerloop/mth.numinneriters)
  bucketnum = convert(Int64,min(embeddim,bucketnum));
  if(bucketnum ==0) bucketnum=convert(Int64,embeddim); end

  if(N%mth.numinneriters ==0)# Reset reference point, grad estimate and Hessian estimate
    mth.prevx[:]  = x; # Stores previous x
    prob.Hess_opt!(x,1:prob.numdata,mth.S,mth.grad,mth.HS );
    mth.SHS[:] = pinv(full(((mth.S')*(mth.HS).+ 0.0001)^(1/2))); # pinv doesn't work for Symmetric matrices, need to use full matrices
    mth.S[:] =mth.S*mth.SHS;  #Add on last direction for embedding
    mth.HS[:] = mth.HS*mth.SHS;
    mth.Sold[:] = mth.S; #Stores the previous saved embedding space S
    mth.S[:] .= 0;
    d[:]= -mth.grad;
    mth.S[:,bucketnum] =   mth.S[:,bucketnum]+d[:]./bucksize;  #Add on last direction for embedding
  elseif(N>mth.numinneriters)     #SVRGkemb inner step
    sample!(1:prob.numdata,mth.ind,replace=false);
    mth.diffpnt[:] = x-mth.prevx;
    prob.Hess_opt!(mth.prevx,mth.ind,mth.Sold, mth.gradsamp, mth.HSi); # Stores H_i S
    mth.aux[:] = (mth.HS)'*mth.diffpnt;
    d[:] = mth.diffpnt-mth.Sold*mth.aux;
    d[:] = (mth.HS)*(mth.HSi'*d-mth.aux)+mth.HSi*mth.aux;
    d[:] =  d - prob.g_eval(x,mth.ind)+mth.gradsamp-mth.grad;
    # y =  (HS)'* v
   # 	d = v-Sy
   # 	d = HS(S' H_i d-y)	+ HiS*y
    mth.S[:,bucketnum] =   mth.S[:,bucketnum]+d[:]./bucksize; #calculating average step directions in current bucket
  else # SVRG
    sample!(1:prob.numdata,mth.ind,replace=false);
    d[:] =-prob.g_eval(x,mth.ind) + prob.g_eval(mth.prevx,mth.ind) -mth.grad;
    mth.S[:,bucketnum] =   mth.S[:,bucketnum]+d[:]./bucksize; #calculating average step directions in current bucket
  end
  # if(options.precondition && N>mth.numinneriters)
  #    Id = 10.0^(-10)*eye(size(mth.H)[1],size(mth.H)[1]);
  #    d[:] = d[:] - mth.Sold*inv(inv(mth.H+Id)+ mth.Sold'*mth.Sold+Id)*mth.Sold'*d;
  # end
end
