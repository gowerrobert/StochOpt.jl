function descent_CMprev(x::Array{Float64}, prob::Prob, options::MyOptions, mth::Method, N::Int64, d::Array{Float64})
    # SVRGkemb: A 2nd variant of SVRG that uses a embed Hessian matrix mth.S'*H*mth.S
    embeddim = Int64(options.aux);
    reminnerloop = N%mth.numinneriters;
    bucksize = convert(Int64, floor(mth.numinneriters/embeddim))
    bucketnum =  convert(Int64, ceil(reminnerloop/bucksize)); #(reminnerloop/mth.numinneriters)
    bucketnum = convert(Int64, min(embeddim, bucketnum));
    if(bucketnum == 0) bucketnum = convert(Int64, embeddim); end
    #  println("N,  reminnerloop, bucksize, bucketnum:",N, ", ",  reminnerloop, ", ", bucksize,", ", bucketnum)
    if(N%mth.numinneriters == 0)# Reset reference point, grad estimate and Hessian estimate
        mth.prevx[:] = x; # Stores previous x
        prob.Hess_opt!(x, 1:prob.numdata, mth.S, mth.grad, mth.HS);
        mth.H[:] = (mth.S')*mth.HS;
        #  Version Prev
        mth.Sold[:] = mth.S; #Stores the previous saved embedding space S
        mth.S[:] .= 0;
        d[:] = -mth.grad;
        mth.S[:,bucketnum] = mth.S[:,bucketnum] + d[:]./bucksize;  #Add on last direction for embedding
    elseif(N>mth.numinneriters) #SVRGkemb inner step
        sample!(1:prob.numdata, mth.ind, replace=false);
        mth.diffpnt[:] = x - mth.prevx;
        prob.Hess_opt!(mth.prevx, mth.ind, mth.Sold, d, mth.HSi);
        mth.aux[:] = mth.HS'*mth.diffpnt;
        mth.aux[:] = \(mth.H, mth.aux);  # mth.aux[:]\mth.H;
        d[:] = d - mth.HS*mth.aux;
        mth.aux[:] = (mth.HSi')*(mth.Sold*mth.aux[:]);
        mth.aux[:] = \(mth.H, mth.aux);#(mth.aux[:])\mth.H;
        d[:] = d + mth.HS*mth.aux;
    # In Latex
    #  S^\top H S z &= S^\top Hv \quad \mbox{(\bf Linear solve)}\\
    #  d & = - HS z\\
    #  S^\top H S  z &=  S^\top H_i S  z  \quad \mbox{(\bf Linear solve)}\\
    # d & = d + HSz
        d[:] = d - prob.g_eval(x, mth.ind) - mth.grad; # prob.g_eval(mth.prevx,mth.ind) has already been added on
        mth.S[:, bucketnum] = mth.S[:, bucketnum] + d[:]./bucksize; #calculating average step directions in current bucket
    else # SVRG
        sample!(1:prob.numdata, mth.ind, replace=false);
        d[:] = -prob.g_eval(x, mth.ind) + prob.g_eval(mth.prevx, mth.ind) - mth.grad;
        mth.S[:, bucketnum] = mth.S[:, bucketnum] + d[:]./bucksize; #calculating average step directions in current bucket
    end
    # if(options.precondition && N>mth.numinneriters)
    #    Id = 10.0^(-10)*eye(size(mth.H)[1],size(mth.H)[1]);
    #    d[:] = d[:] - mth.Sold*inv(inv(mth.H+Id)+ mth.Sold'*mth.Sold+Id)*mth.Sold'*d;
    # end
end