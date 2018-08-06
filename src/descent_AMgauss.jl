function descent_AMgauss(x::Array{Float64}, prob::Prob, options::MyOptions, mth::Method, N::Int64, d::Array{Float64})
    # SVRGkemb: A 2nd variant of SVRG that uses a embed Hessian matrix mth.S'*H*mth.S
    embeddim = Int64(options.aux[1]);

    if(N%mth.numinneriters == 0)# Reset reference point, grad estimate and Hessian estimate
        mth.prevx[:] = x; # Stores previous x
        mth.S[:] = rand(prob.numfeatures, embeddim);  #Add on last direction for embedding
        prob.Hess_opt!(x, 1:prob.numdata, mth.S, mth.grad, mth.HS); # Calculates gradient and Hessian vector products
        mth.SHS[:] = pinv(((mth.S')*(mth.HS))^(1/2));
        mth.S[:] = mth.S*mth.SHS;  #Add on last direction for embedding
        mth.HS[:] = mth.HS*mth.SHS;
        d[:] = -mth.grad;
    elseif(N > mth.numinneriters)     #SVRGkemb inner step
        sample!(1:prob.numdata, mth.ind, replace=false);
        mth.diffpnt[:] = x-mth.prevx;
        prob.Hess_opt!(mth.prevx, mth.ind, mth.S, mth.gradsamp, mth.HSi); # Stores H_i S
        mth.aux[:] = (mth.HS)'*mth.diffpnt;
        d[:] = mth.diffpnt - mth.S*mth.aux;
        d[:] = (mth.HS)*(mth.HSi'*d-mth.aux) + mth.HSi*mth.aux;
        d[:] = d - prob.g_eval(x, mth.ind) + mth.gradsamp - mth.grad;
    else # SVRG
        sample!(1:prob.numdata, mth.ind, replace=false);
        d[:] = -prob.g_eval(x, mth.ind) + prob.g_eval(mth.prevx, mth.ind) - mth.grad;
    end
    # if(options.precondition && N > mth.numinneriters)
    #     Id = 10.0^(-10)*eye(size(mth.H)[1],size(mth.H)[1]);
    #     d[:] = d[:] - mth.Sold*inv(inv(mth.H+Id)+ mth.Sold'*mth.Sold+Id)*mth.Sold'*d;
    # end
end
