function descent_AMcoord(x::Array{Float64}, prob::Prob, options::MyOptions, mth::Method, N::Int64, d::Array{Float64})
    # SVRGkemb: A 2nd variant of SVRG that uses a embed Hessian matrix mth.S'*H*mth.S
    embeddim = Int64(options.aux[1]);

    if(N%mth.numinneriters == 0)# Reset reference point, grad estimate and Hessian estimate
        mth.prevx[:] = x; # Stores previous x
        mth.S[:] = sample(1:prob.numfeatures, embeddim, replace=false);
        C = convert(Array{Int64}, mth.S);
        prob.Hess_C!(x, 1:prob.numdata, C, mth.grad, mth.HS);
        mth.H[:] = pinv(mth.HS[C, :]);
        d[:] = -mth.grad;
    elseif(N > mth.numinneriters)     #SVRGkemb inner step
        sample!(1:prob.numdata, mth.ind, replace=false);
        mth.diffpnt[:] = x - mth.prevx;
        C = convert(Array{Int64}, mth.S);
        prob.Hess_C!(mth.prevx, mth.ind, C, mth.gradsamp, mth.HSi); # Stores H_i S
        mth.SHS[:] = mth.HSi[C, :];
        mth.aux[:] = mth.H* mth.HS'* mth.diffpnt ;
        d[:] = mth.HSi* mth.aux;
        d[:] = d - mth.HS* (mth.H* d[C,:]) - mth.HS* mth.aux ;
        d[:] = d + mth.HS* (mth.H* ((mth.HSi')* mth.diffpnt));
        d[:] = d - prob.g_eval(x, mth.ind) + mth.gradsamp - mth.grad;
        # y =  (HS)'* v
        # d = v-Sy
        # d = HS(S' H_i d-y)	+ HiS*y
    else # SVRG  
        sample!(1:prob.numdata, mth.ind, replace=false);  
        d[:] = -prob.g_eval(x, mth.ind) + prob.g_eval(mth.prevx, mth.ind) - mth.grad;
    end
    # if(options.precondition && N>mth.numinneriters)
    #     Id = 10.0^(-10)*eye(size(mth.H)[1],size(mth.H)[1]);
    #     d[:] = d[:] - mth.Sold*inv(inv(mth.H+Id)+ mth.Sold'*mth.Sold+Id)*mth.Sold'*d;
    # end
end
