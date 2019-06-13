function descent_SPIN(x::Array{Float64}, prob::Prob, options::MyOptions, mth::SPIN, N::Int64, d::Array{Float64})
    # Sketching Projected Newtin x,prob,options,method,iter,d
    # if(N == 1)
    #     prob.g_eval!(x, 1:prob.numdata, mth.dn);
    #     avrg_dir = mth.dn;
    # end
    if (mth.sketchtype == "prev")
        mth.S[:] = mth.avrg_dir .+ randn(prob.numfeatures, mth.sketchsize);
        # mth.S[:,1:end-1] =mth.S[:,2:end];
        # mth.S[:,end] = mth.dn;
        # mth.S[:,:] =  mth.S[:,:] + randn(prob.numfeatures, mth.sketchsize);
    # elseif (mth.sketchtype == "cov")
        # mth.cov_dir[:,:] = mth.weight*mth.cov_dir + (1.0-mth.weight)* mth.dn *(mth.dn');
        # mth.S[:] = mth.avrg_dir .+ LinearAlgebra.cholesky(mth.cov_dir)*randn(prob.numfeatures, mth.sketchsize);
    elseif (mth.sketchtype == "coord")
        C = convert(Array{Int64}, sample(1:prob.numfeatures, mth.sketchsize, replace=false));
        prob.Hess_CC_g_C!(x, 1:prob.numdata, C, mth.rhs, mth.SHS);
    else
        mth.S[:] = randn(prob.numfeatures, mth.sketchsize);  #Add on last direction for embedding
    end

    if (mth.sketchtype != "coord")
        prob.Hess_opt!(x, 1:prob.numdata, mth.S, mth.grad, mth.HS); # Calculates gradient and Hessian vector products
        mth.SHS[:] = (mth.S')*(mth.HS);
        mth.rhs[:] =  mth.S'*mth.grad;
    end

 
    mth.rhs[:] =  mth.SHS\mth.rhs;
    d[:]       =  -mth.S*mth.rhs;
    mth.dn[:]  = d;


  
    mth.avrg_dir[:] =  (mth.weight*mth.avrg_dir[:] + (1.0-mth.weight)*d);
    # if (mth.sketchtype == "prev")
    #     mth.S[:,1:end-1] =mth.S[:,2:end];
    #     mth.S[:,end] = d;
    # end
end


# Using the last direction as a "bias".
    # mth.SHS[:] = (mth.S')*(mth.HS);
    # mth.rhs[:] =  mth.S'*mth.grad + mth.HS'*mth.dn;
    # mth.rhs[:] =  mth.SHS\mth.rhs;
    # d[:]       = mth.dn  -mth.S*mth.rhs;
    # mth.dn[:]  = d;