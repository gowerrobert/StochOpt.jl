function descent_SPIN(x::Array{Float64}, prob::Prob, options::MyOptions, mth::SPIN, N::Int64, d::Array{Float64})
    # Sketching Projected Newtin x,prob,options,method,iter,d
    if(N == 1)
        prob.g_eval!(x, 1:prob.numdata, mth.dn);
    end
    if (mth.sketchtype == "prev")
        mth.S[:] = mth.dn .+ randn(prob.numfeatures, mth.sketchsize);
    else
        mth.S[:] = randn(prob.numfeatures, mth.sketchsize);  #Add on last direction for embedding
    end
    prob.Hess_opt!(x, 1:prob.numdata, mth.S, mth.grad, mth.HS); # Calculates gradient and Hessian vector products
    # mth.S[:]  = mth.S*mth.HS;  #Add on last direction for embedding
    # mth.HS[:] = mth.HS*mth.SHS;
    mth.SHS[:] = (mth.S')*(mth.HS);
    mth.rhs[:] =  mth.S'*mth.grad + mth.HS'*mth.dn;
    mth.rhs[:] =  mth.SHS\mth.rhs;
    d[:]       = mth.dn  -mth.S*mth.rhs;
    mth.dn[:]  = d;
end