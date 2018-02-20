function descent_BFGS_accel(x::Array{Float64},prob::Prob,options::MyOptions,mth::Method, N::Int64,d::Array{Float64})
# BFGS_accel: An accelerated version of The full memory original verion:
# Y_k =  \alpha V_k+(1- \alpha)X_k  \nonumber \\
# X_{k+1} =  BFGS_update!(Y_k)
# V_{k+1} = \beta V_k+(1-\beta)Y_k-\gamma (Y_k - X_{k+1})

# In this implementation using
# mth.HSi   # stores the Y_k's
# mth.SHS   # stores the V_k's

## Adaptive mu and nu:
# mu = options.embeddim[1];
# Hd = prob.Hess_D(x, 1:prob.numdata)
# nu =  sum(Hd)/minimum(Hd);
# # mu = 1/sum(Hd);
# beta = 1 - sqrt(mu/nu);
# gamma = sqrt(1/(mu*nu));
# mth.aux[1] = 1/(1+gamma*nu);
# mth.aux[2] = beta;
# mth.aux[3] = gamma;
# println("(mu, nu) =  (", round(mu,3), ", ", round(nu,2), ")")

prob.g_eval!(x,1:prob.numdata, mth.grad);
if(N==1)
   d[:] = -mth.grad; # Times starting matrix H0!
   # println("grad"); println(sum(mth.grad));
 else
   # println("V  :", mth.SHS)
   mth.S[:] = mth.grad - mth.gradsamp; # updating the difference between gradients
   mth.diffpnt[:] = x - mth.prevx;
   mth.HSi[:] = mth.aux[1]* mth.SHS +(1-mth.aux[1])*mth.H; # Y_k =  \alpha V_k+(1- \alpha)X_k
   dx_dy = vecdot(mth.diffpnt, mth.S);
   mth.HS[:] = mth.HSi*mth.S;  # storing the Y dy product;
   # println("H before :", mth.H)
   BFGS_update!(mth.H,mth.HSi, mth, dx_dy); # X_k =   BFGS_update(Y_k)
   # println("H after :", mth.H)
   mth.SHS[:]= mth.aux[2]* mth.SHS +(1-mth.aux[2])*mth.HSi - mth.aux[3]*(mth.HSi -mth.H); #V_{k+1} = beta V_k+(1-beta)Y_k -gamma (Y_k - X_{k+1})
   d[:] = -mth.H* mth.grad;
end
mth.prevx[:]  = x;
mth.gradsamp[:] = mth.grad;

end
