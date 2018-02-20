function descent_BFGS(x::Array{Float64},prob::Prob,options::MyOptions,mth::Method, N::Int64,d::Array{Float64})
# BFGS: The full memory original verion:
# H_{k+1} = \frac{\delta_k \delta_k^\top}{\delta_k^\top y_k}+ (I-\frac{\delta_k y_k^\top}{\delta_k^\top y_k} ) H_{k} (I -\frac{y_k\delta_k^\top}{\delta_k^\top y_k}  )

prob.g_eval!(x,1:prob.numdata, mth.grad);
if(N==1)
   d[:] = -mth.grad; # Times starting matrix H0!
   # println("grad"); println(sum(mth.grad));
 else
   mth.S[:] = mth.grad - mth.gradsamp; # updating the difference between gradients
   mth.diffpnt[:] = x - mth.prevx;
   dx_dy = vecdot(mth.diffpnt, mth.S);
   mth.HS[:] = mth.H*mth.S;  # storing the H dy product;
   BFGS_update!(mth.H,mth.H, mth, dx_dy);
   #mth.H[:] = mth.H[:]  + (1+ vecdot(mth.HS,mth.S)/dx_dy)*(mth.diffpnt*mth.diffpnt')/dx_dy- (mth.HS*mth.diffpnt'+mth.diffpnt*mth.HS')/dx_dy;
   d[:] = -mth.H* mth.grad;
end
mth.prevx[:]  = x;
mth.gradsamp[:] = mth.grad;

end

# \frac{\delta_k \delta_k^\top}{\delta_k^\top y_k}
#    + (H_{k}-\frac{\delta_k y_k^\top}{\delta_k^\top y_k} H_{k} ) (I -\frac{y_k\delta_k^\top}{\delta_k^\top y_k}  )
# = \frac{\delta_k \delta_k^\top}{\delta_k^\top y_k}
#    + H_{k}- \frac{\delta_k (y_k^\top H_{k})}{\delta_k^\top y_k}  - \frac{(H_k y_k)\delta_k^\top}{\delta_k^\top y_k}
#    + \frac{\delta_k \delta_k^\top }{(\delta_k^\top y_k)^2 }
# = H_k - \frac{\delta_k (y_k^\top H_{k})}{\delta_k^\top y_k}  - \frac{(H_k y_k)\delta_k^\top}{\delta_k^\top y_k}
#       +  (1+ y_k^\top H_{k} y_k/(\delta_k^\top y_k))\frac{\delta_k \delta_k^\top}{\delta_k^\top y_k}
