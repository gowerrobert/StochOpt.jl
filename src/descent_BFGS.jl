function descent_BFGS(x::Array{Float64}, prob::Prob, options::MyOptions, method::Method, N::Int64, d::Array{Float64})
    # BFGS: The full memory original verion:
    # H_{k+1} = \frac{\delta_k \delta_k^\top}{\delta_k^\top y_k}+ (I-\frac{\delta_k y_k^\top}{\delta_k^\top y_k} ) H_{k} (I -\frac{y_k\delta_k^\top}{\delta_k^\top y_k}  )

    prob.g_eval!(x, 1:prob.numdata, method.grad); # Out of memory error() when numdata or numfeatures are too large
    if(N == 1)
        d[:] = -method.grad; # Times starting matrix H0!
        # println("grad"); println(sum(method.grad));
    else
        method.S[:] = method.grad - method.gradsamp; # updating the difference between gradients
        method.diffpnt[:] = x - method.prevx;
        dx_dy = dot(method.diffpnt, method.S);
        method.HS[:] = method.H*method.S;  # storing the H dy product;
        BFGS_update!(method.H, method.H, method, dx_dy);
        #method.H[:] = method.H[:]  + (1+ dot(method.HS,method.S)/dx_dy)*(method.diffpnt*method.diffpnt')/dx_dy- (method.HS*method.diffpnt'+method.diffpnt*method.HS')/dx_dy;
        d[:] = -method.H*method.grad;
    end
    method.prevx[:] = x;
    method.gradsamp[:] = method.grad;
end

# \frac{\delta_k \delta_k^\top}{\delta_k^\top y_k}
#    + (H_{k}-\frac{\delta_k y_k^\top}{\delta_k^\top y_k} H_{k} ) (I -\frac{y_k\delta_k^\top}{\delta_k^\top y_k}  )
# = \frac{\delta_k \delta_k^\top}{\delta_k^\top y_k}
#    + H_{k}- \frac{\delta_k (y_k^\top H_{k})}{\delta_k^\top y_k}  - \frac{(H_k y_k)\delta_k^\top}{\delta_k^\top y_k}
#    + \frac{\delta_k \delta_k^\top }{(\delta_k^\top y_k)^2 }
# = H_k - \frac{\delta_k (y_k^\top H_{k})}{\delta_k^\top y_k}  - \frac{(H_k y_k)\delta_k^\top}{\delta_k^\top y_k}
#       +  (1+ y_k^\top H_{k} y_k/(\delta_k^\top y_k))\frac{\delta_k \delta_k^\top}{\delta_k^\top y_k}
