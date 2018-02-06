function BFGS_update!(H, mth::Method, dx_dy)
# BFGS: The full memory original verion:
# H_{k+1} = \frac{\delta_k \delta_k^\top}{\delta_k^\top y_k}+ (I-\frac{\delta_k y_k^\top}{\delta_k^\top y_k} ) H_{k} (I -\frac{y_k\delta_k^\top}{\delta_k^\top y_k}  )

c1 = (dx_dy + vecdot(mth.HS,mth.S)) / (dx_dy * dx_dy);
c2 = 1 / dx_dy;
H[:] = H  .+ c1*(mth.diffpnt*mth.diffpnt').- c2*(mth.HS*mth.diffpnt'+mth.diffpnt*mth.HS');
# Test speed of looping instead
# for i =1:length(dx_dy)
#     for j =1:length(dx_dy)
#           @inbounds mth.H[i,j] += c1*mth.diffpnt[i]*mth.diffpnt[j]- c2*(mth.HS[i]*mth.diffpnt[j]+mth.diffpnt[i]*mth.HS[j]);
#     end
# end


end

# How the update was simplified:
# \frac{\delta_k \delta_k^\top}{\delta_k^\top y_k}
#    + (H_{k}-\frac{\delta_k y_k^\top}{\delta_k^\top y_k} H_{k} ) (I -\frac{y_k\delta_k^\top}{\delta_k^\top y_k}  )
# = \frac{\delta_k \delta_k^\top}{\delta_k^\top y_k}
#    + H_{k}- \frac{\delta_k (y_k^\top H_{k})}{\delta_k^\top y_k}  - \frac{(H_k y_k)\delta_k^\top}{\delta_k^\top y_k}
#    + \frac{\delta_k \delta_k^\top }{(\delta_k^\top y_k)^2 }
# = H_k - \frac{\delta_k (y_k^\top H_{k})}{\delta_k^\top y_k}  - \frac{(H_k y_k)\delta_k^\top}{\delta_k^\top y_k}
#       +  (1+ y_k^\top H_{k} y_k/(\delta_k^\top y_k))\frac{\delta_k \delta_k^\top}{\delta_k^\top y_k}
