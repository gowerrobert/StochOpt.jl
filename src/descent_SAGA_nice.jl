
"""
    descent_SAGA_nice(x::Array{Float64}, prob::Prob, options::MyOptions, sg::SAGA_nice_method, N::Int64, d::Array{Float64})

Compute the descent direction (d)

#INPUTS:\\
    - **Array{Float64}** x: point at the current iteration
    - **Prob** prob: considered problem, i.e. logistic regression, ridge ression... (see src/StochOpt.jl)\\
    - **MyOptions** options: different options such as the mini-batch size, the stepsize_multiplier... (see src/StochOpt.jl)\\
    - **SAGA_nice_method** sg: method of SAGA for ``τ``--nice sampling\\
    - **Int64** iter: current iteration, _useless for SAGA nice_\\
    - **Array{Float64}** d: descent direction\\
#OUTPUTS:\\
    - NONE
"""
function descent_SAGA_nice(x::Array{Float64}, prob::Prob, options::MyOptions, sg::SAGA_nice_method, iter::Int64, d::Array{Float64})
    # /!\ WARNING: this function modifies its own arguments (d and sg) and return nothing! Shouldn't we name it "descent_SAGA_nice!(...)" with an "!" ?
    s = sample(1:prob.numdata, options.batchsize, replace=false);
    # Assign each gradient to a different column of Jac
    sg.aux[:] = -sum(sg.Jac[:,s], 2); # Calculating the update vector aux = (DF^k-J^k) Proj 1 = sum_{i \in S_k} (\nabla f_i (x^k) - J_{:i}^k)
    prob.Jac_eval!(x, s, sg.Jac); # Update of the Jacobian estimate
    sg.aux[:] += sum(sg.Jac[:,s], 2);

    # ## Using the scalar gradient trick for "linear models"
    # scalargrad = prob.scalar_grad_eval(x, s);
    # sg.gi[:] = (prob.X[:, s])*scalargrad;
    # ## Calculating the update vector (DF^k-J^k) Proj 1
    # sg.aux[:] = -(prob.X[:, s])*sg.Jac[s]; # - sum_{i\in S_k} J_ {:i}^k
    # sg.aux[:] += sg.gi; # + sum_{i\in S_k} grad f_i (x^k)
    # sg.Jac[s] = scalargrad;
    
    # Minus the gradient estimate = descent direction
    # Update of minus the unbiased gradient estimate: -g^k
    if(sg.unbiased)
        d[:] = -sg.SAGgrad - (1/options.batchsize)*sg.aux;
    else
        d[:] = -sg.SAGgrad; #- (1/options.batchsize)*sg.aux;
    end

    # Update SAG estimate:   1/n J^{k+1}1 = 1/n J^k 1 + 1/n (DF^k-J^k) Proj 1
    # Update of the biased gradient estimate v^k
    sg.SAGgrad[:] = sg.SAGgrad + (1/prob.numdata)*sg.aux;
end
