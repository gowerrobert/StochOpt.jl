function logistic_phi(t::Array{Float64})
    idx = t .> 0;
    out = zeros(size(t));
    out[idx] = (1 + exp.(-t[idx])).^(-1);
    exp_t = exp.(t[.~idx]);
    out[.~idx] = exp_t ./ (1. + exp_t);
    return out
end
