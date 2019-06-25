# Small file to test the functions used to compute the optimal mini-batch size for Free-SVRG and L-SVRG-D

# include("./src/StochOpt.jl")

## All outputs should be true
isapprox(b_hat_tight(0, 1, 1), 0.0, atol=1e-10)
isapprox(b_hat_tight(2, 3, 1), 0.0, atol=1e-10)
isnan(b_hat_tight(1, 3, 1))
isapprox(b_hat_tight(12, 1, 1), 2/sqrt(3), atol=1e-10)

isapprox(b_tilde_tight(0, 1, 1, 1), 0.0, atol=1e-10)
isapprox(b_tilde_tight(2, 3, 1, 1), 0.0, atol=1e-10)
isinf(b_tilde_tight(2, 5, 1, 3.5))
isapprox(b_tilde_tight(2, 1, 1, 1), 4/3, atol=1e-10)

isapprox(b_hat(0, 2, 1), 0.0, atol=1e-10)
isapprox(b_hat(2, 1, 1), 0.0, atol=1e-10)
isnan(b_hat(1, 1, 1))
isapprox(b_hat(12, 1, 2), sqrt(3/5), atol=1e-10)

isapprox(b_tilde(0, 1, 1, 1), 0.0, atol=1e-10)
isapprox(b_tilde(2, 1, 1, 1), 0.0, atol=1e-10)
isnan(b_tilde(2, 1, 1, 1.5))
isapprox(b_tilde(2, 1, 3, 3), 4/3, atol=1e-10)