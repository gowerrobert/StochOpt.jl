# Small file to test the functions used to compute the optimal mini-batch size for Free-SVRG and L-SVRG-D

using JLD
using Plots
using StatsBase
using Match
using Combinatorics
using Random
using Printf
using LinearAlgebra
using Statistics
using Base64
using Formatting
using SharedArrays

using Test

include("../src/StochOpt.jl")

@testset "Optimal mini-batch functions for Free-SVRG" begin
    # Testing b_hat for the expected residual analysis
    @test isapprox(b_hat_tight(0, 1, 1), 0.0, atol=1e-10)
    @test isapprox(b_hat_tight(2, 3, 1), 0.0, atol=1e-10)
    @test isnan(b_hat_tight(1, 3, 1))
    @test isapprox(b_hat_tight(12, 1, 1), 2/sqrt(3), atol=1e-10)

    # Testing b_tilde for the expected residual analysis
    @test isapprox(b_tilde_tight(0, 1, 1, 1), 0.0, atol=1e-10)
    @test isapprox(b_tilde_tight(2, 3, 1, 1), 0.0, atol=1e-10)
    @test isinf(b_tilde_tight(2, 5, 1, 3.5))
    @test isapprox(b_tilde_tight(2, 1, 1, 1), 4/3, atol=1e-10)

    # Testing loose b_hat
    @test isapprox(b_hat(0, 2, 1), 0.0, atol=1e-10)
    @test isapprox(b_hat(2, 1, 1), 0.0, atol=1e-10)
    @test isnan(b_hat(1, 1, 1))
    @test isapprox(b_hat(12, 1, 2), sqrt(3/5), atol=1e-10)

    # Testing loose b_tilde
    @test isapprox(b_tilde(0, 1, 1, 1), 0.0, atol=1e-10)
    @test isapprox(b_tilde(2, 1, 1, 1), 0.0, atol=1e-10)
    @test isnan(b_tilde(2, 1, 1, 1.5))
    @test isapprox(b_tilde(2, 1, 3, 3), 4/3, atol=1e-10)
end
