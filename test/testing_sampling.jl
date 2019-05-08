## Test Sampling object

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
include("./src/StochOpt.jl")

## Basic parameters and options for solvers
options = set_options()

## Sampling procedure
n = 20
s = Sampling("nice", n, 1, Float64[], nice_sampling)

# b-nice sampling
options.batchsize = 4
s = build_sampling("nice", n, options)
s.sampleindices(s)
s.batchsize
s.name

# uniform b-independent sampling
options.batchsize = 3
s = build_sampling("independent", n, options)
s.sampleindices(s)
s.sampleindices(s)
s.batchsize
s.name

# nonuniform b-independent sampling
options.batchsize = 4
probas = [0.1, 0.3, 0.3, 0.9, 0.9]
s = build_sampling("independent", length(probas), options, probas=probas)
s.sampleindices(s)
s.batchsize
s.name

# nonuniform b-independent sampling
options.batchsize = 4
probas = [0.1, 0.1, 0.1, 0.1, 0.3]
s = build_sampling("independent", length(probas), options, probas=probas)
s.sampleindices(s)
s.batchsize
s.name