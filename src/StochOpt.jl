# The types exported
#export MyOptions, Prob, Method, Output
# The exported functions
#export pseudoinvert_matrix, uniform_mat_rank, uniform_sym_rank
#export plot_outputs_Plots

type MyOptions
    tol::Float64
    aux::Float64
    max_iter::Int
    skip_error_calculation::Int
    max_time::Float64
    max_epocs::Int64
    printiters::Bool
    exacterror::Bool
    repeat_stepsize_calculation::Bool
    batchsize
    regulatrizor_parameter::AbstractString
    stepsize_multiplier::Float64
    precondition::Bool
    force_continue::Bool
    rep_number::Int64 # number of times the optimization should be repeated. Only the average is reported.
    embeddim# Hessian_type::AbstractString
end

function set_options(; tol::Float64 = 10.0^(-6.0),
    aux::Float64 = Inf,
    max_iter::Int = 10^8,
    skip_error_calculation::Int = 0,
    max_time::Float64 =350.0,
    max_epocs::Int64 =30,
    printiters::Bool =true,
    exacterror::Bool =true,
    repeat_stepsize_calculation::Bool = false,
    batchsize::Int64 =100,
    regulatrizor_parameter::AbstractString = "normalized",
    stepsize_multiplier::Float64=0.0,
    precondition::Bool=false,
    force_continue::Bool=false,
    rep_number::Int64=10, # number of times the optimization should be repeated. Only the average is reported.
    embeddim=0)

    options = MyOptions(tol,aux,max_iter,skip_error_calculation,max_time,max_epocs,
    printiters,exacterror,repeat_stepsize_calculation, batchsize,"normalized",stepsize_multiplier,precondition, force_continue,rep_number,embeddim)
end

type Prob
    X::Array{Float64}  # why not a sparse array?
    y::Array{Float64}
    numfeatures::Int64
    numdata::Int64
    fsol::Float64
    name::AbstractString
    f_eval::Function
    g_eval::Function
    g_eval!::Function
    Jac_eval!::Function
    Hess_eval::Function# Calculates the Hessian Hess_eval,Hess_opt,Hess_vv,Hess_D
    Hess_eval!::Function# Calculates the Hessian Hess_eval,Hess_opt,Hess_vv,Hess_D
    Hess_opt::Function # Calculates the Hessian-vector product Hv
    Hess_opt!::Function # Calculates the Hessian-vector product Hv
    Hess_D::Function   # Calculates Diagonal
    Hess_D!::Function   # Calculates Diagonal
    Hess_C::Function   # Gets subset of the columns of Hessian
    Hess_C!::Function   # Gets subset of the columns of Hessian
    Hess_C2::Function
    lambda::Float64
    #    Hess_vv::Function  # Calculates the Hessian-vector-vector product v'Hv
end

type SAGAmethod
    epocsperiter::Float64
    gradsperiter::Float64
    name::AbstractString
    stepmethod::Function
    bootmethod::Function
    minibatches::Array{Int64}
    minibatch_type::AbstractString  # Type of mini-batching, e.g, tau-nice, tau-partition, tau-with replacement
    unbiased::Bool        # unbiased = true for SAGA and unbiased = false for SAG
    Jac::Array{Float64}  #Jacobian estimate
    Jacsp::SparseMatrixCSC #Sparse JAcobian estimate
    SAGgrad::Array{Float64}  # The SAG estimate of full gradient, needed for computing unbiased gradient estimate.
    gi::Array{Float64} # Storage for a single stochastic gradient
    aux::Array{Float64}  # Storage for an auxiliary vector, used as the update vectoe
    stepsize::Float64     # The stepsize
    probs::Array{Float64}  # Probability of selecting a coordinate or mini-batch
    probability_type::AbstractString # type of probabilities used, e.g., uniform, nonuniform, nonuniform_opt
    L::Float64  # An estimate of the smoothness constant
end
type Method
    epocsperiter::Float64
    gradsperiter::Float64
    name::AbstractString
    stepmethod::Function
    grad::Array{Float64}  # Current estimate of full gradient
    gradsamp::Array{Float64} # subsampled local gradient
    S:: Array{Float64}   # embedding space
    H:: Array{Float64}   # An approximation of the  Hessian
    Hsp::SparseMatrixCSC # Sparse Hessian matrix
    HS::Array{Float64}   # action of the  Hessian on S
    HSi::Array{Float64}   # action of the ith Hessian
    SHS:: Array{Float64}   # An embedded Hessian
    stepsize::Float64     # The stepsize
    prevx::Array{Float64} # Storing a reference point or previous point
    diffpnt::Array{Float64} #Stores differences between old and new
    Sold::Array{Float64} #old embedding space
    ind::Array{Int64} # place holder for aan array of indices
    aux::Array{Float64} # An auxiliary vector used in computations
    numinneriters::Int64
end

type Output
    name::AbstractString
    iterations::Int
    epocsperiter::Float64 #Array{Float64}
    gradsperiter::Float64
    times::Array{Float64}
    fs::Array{Float64}   # recorded function values
    xfinal::Array{Float64}  #The final x of the method
    fail::AbstractString
    stepsize_multiplier::Float64
end
#Functions for getting and manipulating data
include("dataLoad.jl")
include("logistic_defintions.jl") # includes all functions and definitions pertaining to logistic regression
# load_ridge_regression
#Including method wrappers
include("minimizeFunc.jl") # minimizing a given prob::Prob using a method
include("minimizeFunc_grid_stepsize.jl")  # minimizing a given prob::Prob using a method and determines the stepsize using a grid search
#include("minimizeFunc_with_test.jl")
include("boot_method.jl")
#Including test and problem generating functions
#Including iterative methods for calculating search direction
allmethods = ["SAGA", "SVRG", "SVRG2",  "2D", "2Dsec", "CMcoord", "CMgauss", "CMprev", "DFPgauss","DFPprev",  "DFPcoord", "BFGS", "BFGS_accel", "grad" ] ;
for method in allmethods
  include(string("boot_", method ,".jl"))
  include(string("descent_", method ,".jl"))
end
include("descent_SAGApartition.jl")
#Including utilities, plotting, data analysis
include("plot_outputs_Plots.jl")
include("get_saved_stepsize.jl");
include("load_fsol.jl");
#Additional
include("BFGS_update!.jl")
