using Random
using Printf
using LinearAlgebra # julia 0.7

include("../util/power_iteration.jl") # Be carefull about the path here

Random.seed!(1);

## Display function for elapsed time
function sec2time(t::Float64)
    (remainder, nanoseconds) = fldmod(t*1e9, 1000)
    (remainder, microseconds) = fldmod(remainder, 1000)
    (remainder, milliseconds) = fldmod(remainder, 1000)
    (minutes, seconds) = fldmod(remainder, 60)

    nanoseconds = ceil(nanoseconds)
    microseconds = floor(microseconds)
    milliseconds = floor(milliseconds)
    seconds = floor(seconds)
    minutes = floor(minutes)
    return minutes, seconds, milliseconds, microseconds, nanoseconds
end


function gen_sym_matrix(d)
    A = rand(d, d)
    return A'*A
end

## Symmetric version
function gen_typed_sym_matrix(d)
    A = rand(d, d)
    return Symmetric(A'*A)
end


## Julia eigmax and eigmin
function test_eigmax(d, numruns=10000)
    t = 0.0
    for it=1:numruns
        A = gen_sym_matrix(d)
        t += @elapsed eigmax(A)
    end
    t /= numruns
    m, s, ms, microsec, ns = sec2time(t)
    @printf("eigmax average time                      : %02d m %02d s %03d ms %03d µs %03d ns (%d runs)\n", m, s, ms, microsec, ns, numruns)
end

function test_eigmin(d, numruns=10000)
    t = 0.0
    for it=1:numruns
        A = gen_sym_matrix(d)
        t += @elapsed eigmin(A)
    end
    t /= numruns
    m, s, ms, microsec, ns = sec2time(t)
    @printf("eigmin average time                      : %02d m %02d s %03d ms %03d µs %03d ns (%d runs)\n", m, s, ms, microsec, ns, numruns)
end

## Symmetric versions
function test_Symmetric_eigmax(d, numruns=10000)
    t = 0.0
    for it=1:numruns
        A = gen_typed_sym_matrix(d)
        t += @elapsed eigmax(A)
    end
    t /= numruns
    m, s, ms, microsec, ns = sec2time(t)
    @printf("Symmetric_eigmax average time            : %02d m %02d s %03d ms %03d µs %03d ns (%d runs)\n", m, s, ms, microsec, ns, numruns)
end

function test_Symmetric_eigmin(d, numruns=10000)
    t = 0.0
    for it=1:numruns
        A = gen_typed_sym_matrix(d)
        t += @elapsed eigmin(A)
    end
    t /= numruns
    m, s, ms, microsec, ns = sec2time(t)
    @printf("Symmetric_eigmin average time            : %02d m %02d s %03d ms %03d µs %03d ns (%d runs)\n", m, s, ms, microsec, ns, numruns)
end

## Power and inverse iteration
function test_power_iteration(d, numruns=10000)
    t = 0.0
    for it=1:numruns
        A = gen_sym_matrix(d)
        t += @elapsed power_iteration(A)
    end
    t /= numruns
    m, s, ms, microsec, ns = sec2time(t)
    @printf("power_iteration average time             : %02d m %02d s %03d ms %03d µs %03d ns (%d runs)\n", m, s, ms, microsec, ns, numruns)
end

function test_inverse_iteration(d, numruns=10000)
    t = 0.0
    for it=1:numruns
        A = gen_sym_matrix(d)
        t += @elapsed inverse_iteration(A)
    end
    t /= numruns
    m, s, ms, microsec, ns = sec2time(t)
    @printf("inverse_iteration average time           : %02d m %02d s %03d ms %03d µs %03d ns (%d runs)\n", m, s, ms, microsec, ns, numruns)
end

## Symmetric versions
function test_Symmetric_power_iteration(d, numruns=10000)
    t = 0.0
    for it=1:numruns
        A = gen_typed_sym_matrix(d)
        t += @elapsed Symmetric_power_iteration(A)
    end
    t /= numruns
    m, s, ms, microsec, ns = sec2time(t)
    @printf("Symmetric_power_iteration average time   : %02d m %02d s %03d ms %03d µs %03d ns (%d runs)\n", m, s, ms, microsec, ns, numruns)
end

function test_Symmetric_inverse_iteration(d, numruns=10000)
    t = 0.0
    for it=1:numruns
        A = gen_typed_sym_matrix(d)
        t += @elapsed Symmetric_inverse_iteration(A)
    end
    t /= numruns
    m, s, ms, microsec, ns = sec2time(t)
    @printf("Symmetric_inverse_iteration average time : %02d m %02d s %03d ms %03d µs %03d ns (%d runs)\n", m, s, ms, microsec, ns, numruns)
end



## --- Correctness of the approximation --- ##
println("######## CORRECTNESS OF THE APPROXIMATION ########\n")
numsim=10;
A = gen_sym_matrix(100);
B = gen_typed_sym_matrix(100);
println("------ Largest eigenvalue approximation error ------")
println("power_iteration              : ", abs(eigmax(A)-power_iteration(A, numsim=numsim)))
println("Symmetric_power_iteration    : ", abs(eigmax(B)-Symmetric_power_iteration(B, numsim=numsim)), "\n")

println("------ Smallest eigenvalue approximation error ------")
println("inverse_iteration            : ", abs(eigmin(A)-inverse_iteration(A, numsim=numsim)))
println("Symmetric_inverse_iteration  : ", abs(eigmin(B)-Symmetric_inverse_iteration(B, numsim=numsim)), "\n")



# ## --- Time comparison --- ##
# println("######## TIME COMPARISON ########\n")

# ## --- Largest eigenvalue ---
# println("------ eigmax vs Symmetric_eigmax vs power_iteration vs Symmetric_power_iteration ------")
# # Dimension = 10
# d = 10
# numruns = 100000
# println("d = ", d, ", number of runs = ", numruns)
# test_eigmax(d, numruns)
# test_Symmetric_eigmax(d, numruns)
# test_power_iteration(d, numruns)
# test_Symmetric_power_iteration(d, numruns)

# # Dimension = 100
# d = 100
# numruns = 10000
# println("\nd = ", d, ", number of runs = ", numruns)
# test_eigmax(d, numruns)
# test_Symmetric_eigmax(d, numruns)
# test_power_iteration(d, numruns)
# test_Symmetric_power_iteration(d, numruns)

# # Dimension = 1000
# d = 1000
# numruns = 100
# println("\nd = ", d, ", number of runs = ", numruns)
# test_eigmax(d, numruns)
# test_Symmetric_eigmax(d, numruns)
# test_power_iteration(d, numruns)
# test_Symmetric_power_iteration(d, numruns)

# # Dimension = 10000
# d = 10000
# numruns = 1
# println("\nd = ", d, ", number of runs = ", numruns)
# test_eigmax(d, numruns)
# test_Symmetric_eigmax(d, numruns)
# test_power_iteration(d, numruns)
# test_Symmetric_power_iteration(d, numruns)



# ## --- Smallest eigenvalue ---
# println("\n\n------ eigmin vs Symmetric_eigmin vs inverse_iteration vs Symmetric_inverse_iteration ")
# # Dimension = 10
# d = 10
# numruns = 100000
# println("d = ", d, ", number of runs = ", numruns)
# test_eigmin(d, numruns)
# test_Symmetric_eigmin(d, numruns)
# test_inverse_iteration(d, numruns)
# test_Symmetric_inverse_iteration(d, numruns)

# # Dimension = 100
# d = 100
# numruns = 10000
# println("\nd = ", d, ", number of runs = ", numruns)
# test_eigmin(d, numruns)
# test_Symmetric_eigmin(d, numruns)
# test_inverse_iteration(d, numruns)
# test_Symmetric_inverse_iteration(d, numruns)

# # Dimension = 1000
# d = 1000
# numruns = 100
# println("\nd = ", d, ", number of runs = ", numruns)
# test_eigmin(d, numruns)
# test_Symmetric_eigmin(d, numruns)
# test_inverse_iteration(d, numruns)
# test_Symmetric_inverse_iteration(d, numruns)

# # Dimension = 10000
# d = 10000
# numruns = 1
# println("\nd = ", d, ", number of runs = ", numruns)
# test_eigmin(d, numruns)
# test_Symmetric_eigmin(d, numruns)
# test_inverse_iteration(d, numruns)
# test_Symmetric_inverse_iteration(d, numruns)


##################################################################################

################ ESTIMATING COMPUTATION TIME ON THE SAME MATRICES ################
function test_estimate_largest_eigval(d, numruns)
    t1 = 0.0 # t_eigmax
    t2 = 0.0 # t_Symmetric_eigmax
    t3 = 0.0 # t_power_iteration
    t4 = 0.0 # t_Symmetric_power_iteration
    for it=1:numruns
        # type(A) = Array{Float64,2}
        A = gen_sym_matrix(d)
        t1 += @elapsed eigmax(A)
        t3 += @elapsed power_iteration(A)
        # type(A) = Symmetric{Float64,Array{Float64,2}}
        A = Symmetric(A)
        t2 += @elapsed eigmax(A)
        t4 += @elapsed Symmetric_power_iteration(A)
    end
    t1 /= numruns
    t2 /= numruns
    t3 /= numruns
    t4 /= numruns
    m1, s1, ms1, microsec1, ns1 = sec2time(t1)
    m2, s2, ms2, microsec2, ns2 = sec2time(t2)
    m3, s3, ms3, microsec3, ns3 = sec2time(t3)
    m4, s4, ms4, microsec4, ns4 = sec2time(t4)

    @printf("eigmax average time                        : %02d m %02d s %03d ms %03d µs %03d ns\n", m1, s1, ms1, microsec1, ns1)
    @printf("Symmetric_eigmax average time              : %02d m %02d s %03d ms %03d µs %03d ns\n", m2, s2, ms2, microsec2, ns2)
    @printf("power_iteration average time               : %02d m %02d s %03d ms %03d µs %03d ns\n", m3, s3, ms3, microsec3, ns3)
    @printf("Symmetric_power_iteration average time     : %02d m %02d s %03d ms %03d µs %03d ns\n\n", m4, s4, ms4, microsec4, ns4)
end

function test_estimate_smallest_eigval(d, numruns)
    t1 = 0.0 # t_eigmax
    t2 = 0.0 # t_Symmetric_eigmax
    t3 = 0.0 # t_power_iteration
    t4 = 0.0 # t_Symmetric_power_iteration
    for it=1:numruns
        # type(A) = Array{Float64,2}
        A = gen_sym_matrix(d)
        t1 += @elapsed eigmin(A)
        t3 += @elapsed inverse_iteration(A)
        # type(A) = Symmetric{Float64,Array{Float64,2}}
        A = Symmetric(A)
        t2 += @elapsed eigmin(A)
        t4 += @elapsed Symmetric_inverse_iteration(A)
    end
    t1 /= numruns
    t2 /= numruns
    t3 /= numruns
    t4 /= numruns
    m1, s1, ms1, microsec1, ns1 = sec2time(t1)
    m2, s2, ms2, microsec2, ns2 = sec2time(t2)
    m3, s3, ms3, microsec3, ns3 = sec2time(t3)
    m4, s4, ms4, microsec4, ns4 = sec2time(t4)

    @printf("eigmin average time                        : %02d m %02d s %03d ms %03d µs %03d ns\n", m1, s1, ms1, microsec1, ns1)
    @printf("Symmetric_eigmin average time              : %02d m %02d s %03d ms %03d µs %03d ns\n", m2, s2, ms2, microsec2, ns2)
    @printf("inverse_iteration average time             : %02d m %02d s %03d ms %03d µs %03d ns\n", m3, s3, ms3, microsec3, ns3)
    @printf("Symmetric_inverse_iteration average time   : %02d m %02d s %03d ms %03d µs %03d ns\n\n", m4, s4, ms4, microsec4, ns4)
end


## --- Largest eigenvalue ---
println("------ eigmax vs Symmetric_eigmax vs power_iteration vs Symmetric_power_iteration ------")

d = 10
numruns = 100000
@printf("Dimension = %d, number of runs %d\n", d, numruns)
test_estimate_largest_eigval(d, numruns)

d = 100
numruns = 10000
@printf("Dimension = %d, number of runs %d\n", d, numruns)
test_estimate_largest_eigval(d, numruns)

d = 1000
numruns = 100
@printf("Dimension = %d, number of runs %d\n", d, numruns)
test_estimate_largest_eigval(d, numruns)

d = 2000
numruns = 100
@printf("Dimension = %d, number of runs %d\n", d, numruns)
test_estimate_largest_eigval(d, numruns)

d = 10000
numruns = 10
@printf("Dimension = %d, number of runs %d\n", d, numruns)
test_estimate_largest_eigval(d, numruns)


## --- Smallest eigenvalue ---
println("------ eigmin vs Symmetric_eigmin vs inverse_iteration vs Symmetric_inverse_iteration ------")

d = 10
numruns = 100000
@printf("Dimension = %d, number of runs %d\n", d, numruns)
test_estimate_smallest_eigval(d, numruns)

d = 100
numruns = 10000
@printf("Dimension = %d, number of runs %d\n", d, numruns)
test_estimate_smallest_eigval(d, numruns)

d = 1000
numruns = 100
@printf("Dimension = %d, number of runs %d\n", d, numruns)
test_estimate_smallest_eigval(d, numruns)

d = 2000
numruns = 100
@printf("Dimension = %d, number of runs %d\n", d, numruns)
test_estimate_smallest_eigval(d, numruns)

d = 10000
numruns = 10
@printf("Dimension = %d, number of runs %d\n", d, numruns)
test_estimate_smallest_eigval(d, numruns)