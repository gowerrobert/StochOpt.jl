using Printf
using LinearAlgebra # julia 0.7

include("./util/power_iteration.jl") # Be carefull about the path here

function gen_sym_matrix(d)
    A = rand(d, d)
    return A'*A
end

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

function test_eigmax(d, numruns=10000)
    t = 0.0
    for it=1:numruns
        A = gen_sym_matrix(d)
        t += @elapsed eigmax(A)
    end
    t /= numruns
    m, s, ms, microsec, ns = sec2time(t)
    @printf("eigmax average time          : %02d m %02d s %03d ms %03d µs %03d ns (%d runs)\n", m, s, ms, microsec, ns, numruns)
end

function test_eigmin(d, numruns=10000)
    t = 0.0
    for it=1:numruns
        A = gen_sym_matrix(d)
        t += @elapsed eigmin(A)
    end
    t /= numruns
    m, s, ms, microsec, ns = sec2time(t)
    @printf("eigmin average time           : %02d m %02d s %03d ms %03d µs %03d ns (%d runs)\n", m, s, ms, microsec, ns, numruns)
end

function test_power_iteration(d, numruns=10000)
    t = 0.0
    for it=1:numruns
        A = gen_sym_matrix(d)
        t += @elapsed power_iteration(A)
    end
    t /= numruns
    m, s, ms, microsec, ns = sec2time(t)
    @printf("power_iteration average time : %02d m %02d s %03d ms %03d µs %03d ns (%d runs)\n", m, s, ms, microsec, ns, numruns)
end

function test_inverse_iteration(d, numruns=10000)
    t = 0.0
    for it=1:numruns
        A = gen_sym_matrix(d)
        t += @elapsed inverse_iteration(A)
    end
    t /= numruns
    m, s, ms, microsec, ns = sec2time(t)
    @printf("inverse_iteration average time: %02d m %02d s %03d ms %03d µs %03d ns (%d runs)\n", m, s, ms, microsec, ns, numruns)
end


## --- Correctness of the approximation --- ##
println("######## CORRECTNESS OF THE APPROXIMATION ########\n")
A = gen_sym_matrix(100);
println("Approximation error of the largest eigenvalue  : ", abs(eigmax(A)-power_iteration(A)))
println("Approximation error of the smallest eigenvalue : ",abs(eigmin(A)-inverse_iteration(A)), "\n")


## --- Time comparison --- ##
println("######## TIME COMPARISON ########\n")
println("------ eigmax vs power_iteration")
# Dimension = 10
d = 10
numruns = 100000
println("d = ", d, ", number of runs = ", numruns)
test_eigmax(d, numruns)
test_power_iteration(d, numruns)

# Dimension = 100
d = 100
numruns = 1000
println("\nd = ", d, ", number of runs = ", numruns)
test_eigmax(d, numruns)
test_power_iteration(d, numruns)

# Dimension = 10000
d = 10000
numruns = 1
println("\nd = ", d, ", number of runs = ", numruns)
test_eigmax(d, numruns)
test_power_iteration(d, numruns)

println("\n\n------ eigmin vs inverse_iteration")
# Dimension = 10
d = 10
numruns = 100000
println("d = ", d, ", number of runs = ", numruns)
test_eigmin(d, numruns)
test_inverse_iteration(d, numruns)

# Dimension = 100
d = 100
numruns = 1000
println("\nd = ", d, ", number of runs = ", numruns)
test_eigmin(d, numruns)
test_inverse_iteration(d, numruns)

# Dimension = 10000
d = 10000
numruns = 1
println("\nd = ", d, ", number of runs = ", numruns)
test_eigmin(d, numruns)
test_inverse_iteration(d, numruns)

## TEST SYMMETRIC
# need to explore a bit more the effect of giving a Symmetric matrix as input...

A = gen_sym_matrix(20);
typeof(A)
B = Symmetric(A);
typeof(B) #Symmetric{Float64,Array{Float64,2}}
@elapsed eigmin(A)
@elapsed eigmin(B)
