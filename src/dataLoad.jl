################
# INSTRUCTIONS #
################

# 1) Install the JLD package using Pkg.add("JLD")
# 2) Change the default_path so that it points to your data directory
# 3) Use the naming convention *.data for your data files (e.g. a9a.data)
# 4) Run include("dataLoad.jl")
# 5) Run initDetails()
#
# After the initial steps, you can load a single dataset by following these steps
#
# 1) Choose a dataset (e.g. dataset = "a9a") and run transformDataJLD(dataset)
#    (it is sufficient to do this only once for each dataset)
# 2) Load the dataset using X,y = loadDataset(dataset)
# 3) Observe the details of the dataset using showDetails(dataset)
# 4) Enjoy.

using JLD
using SparseArrays # julia 0.7
using Statistics # julia 0.7

# this needs to be changed to your personal path.
#default_path = "/local/rgower/git/online_saga/julia-online-SAGA/data/" #/local/rgower/libsvmdata/"
# default_path = "./data/"; #"/home/robert/git/online_saga/julia-online-SAGA/data/"

function initDetails(default_path::AbstractString) # creates a blank details dictionary
    details = Dict();
    save("$(default_path)details.jld", "details", details)
end

function transformDataJLD(default_path::AbstractString, dataset, classification) # transforms LIBSVM to JLD for faster loading
    X, y = readLIBSVM(string(default_path, "$(dataset)"), classification) # classification=false leads to regression
    save("$(default_path)$(dataset).jld", "X", X, "y", y)
    saveDetails(default_path, dataset, X)
end

function loadDataset(default_path::AbstractString, dataset) # once transformed, load the dataset using JLD
    try
        X, y = load("$(default_path)$(dataset).jld", "X", "y");
        println("$(default_path)$(dataset).jld");
        return X, y
    catch loaderror
        println(loaderror);
        error("Check the list of available datasets in: \"$(default_path)available_datasets.txt\"");
    end
end

function saveDetails(default_path::AbstractString, dataset::String, X::SparseMatrixCSC{Float64,Int64}) # saves the details of the dataset
    details = load("$(default_path)details.jld", "details")
    detail = Dict(:dims => size(X, 1), :n => size(X, 2), :sparsity => nnz(X)/(size(X, 1)*size(X, 2)))
    details[dataset] = detail
    save("$(default_path)details.jld", "details", details)
end

function showDetails(default_path::AbstractString, dataset) # shows the details of the dataset
    return load("$(default_path)details.jld", "details")[dataset]
end

function readLIBSVM(fname::AbstractString, classification::Bool) # the function to read the standard LIBSVM format
    # classification = true leads to transformation of labels into {-1,1}
    # classfication = false keeps the original labels
    b = Float64[]
    Ir = Int64[]
    Jr = Int64[]
    Pr = Float64[]
    fi = open("$fname", "r")
    n = 1
    for line in eachline(fi)
        line = split(line, " ")
        append!(b, [parse(Float64, line[1])]) # julia 0.7 `float(x::AbstractString)` is deprecated, use `parse(Float64, x)` instead.
        line = line[2:end]
        for itm in line
            if !(strip(itm) == "")
                itm = split(itm, ":")
                append!(Ir, [n])
                append!(Jr, [parse(Int, strip(itm[1]))])
                # append!(Jr, [Meta.parse(Int, strip(itm[1]))]) # julia 0.7
                append!(Pr, [parse(Float64, strip(itm[2]))])
            end
        end
        n += 1
    end
    close(fi)
    if classification
        mb = mean(b)
        for i=1:length(b)
            b[i] = (b[i] > mb) ? 1. : -1. # julia 0.7
        end
    end
    A = sparse(round.(Integer, Ir), round.(Integer, Jr), Pr)
    if size(A)[1] == length(b)
        A = A'
    end
    return SparseMatrixCSC(A), b
end
