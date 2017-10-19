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

# this needs to be changed to your personal path.
#default_path = "/local/rgower/git/online_saga/julia-online-SAGA/data/" #/local/rgower/libsvmdata/"
default_path = "./data/"; #"/home/robert/git/online_saga/julia-online-SAGA/data/"

function initDetails() # creates a blank details dictionary
   details = Dict()
   save("$(default_path)details.jld","details", details)
 end

function transformDataJLD(dataset) # transforms LIBSVM to JLD for faster loading
  details = load( "$(default_path)details.jld", "details")
  X,y = readLIBSVM(string(default_path, "$(dataset)"), true) # false leads to regression
  save("$(default_path)$(dataset).jld", "X", X, "y", y)
  detail = Dict(:dims => size(X,1), :n => size(X,2), :sparsity => nnz(X)/(size(X,1)*size(X,2)))
  details[dataset] = detail
  save("$(default_path)details.jld","details", details)
end

function loadDataset(dataset) # once transformed, load the dataset using JLD
  println("$(default_path)$(dataset).jld")
  return load("$(default_path)$(dataset).jld", "X", "y")
end

function showDetails(dataset) # shows the details of the dataset
  return load( "$(default_path)details.jld", "details")[dataset]
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
    append!(b, [float(line[1])])
    line = line[2:end]
    for itm in line
      if !(strip(itm) == "")
        itm = split(itm, ":")
        append!(Ir,[n])
        append!(Jr, [parse(Int,strip(itm[1]))])
        append!(Pr, [float(strip(itm[2]))])
      end
    end
    n += 1
  end
  close(fi)
  if classification
    mb = mean(b)
    for i=1:length(b)
      b[i] = (b[i] > mb)? 1. : -1.
    end
  end
  A = sparse(round(Integer,Ir), round(Integer,Jr), Pr)
  if size(A)[1] == length(b)
    A = A'
  end
  return A,b
end
