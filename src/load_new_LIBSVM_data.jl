# include("./dataLoad.jl")
include("../src/dataLoad.jl")
initDetails()

default_path = "./data/";

datasets = ["housing"] #  w1a, SUSY, pendigits, heart, YearPredictionMSD
# leukemia is the concatenation of leu and leu.t (resp. training and test sets)
classification = false; # Regression
# classification = true; # Binary classification
for dataset in datasets
    try
        X, y = loadDataset(dataset);
        saveDetails(dataset, X); # In case it was not previously done
        println("The following data set already exists in: ", "$(default_path)$(dataset).jld");
    catch loaderror
        println(loaderror);
        transformDataJLD(dataset, classification); # 2nd arg = classification
        X, y = loadDataset(dataset);
    end
    lines = readlines("$(default_path)available_datasets.txt");
    if !(dataset in lines)
        println("Writing its name in: ", "$(default_path)available_datasets.txt");
        open("$(default_path)available_datasets.txt", "a") do file
            write(file, "$(dataset)\n");
        end
    end
    showDetails(dataset);
end
load("$(default_path)details.jld")