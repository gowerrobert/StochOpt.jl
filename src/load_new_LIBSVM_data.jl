# include("./dataLoad.jl")
include("../src/dataLoad.jl")
initDetails()

default_path = "./data/";

datasets = ["slice"] #  w1a, SUSY, pendigits, heart, YearPredictionMSD_full, leukemia_full, news20.binary, covtype.binary, ijcnn1_full
# leukemia_full is the concatenation of leu and leu.t (resp. training and test sets)

## WARNING: be careful and select properly the following setting!
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