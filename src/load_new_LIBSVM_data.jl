
include("dataLoad.jl")
initDetails()

datasets = ["covtype"] #  w1a, SUSY, pendigits, heart
for dataset in datasets
    transformDataJLD(dataset)
    X,y = loadDataset(dataset) #
    showDetails(dataset)
end
