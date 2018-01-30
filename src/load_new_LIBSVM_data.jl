
include("dataLoad.jl")
initDetails()

datasets = ["australian"] #  w1a, SUSY,
for  dataset in datasets
transformDataJLD(dataset)
X,y = loadDataset(dataset) #
showDetails(dataset)
end
