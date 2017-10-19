
include("dataLoad.jl")
initDetails()

datasets = ["heart_scale"] #  w1a, SUSY,
for  dataset in datasets
transformDataJLD(dataset)
X,y = loadDataset(dataset) #
showDetails(dataset)
end
