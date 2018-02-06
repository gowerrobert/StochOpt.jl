
include("dataLoad.jl")
initDetails()

datasets = ["liver-disorders"] #  w1a, SUSY,
for  dataset in datasets
transformDataJLD(dataset)
X,y = loadDataset(dataset) #
showDetails(dataset)
end
