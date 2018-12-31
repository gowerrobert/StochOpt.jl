using JLD
using CSV

"""
    csv2libsvm(path, skipheader)

Converts a csv file with COMMA-separated values in the format of the slice localization dataset, 
e.g. "feature_1,feature_2,...,feature_d,output"
# Ref: "https://archive.ics.uci.edu/ml/datasets/Relative+location+of+CT+slices+on+axial+axis#" 

#INPUTS:\\
    - **AbstractString** path: path to the csv file (with or without the extension ".csv")\\
    - **Bool** skipheader: skips the first file of the file if set to true\\
#OUTPUTS:
"""
function csv2libsvm(path::AbstractString, skipheader::Bool)
    ## Check if there is an extension in the name
    if occursin(".csv", path)
        path = split(path, ".csv")[1];
    end
    println("Path:", path);

    ## Skip the header or not
    if skipheader; startidx = 2; else startidx = 1; end

    ## Writing the dataset in the libsvm format
    fin = open("$(path).csv", "r")
    fout = open("$(path)_libsvm", "w")
    for line in enumerate(eachline(fin))
        if line[1] >= startidx # here line[1] is the line number
            line = split(line[2], ",");
            write(fout, line[end]);
            line = line[2:end-1] # skip first column: patientID
            for itm in enumerate(line)
                write(fout, " ", string(itm[1]), ":", itm[2]);
            end
            write(fout, "\n")
        end
    end
    close(fin);
    close(fout);
end

## Test on pyrim (cf LIBSVM) dataset subsample
pathtest = "/home/nidham/Downloads/datasets/subsample_pyrim";
pathtest2 = "/home/nidham/Downloads/datasets/subsample_pyrim.csv";

csv2libsvm(pathtest, true)
csv2libsvm(pathtest2, true)

## Application to slice localization
path = "/home/nidham/Downloads/datasets/slice_localization_data";
path2 = "/home/nidham/Downloads/datasets/slice_localization_data.csv";

csv2libsvm(path, true)
csv2libsvm(path2, true)