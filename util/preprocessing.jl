function fit_apply_datascaling(X, scaling::AbstractString)
    ## WARNING: mutating function without "!" in its name 
    # centering
    rowscaling = [];
    colscaling =[];
    colsmean = [0.0];
    if(scaling == "column-scaling")
        stdX = std(X, dims=2);
        if any(stdX .== 0)
            X = remove_constant_features(X);
            stdX = std(X, dims=2); 
        end
    #   ind = (0. == stdX);
        ind = findall((in)(stdX), 0.); # Reparing the error
        stdX[ind] .= 1.0; # replace 0 in std by 1 in case there is a constant feature
        colsmean = mean(X, dims=2);
        X = (X .- colsmean);
        X = X ./ stdX; # Centering and scaling the data.
        colscaling = stdX.^(-1);
    elseif(scaling == "column-row-scaling")
        rowscaling, colscaling = knight_scaling(X);
    end

    if(scaling != "none")
        X = [X; ones(size(X, 2))'];  # WARNING: This reallocates X and so loses the pointer!
    end
    datascaling = DataScaling(rowscaling, colscaling, colsmean, scaling);
    return X, datascaling
end

function apply_datascaling(X, datascaling::DataScaling)
    rowscaling, colscaling = knight_scaling(X);
    X[:, :] = (X.-datascaling.colsmean);
    if(!isempty(datascaling.colscaling))
        X[:, :] = X.*(datascaling.colscaling);
    end
    # if(!isempty(datascaling.rowscaling))
    #   X[:,:] =  (datascaling.rowscaling).*X;
    # end
    X = [X; ones(size(X,2))'];
end

function remove_constant_features(X)
    stdX = std(X, dims=2)[:];
    nullidx = findall(stdX .== 0);
    complementaryidx = setdiff(1:size(X,1), nullidx);
    return X[complementaryidx, :]
end