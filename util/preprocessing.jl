
function fit_apply_datascaling(X, scaling::AbstractString)
  # centering

  rowscaling = [];
  colscaling =[];
  colsmean =[0.0];
  if(scaling == "column-scaling")
    stdX = std(X,2);
    ind = (0.==stdX); stdX[ind] =1.0; #     #replace 0 in std by 1 incase there is a constant feature
    colsmean =mean(X,2);
    X[:]= (X.-colsmean);
    X[:]= X./stdX; # Centering and scaling the data.
    colscaling = stdX.^(-1);
  elseif(scaling == "column-row-scaling")
    rowscaling, colscaling = knight_scaling(X);
  end

  if(scaling != "none")
   X = [ X ; ones(size(X,2))'];  # WAARNING: This reallocates X and so loses the pointer!
  end
  datascaling = DataScaling(rowscaling,colscaling,colsmean,scaling);
  return datascaling
end


function apply_datascaling(X, datascaling::DataScaling)
  rowscaling, colscaling = knight_scaling(X);
  X[:,:] = (X.-datascaling.colsmean);
  if(!isempty(datascaling.colscaling))
    X[:,:] = X.*(datascaling.colscaling);
  end
  # if(!isempty(datascaling.rowscaling))
  #   X[:,:] =  (datascaling.rowscaling).*X;
  # end
  X = [ X ; ones(size(X,2))'];
end
