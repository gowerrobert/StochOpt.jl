function gen_gauss_data(numfeatures::Int64, numdata::Int64; lambda=1, err=0.001)
    X = rand(numfeatures, numdata);
    y = X'*rand(numfeatures) .+ err*rand(numdata);
    probname = string("gauss-", numfeatures, "-", numdata, "-", lambda);
    return X, y, probname
end

function gen_diag_data(numdata::Int64; lambda=1, Lmax=numdata, err=0.001)
    # X = diagm([1; (1.0:1.0:numdata-1).*(Lmax/numdata)]); # Robert's implementation
    X = diagm(0 => sqrt.([1; (1.0:1.0:numdata-2).*(Lmax/numdata); Lmax])); # julia 0.7 equiv to 'Matrix(Diagonal(v))'
    # X = diagm(sqrt.([1; (1.0:1.0:numdata).*(Lmax/numdata)])); # Nidham's implementation
    y = X'*rand(numdata) .+ err*rand(numdata);
    probname = string("diagints-", numdata, "-", lambda, "-", Lmax);
    return X, y, probname
end

function gen_gauss_scaled_data(numfeatures::Int64, numdata::Int64; lambda=1, Lmin=1.0/100.0, err=0.001)
    X = rand(numfeatures,numdata);
    colnorms =sum(X.^2, 1);
    X[:, :]= X./sqrt.(colnorms);
    X[:, 1] = X[:, 1]/Lmin;
    X[:, :] = X[:, :]*Lmin;
    y = X'*rand(numfeatures) .+ err*rand(numdata);
    probname = string("gaussscal-", numfeatures, "-", numdata, "-", lambda, "-", Lmin);
    return X, y, probname
end

function gen_diag_alone_eig_data(numfeatures::Int64, numdata::Int64; lambda=1, a=1, err=0.001)
    X = diagm(0 => [ones(numdata-1); a]); # julia 0.7
    y = X'*rand(numdata) .+ err*rand(numdata);
    probname = string("diagalone-", numdata, "-", lambda, "-", a);
    return X, y, probname
end
