function sinkhorn_scaling(A; colsum=1/size(A)[2], rowsum=1/size(A)[1], eps=10.0^(-3), maxiter=10)
    colsum = 1/size(A)[2]; rowsum = 1/size(A)[1]; eps = 10.0^(-3);
    A = A./maximum(A)
    sA = size(A);
    u = ones(sA[1]);
    v = ones(sA[2]);
    i = 1;
    while (i < maxiter && matscale_error(A, u, v, colsum, rowsum) > eps)
        u[:] = rowsum./ (A*v);
        v[:] = colsum./ (A'*u);
        i = i + 1;
    end
    B = u.*A.*v'
    [norm(sum(B, 1) - rowsum, 1), norm(sum(B, 2) - colsum, 2)]
    return u.*A.*v'
end

function knight_scaling(A; eps=10.0^(-3), maxiter=10)
    sA = size(A);
    i = 1;
    rowscaling = ones(1,sA[2]);
    colscaling = ones(sA[1],1);
    while (i < maxiter) #&& matscale_error(A,u,v,colsum,rowsum) > eps
        rs = sqrt(sum(A.^2,1)).^(-1);
        cs = sqrt(sum(A.^2,2)).^(-1);
        A[:] = rs.* (A .* cs);
        rowscaling = rowscaling.*rs;
        colscaling = colscaling.*cs;
        # B[:] = (sqrt(sum(B,1)).^(-1)).* (B .* ((sqrt(sum(B,2))).^(-1))); # for constant L1
        i = i + 1;
    end
    return rowscaling, colscaling
end

function sinkhorn_scaling_log_exp(A; colsum=1, rowsum=1, eps=10.0^(-3), maxiter=10)
    sA = size(A);
    f = zeros(sA[1]);
    g = zeros(sA[2]);
    ferr = zeros(maxiter);
    gerr = zeros(maxiter);
    i = 1;
    while (i < maxiter && (matscale_error(A, exp(f)./eps, exp(g)./eps, colsum,rowsum) > eps))
        u = exp(g./eps);
        v = exp(f./eps);
        ferr[i] = norm(sum(u'.*A.*v, 2) - colsum, 2);
        f[:] = eps.*log(rowsum) - eps.*log(A*(exp(g./eps)));
        gerr[i] = norm(sum(u'.*A.*v,1) - rowsum, 1);
        g[:] = eps.*log(colsum) - eps.*log(A'*(exp(f./eps)));
        i = i + 1;
    end
    u = exp(g./eps);
    v = exp(f./eps);
    B = u'.*A.*v
    [norm(sum(B,1) - rowsum, 1), norm(sum(B, 2) - colsum, 2)]
    return u.*A.*v'
end

function matscale_error(A, u, v, colsum, rowsum)
    return norm(sum(u.*A.*v', 1) - rowsum, 1) + norm(sum(u.*A.*v',2) - colsum, 2)
end

function double_diag_scaling(A; epsilon=10^(-3), maxiter=10)
    sA = size(A);
    u = ones(1, sA[1]);
    v = ones(1, sA[2]);
    i = 1;
    verr = zeros(maxiter);
    uerr = zeros(maxiter);
    while (i < maxiter) #  || matscale_error(A, colsum, rowsum) > epsilon
        u[:] = 1./(sqrt(sum((A.*v).^2, 2)))';
        verr[i] = norm(v.* sqrt(sum((A'.*u).^2, 2))', 1);
        v[:] = 1./sqrt(sum((A'.*u).^2, 2))';
        uerr[i] = norm(u.* (sqrt(sum((A.*v).^2, 2)))', 1);
        i = i + 1;
    end
    B = u'.*A.*v
    return u'.*A.*v
end