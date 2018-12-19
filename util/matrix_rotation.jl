"""
    rotate_matrix(X)

Rotate a matrix by multiplying it on both sides by an orthogonal matrix 
obtained in the QR decomposition of another randomly generated matrix.
X_rotated = V * X * V^T
#INPUTS:\\
    - X: squared matrix\\
#OUTPUTS:\\
    - X_rotated: rotated matrix\\
"""
function rotate_matrix(X)
    szX = size(X);
    if(szX[1]!=szX[2])
        error("Not a squared matrix")
    end

    A = rand(szX[1], szX[1])
    V, R = qr(A)
    
    X_rotated = V*X*V'

    return X_rotated
end