include("misc.jl")
using LinearAlgebra
function leastSquaresRBFL2(X,y, sigma, lambda)
    n, d = size(X)
	# Add bias column
	Z = rbfBasis(X, X, sigma)
	# Find regression weights minimizing squared error
	w = (Z'*Z + lambda * I)\(Z'*y)

	# Make linear prediction function
	predict(Xtilde) = rbfBasis(Xtilde, X, sigma) * w

	# Return model
	return LinearModel(predict,w)
end

function rbfBasis(X_tilde, X, sigma)
t, d = size(X_tilde)
n = size(X, 1)
Z = zeros(t, n)
for i = 1:t
    for j = 1:n
        Z[i, j] = exp(-norm(X_tilde[i, :]- X[j, :])^2/ (2*sigma^2))
    end
end
return Z
end