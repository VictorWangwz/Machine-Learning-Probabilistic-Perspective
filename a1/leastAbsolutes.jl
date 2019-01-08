include("misc.jl")
using MathProgBase, GLPKMathProgInterface, Clp
using LinearAlgebra
function leastAbsolutes(X,y)

	# Add bias column
	(n,d) = size(X)
	Z = [ones(n,1) X]
    print(size(Z))
    c = [zeros(d+1,1); ones(n,1)]
    A = [Z -I; -Z -I]
    b = [y; -y]
    print(size(A))
    sol = linprog(vec(c), A, vec(fill(-Inf, (2*n, 1))), vec(b), vec(fill(-Inf, (d+n+1, 1))),vec(fill(Inf, (d+n+1, 1))), ClpSolver())
    print(sol.sol)
    w = sol.sol[1:d+1,:]
    predict(Xtilde) = [ones(size(Xtilde,1),1) Xtilde]*w
	# Return model
	return LinearModel(predict,w)
end

