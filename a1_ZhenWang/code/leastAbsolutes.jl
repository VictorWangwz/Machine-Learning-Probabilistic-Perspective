include("misc.jl")
using MathProgBase, Clp
using LinearAlgebra
function leastAbsolutes(X,y)

	# Add bias column
	(n,d) = size(X)
	Z = [ones(n,1) X]
    print(size(Z))
    rx = [zeros(d+1,1); ones(n,1)]
    A = [Z -I; -Z -I]
    b = [y; -y]
    print(size(A))
    sense = fill(-Inf, (2*n, 1))
    l = fill(-Inf, (d+1+n, 1))
    u = fill(Inf, (d+1+n, 1))
    sol = linprog(vec(rx), A, vec(sense), vec(b), vec(l),vec(u), ClpSolver())
    print(sol.sol)
    w = sol.sol[1:d+1,:]
    # Add bias column
    predict(Xtilde) = [ones(size(Xtilde,1),1) Xtilde]*w
	# Return model
	return LinearModel(predict,w)
end

