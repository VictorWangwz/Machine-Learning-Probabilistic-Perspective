include("misc.jl")
using MathProgBase, Clp
using LinearAlgebra
function leastMax(X,y)

	# Add bias column
	(n,d) = size(X)
	Z = [ones(n,1) X]
    print(size(Z))
    rx = [zeros(d+1,1); ones(1,1)]
    A = [Z -ones(n,1); -Z -ones(n,1)]
    b = [y; -y]
    print(size(A))
    sense = fill(-Inf, (2*n, 1))
    l = fill(-Inf, (d+2, 1))
    u = fill(Inf, (d+2, 1))
    sol = linprog(vec(rx), A, vec(sense), vec(b), vec(l),vec(u), ClpSolver())
    print(sol.sol)
    w = sol.sol[1:d+1,:]
    predict(Xtilde) = [ones(size(Xtilde,1),1) Xtilde]*w
	# Return model
	return LinearModel(predict,w)
end
