include("marginalCK.jl")

function forwardBackwards(p0, pT, xd, d)
	M = marginalCK(p1, pT, d)
	n = size(p0, 1)
	V = zeros(n, d)

	V[xd,d] = 1
	for j in d-1:-1:1
		V[:, j] = pT*V[:, j+1]
	end 

	MV = M.*V
	for j in 1:d
		MV[:, j] ./= sum(MV[:, j])
	end
	return MV
end
