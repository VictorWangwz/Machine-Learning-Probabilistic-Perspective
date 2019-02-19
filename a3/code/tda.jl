using LinearAlgebra
include("studentT.jl")
include("misc.jl")

function tda(X, Y)
	(n, d) = size(X)
	K = maximum(Y)

	theta = zeros(K)
	subModel = Array{DensityModel}(undef, K)
	for c in 1:K
		theta[c] = sum(y.==c)/n
		subModel[c] = studentT(X[y.==c, :])
	end
	
	function predict(Xhat)
		(t,d) = size(Xhat)
		Yhat = zeros(t,1)

		PDF = zeros(t,K)
		for c in 1:K
			PDF[:, c] = subModel[c].pdf(Xhat)
		end
		
		for i in 1:t
			logp = zeros(K)
			for c in 1:K
				logp[c] = log(theta[c]) + log(PDF[i,c])
			end
			(~, Yhat[i]) = findmax(logp)
		end

		return Yhat
	end
	return GenericModel(predict)
end
