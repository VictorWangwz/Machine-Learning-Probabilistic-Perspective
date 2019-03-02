using SparseArrays
include("misc.jl")
include("findMin.jl")


function logReg(X,y)
	(n,d) = size(X)

	# Add bias and convert to sparse for speed
	X = sparse([ones(n,1) X])
	Xt = X'

	# The loss function assumes yi in [-1,1] so convert to this
	y[y .< .5] .= -1

	# Initial guess and hyper-parameter
	w = zeros(d+1,1)
	lambda = 1

	# Function we're going to minimize (and that computes gradient)
	funObj(w) = logisticObj(w,X,Xt,y,lambda)

	# Fit parameters
	w = findMin(funObj,w,derivativeCheck=false,verbose=false,maxIter=10)

	# Sample function (returns [0,1] values)
	function sampleFunc(xtilde)
		return rand() < 1 ./ (1+exp(-dot([1;xtilde],w)))
	end
	# Return model
	return SampleModel(sampleFunc)
end

function logisticObj(w,X,Xt,y,lambda)
	Z = 1 .+ exp.(-y.*(X*w))
	sigmoid = 1 ./ Z
	f = sum(log.(Z)) + (lambda/2)*dot(w,w)
	g = -(Xt*(y.*(1 .- sigmoid))) + lambda*w
	return (f,g)
end
