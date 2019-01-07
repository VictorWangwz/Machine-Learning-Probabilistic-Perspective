include("misc.jl")
include("findMin.jl")
function softmaxObj(w, X, y, k)
	n, d = size(X)
	W = reshape(w, d, k)
	XW = X*W
	Z = sum(exp.(XW), dims=2)
	f = 0
	G = zeros(d, k)
    for i in 1:n
		f += -XW[i, y[i]] + log(Z[i])
		p = exp.(XW[i,:])./Z[i]
        for c = 1:k
            G[:,c] +=X[i,:]*(p[c]-(y[i]==c))
        end
    end
    return (f, reshape(G, d*k, 1))
end

function softmaxClassifier(X,y)
	(n,d) = size(X)
	k = maximum(y)
	W = zeros(d,k)
	funObj(w) = softmaxObj(w,X,y, k)
	W[:] = findMin(funObj, W[:], derivativeCheck=true, maxIter=1000, verbose=false)
	predict(Xhat) = mapslices(argmax,Xhat*W,dims=2)

	return LinearModel(predict,W)
end