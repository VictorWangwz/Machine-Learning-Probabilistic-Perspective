include("misc.jl")
include("findMin.jl")
function softmaxObj(W, X, y)
    n, d = size(X)
    k = maximum(y)
    W = zeros(d, k)
    w_cxi = zeros(n)
    w_yi_xi = zeros(n)
    for i = 1:n
        w_yi_xi[i] =X[i]*  W[:, i]
        for c = 1:k
            w_cxi[i] +=exp(X[i]*W[:, c])
        end
    end
    f = sum(w_yi_xi) +sum(log(w_cxi))
end

function softmaxClassifier(X,y)
	(n,d) = size(X)
	k = maximum(y)

	# Each column of 'w' will be a logistic regression classifier
	W = zeros(d,k)

	for c in 1:k
		yc = ones(n,1) # Treat class 'c' as +1
		yc[y .!= c] .= -1 # Treat other classes as -1
		funObj(w) = softmaxObj(w,X,yc)

		W[:,c] = findMin(funObj,W[:,c],verbose=false)
	end

	# Make linear prediction function
	predict(Xhat) = mapslices(argmax,Xhat*W,dims=2)

	return LinearModel(predict,W)
end