include("misc.jl")
include("findMin.jl")

function logReg(X,y)

	(n,d) = size(X)

	# Initial guess
	w = zeros(d,1)

	# Function we're going to minimize (and that computes gradient)
	funObj(w) = logisticObj(w,X,y)

	# Solve least squares problem
	w = findMin(funObj,w,derivativeCheck=true)

	# Make linear prediction function
	predict(Xhat) = sign.(Xhat*w)

	# Return model
	return LinearModel(predict,w)
end

function logisticObj(w,X,y)
	yXw = y.*(X*w)
	f = sum(log.(1 .+ exp.(-yXw)))
	g = -X'*(y./(1 .+ exp.(yXw)))
	return (f,g)
end

# Multi-class one-vs-all version (assumes y_i in {1,2,...,k})
function logRegOnevsAll(X,y)
	(n,d) = size(X)
	k = maximum(y)

	# Each column of 'w' will be a logistic regression classifier
	W = zeros(d,k)

	for c in 1:k
		yc = ones(n,1) # Treat class 'c' as +1
		yc[y .!= c] .= -1 # Treat other classes as -1

		# Each binary objective has the same features but different lables
		funObj(w) = logisticObj(w,X,yc)

		W[:,c] = findMin(funObj,W[:,c],verbose=false)
	end

	# Make linear prediction function
	predict(Xhat) = mapslices(argmax,Xhat*W,dims=2)

	return LinearModel(predict,W)
end

# Multi-class softmax version (assumes y_i in {1,2,...,k})
function logRegSoftmaxL2(X,y,lambda)
	(n,d) = size(X)
	k = maximum(y)

	# Each column of 'w' will be a logistic regression classifier
	W = zeros(d,k)

	funObj(w) = softmaxObjL2(w,X,y,k,lambda)

	W[:] = findMin(funObj,W[:],derivativeCheck=true,maxIter=500)

	# Make linear prediction function
	predict(Xhat) = mapslices(argmax,Xhat*W,dims=2)

	return LinearModel(predict,W)
end

function softmaxObjL2(w,X,y,k,lambda)
	(n,d) = size(X)

	W = reshape(w,d,k)

	XW = X*W
	Z = sum(exp.(XW),dims=2)

	nll = 0
	G = zeros(d,k)
	for i in 1:n
		nll += -XW[i,y[i]] + log(Z[i])

		pVals = exp.(XW[i,:])./Z[i]
		for c in 1:k
			G[:,c] += X[i,:]*(pVals[c] - (y[i] == c))
		end
	end
	return (nll+lambda/2*norm(w)^2,reshape(G,d*k,1)+lambda*w)
end

# Multi-class softmax version (assumes y_i in {1,2,...,k})
function logRegSoftmax(X,y)
	(n,d) = size(X)
	k = maximum(y)

	# Each column of 'w' will be a logistic regression classifier
	W = zeros(d,k)

	funObj(w) = softmaxObj(w,X,y,k)

	W[:] = findMin(funObj,W[:],derivativeCheck=true,maxIter=500)

	# Make linear prediction function
	predict(Xhat) = mapslices(argmax,Xhat*W,dims=2)

	return LinearModel(predict,W)
end

function softmaxObj(w,X,y,k)
	(n,d) = size(X)

	W = reshape(w,d,k)

	XW = X*W
	Z = sum(exp.(XW),dims=2)

	nll = 0
	G = zeros(d,k)
	for i in 1:n
		nll += -XW[i,y[i]] + log(Z[i])

		pVals = exp.(XW[i,:])./Z[i]
		for c in 1:k
			G[:,c] += X[i,:]*(pVals[c] - (y[i] == c))
		end
	end
	return (nll,reshape(G,d*k,1))
end

# Multi-class softmax version (assumes y_i in {1,2,...,k})
function logRegSoftmaxL1(X,y,lambda)
	(n,d) = size(X)
	k = maximum(y)

	# Each column of 'w' will be a logistic regression classifier
	W = zeros(d,k)

	funObj(w) = softmaxObj(w,X,y,k)

	W[:] = findMinL1(funObj,W[:],lambda,maxIter=500)

	# Make linear prediction function
	predict(Xhat) = mapslices(argmax,Xhat*W,dims=2)

	return LinearModel(predict,W)
end

function softmaxObjL1(w,X,y,k,lambda)
	(n,d) = size(X)

	W = reshape(w,d,k)

	XW = X*W
	Z = sum(exp.(XW),dims=2)

	nll = 0
	G = zeros(d,k)
	for i in 1:n
		nll += -XW[i,y[i]] + log(Z[i])

		pVals = exp.(XW[i,:])./Z[i]
		for c in 1:k
			G[:,c] += X[i,:]*(pVals[c] - (y[i] == c))
		end
	end
	return (nll,reshape(G,d*k,1))
end


function softmaxClassifierGL1(X,y,lambda)
	(n,d) = size(X)
	k = maximum(y)

	# Each column of 'w' will be a logistic regression classifier
	W = zeros(d,k)

	funObj(w) = softmaxObj(w,X,y,k)
	W[:] = proxGradGroupL1(funObj,W[:],lambda,d,k,maxIter=500)

	# Make linear prediction function
	predict(Xhat) = mapslices(argmax,Xhat*W,dims=2)

	return LinearModel(predict,W)

end




function softThreshhold(w, d, k, threshold)
	# k groups
	# d features
	v = zeros(k)
	n = length(w)
	for j in 1:k
		for i in 1:d
			v[j] +=w[(i-1)*5+j]^2
		end
		
	end
	v = sqrt.(v)
	wNew = zeros(n,1)
	for j in 1:k
		if v[j] !=0
			for i in 1:d
				wNew[(i-1)*5+j] = abs(w[(i-1)*5+j]/v[j]) * maximum([0 v[j]-threshold])
			end
		end
	end
	return wNew

end


function proxGradGroupL1(funObj,w,lambda, d, k;maxIter=100,epsilon=1e-2)
	# funObj: function that returns (objective,gradient)
	# w: initial guess
	# lambda: value of L1-regularization parmaeter
	# maxIter: maximum number of iterations
	# epsilon: stop if the gradient gets below this

	# Evalluate the intial objective and gradient
	(f,g) = funObj(w)

	# Initial step size and sufficient decrease parameter
	gamma = 1e-4
	alpha = 1
	for i in 1:maxIter

		# Gradient step on smoooth part
		wNew = softThreshhold(w-alpha*g, d, k, alpha*lambda)
		# print(wNew,"/n")
		# Proximal step on non-smooth part
		(fNew,gNew) = funObj(wNew)

		# Decrease the step-size if we increased the function
		gtd = dot(g,wNew-w)
		while fNew + lambda*groupNorm(wNew,d,k) > f + lambda*groupNorm(wNew, d, k) - gamma*alpha*gtd
			@printf("Backtracking\n")
			alpha /= 2

			# Try out the smaller step-size
			wNew = softThreshhold(w- alpha*g, d, k, alpha*lambda)
			(fNew,gNew) = funObj(wNew)
		end

		# Guess the step-size for the next iteration
		y = gNew - g
		alpha *= -dot(y,g)/dot(y,y)

		# Sanity check on the step-size
		if (!isfinitereal(alpha)) | (alpha < 1e-10) | (alpha > 1e10)
			alpha = 1
		end

		# Accept the new parameters/function/gradient
		w = wNew
		f = fNew
		g = gNew

		# Print out some diagnostics
		optCond = norm(w-sign.(w-g).*max.(abs.(w-g) .- lambda,0),Inf)
		@printf("%6d %15.5e %15.5e %15.5e\n",i,alpha,f+lambda*norm(w,1),optCond)

		# We want to stop if the gradient is really small
		if optCond < epsilon
			@printf("Problem solved up to optimality tolerance\n")
			return w
		end
	end
	@printf("Reached maximum number of iterations\n")
	return w
end


function groupNorm(w, d, k)
	v = zeros(k)
	n = length(w)
	for j in 1:k
		for i in 1:d
			v[j] +=w[(i-1)*5+j]^2
		end
		
	end
	v = sqrt.(v)
	return sum(v)
end