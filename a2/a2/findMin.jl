using Printf, LinearAlgebra
include("misc.jl")

function findMin(funObj,w;maxIter=100,epsilon=1e-2,derivativeCheck=false,verbose=true)
	# funObj: function that returns (objective,gradient)
	# w: initial guess
	# maxIter: maximum number of iterations
	# epsilon: stop if the gradient gets below this
	# derivativeCheck: whether to check against numerical gradient

	# Evalluate the intial objective and gradient
	(f,g) = funObj(w)

	# Optionally check if gradient matches finite-differencing
	if derivativeCheck
		g2 = numGrad(funObj,w)

		if maximum(abs.(g-g2)) > 1e-4
			@show([g g2])
			@printf("User and numerical derivatives differ\n")
			sleep(1)
		else
			@printf("User and numerical derivatives agree\n")
		end
	end

	# Initial step size and sufficient decrease parameter
	gamma = 1e-4
	alpha = 1
	for i in 1:maxIter

		# Try out the current step-size
		wNew = w - alpha*g
		(fNew,gNew) = funObj(wNew)

		# Decrease the step-size if we increased the function
		gg = dot(g,g)
		while fNew > f - gamma*alpha*gg

			if verbose
				@printf("Backtracking\n")
			end

			# Fit a degree-2 polynomial to set step-size
			alpha = alpha^2*gg/(2(fNew - f + alpha*gg))

			# Try out the smaller step-size
			wNew = w - alpha*g
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
		gradNorm = norm(g,Inf)
		if verbose
			@printf("%6d %15.5e %15.5e %15.5e\n",i,alpha,f,gradNorm)
		end

		# We want to stop if the gradient is really small
		if gradNorm < epsilon
			if verbose
				@printf("Problem solved up to optimality tolerance\n")
			end
			return w
		end
	end
	if verbose
		@printf("Reached maximum number of iterations\n")
	end
	return w
end


function findMinL1(funObj,w,lambda;maxIter=100,epsilon=1e-2)
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
		wNew = w - alpha*g

		# Proximal step on non-smooth part
		wNew = sign.(wNew).*max.(abs.(wNew) .- lambda*alpha,0)
		(fNew,gNew) = funObj(wNew)

		# Decrease the step-size if we increased the function
		gtd = dot(g,wNew-w)
		while fNew + lambda*norm(wNew,1) > f + lambda*norm(w,1) - gamma*alpha*gtd
			@printf("Backtracking\n")
			alpha /= 2

			# Try out the smaller step-size
			wNew = w - alpha*g
			wNew = sign.(wNew).*max.(abs.(wNew) .- lambda*alpha,0)
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

function proxGradGroupL1(funObj,w,d,k,lambda;maxIter=100,epsilon=1e-2)
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
		wNew = w - alpha*g

		# Proximal step on non-smooth part
		wNew = proximalStep(wNew,lambda,alpha,d,k)
		(fNew,gNew) = funObj(wNew)

		# Decrease the step-size if we increased the function
		gtd = dot(g,wNew-w)
		while fNew + lambda*groupNorm(wNew,d,k) > f + lambda*groupNorm(w,d,k) - gamma*alpha*gtd
			@printf("Backtracking\n")
			alpha /= 2

			# Try out the smaller step-size
			wNew = w - alpha*g
			wNew = proximalStep(wNew, lambda, alpha, d, k)
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
		optCond = groupNorm(w-proximalStep(w-g ,alpha, lambda, d, k), d, k)
		@printf("%6d %15.5e %15.5e %15.5e\n",i,alpha,f+lambda*groupNorm(w,d,k),optCond)

		# We want to stop if the gradient is really small
		if optCond < epsilon
			@printf("Problem solved up to optimality tolerance\n")
			return w
		end
	end
	@printf("Reached maximum number of iterations\n")
	return w
end

function proximalStep(w, lambda, alpha, d, k)
	w = reshape(w, d, k)
	for i in 1:d
		ng = norm(w[i,:])
		if ng != 0
			w[i,:] = w[i,:]/ng * max(0,ng-lambda*alpha)
		end
	end
	w = reshape(w, d*k, 1)
end

function groupNorm(w, d, k)
	w = reshape(w, d, k)
	ng = 0
	for i in 1:d
		ng += norm(w[i,:])
	end
	w = reshape(w, d*k, 1)
	return ng
end
