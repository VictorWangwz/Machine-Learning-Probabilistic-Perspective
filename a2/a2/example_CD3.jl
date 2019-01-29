# Load X and y variable
using JLD, Printf, LinearAlgebra
include("help.jl")
data = load("binaryData.jld")
(X,y) = (data["X"],data["y"])

# Add bias variable, initialize w, set regularization and optimization parameters
(n,d) = size(X)
X = [ones(n,1) X]
d += 1
w = zeros(d,1)
lambda = 1
maxPasses = 500
progTol = 1e-4
verbose = true

## Run and time coordinate descent to minimize L2-regularization logistic loss

# Start timer
time_start = time_ns()

gamma = 1e-4
alpha = 1

# Start running coordinate descent
w_old = copy(w);
r = X*w - y
f = (1/2)norm(r)^2 + (lambda/2)norm(w)^2
for k in 1:maxPasses*d

    # Choose variable to update 'j'
    j = rand(1:d)

    Xj = X'[j,:]
    g_j =(reshape(Xj, 1, length(Xj))*r)[1,1] + lambda*w[j]

    # Try out the current step-size
    wNew = copy(w)
    wNew[j] -= alpha*g_j
    (fNew, rNew) = getfNew(X, wNew, alpha, g_j, j, r, lambda)

    # Decrease the step-size if we increased the function
    gg = g_j*g_j
    while fNew > f - gamma*alpha*gg

	# Fit a degree-2 polynomial to set step-size
	global alpha = alpha^2*gg/(2(fNew - f + alpha*gg))
	# Try out the smaller step-size
	wNew = copy(w)
	wNew[j] -= alpha*g_j
	(fNew, rNew) = getfNew(X, wNew, alpha, g_j, j, r, lambda)
    end

    # Guess the step-size for the next iteration
    sub = getNewGj(X, wNew, alpha, g_j, j, r, lambda) - g_j
    global alpha *= -(sub)/g_j
    global w = wNew
    global f = fNew
    global r = rNew

    # Check for lack of progress after each "pass"
    # - Turn off computing 'f' and printing progress if timing is crucial
    if mod(k,d) == 0
        global r = X*w - y
        global f = (1/2)norm(r)^2 + (lambda/2)norm(w)^2
        delta = norm(w-w_old,Inf);
        if verbose
            @printf("Passes = %d, function = %.4e, change = %.4f\n",k/d,f,delta);
        end
        if delta < progTol
            @printf("Parameters changed by less than progTol on pass\n");
            break;
        end
        global w_old = copy(w);
    end

end

# End timer
@printf("Seconds = %f\n",(time_ns()-time_start)/1.0e9)
