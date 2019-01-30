# Load X and y variable
using JLD, Printf, LinearAlgebra
data = load("quantum.jld")
(X,y) = (data["X"],data["y"])

# Add bias variable, initialize w, set regularization and optimization parameters
(n,d) = size(X)
lambda = 1

# Initialize
maxPasses = 10
progTol = 1e-4
verbose = true
w = zeros(d,1)
L = maximum(sum(X.^2,dims=1))/4+lambda;
lambda_i = lambda/n # Regularization for individual example in expectation
# w_store = zeros(maxPasses*n,d)
# Start running stochastic gradient
delta = 1
D = fill(delta, (d,1))
w_old = copy(w);

# Q4
v = zeros(n, d)
g = zeros(d, 1)
for k in 1:maxPasses*n
    # w_store[k, :] = w
    # Choose example to update 'i'
    i = rand(1:n)

    # Compute gradient for example 'i'
    r_i = -y[i]/(1+exp(y[i]*dot(w,X[i,:])))
    g_i = r_i*X[i,:] + (lambda_i)*w
    global g  = g - v[i, :] + g_i
    v[i,:] = g_i
    # Choose the step-size
    # alpha = 1/(lambda_i*k)
    # Q1. Manually choose alpha
    alpha = 10^-3
    # # Q3. use AdaGrad
    # D_new  = D
    # D_new += g_i .^2
    # D_new = 1 ./sqrt.(D_new)
    # Take thes stochastic gradient step
    # global w -= alpha*D_new .* g_i
    global w -= alpha/n* g
    # Q2.
    # aver = sum(w)/d
    # w = fill(aver, size(w))

    # Check for lack of progress after each "pass"
    if mod(k,n) == 0
        yXw = y.*(X*w)
        f = sum(log.(1 .+ exp.(-y.*(X*w)))) + (lambda/2)norm(w)^2
        delta = norm(w-w_old,Inf);
        if verbose
            @printf("Passes = %d, function = %.4e, change = %.4f\n",k/n,f,delta);
        end
        if delta < progTol
            @printf("Parameters changed by less than progTol on pass\n");
            break;
        end
        global w_old = copy(w);
    end
end
