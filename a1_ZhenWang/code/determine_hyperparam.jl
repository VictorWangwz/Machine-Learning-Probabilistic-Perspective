# Load X and y variable
using JLD
using LinearAlgebra
data = load("nonLinear.jld")
(X,y,Xtest,ytest) = (data["X"],data["y"],data["Xtest"],data["ytest"])

# Compute number of training examples and number of features
(n,d) = size(X)
mid = Int64(n/2)
Xtrain = X[1:mid,:]
ytrain = y[1:mid]
Xvalid = X[mid+1:end,:]
yvalid = y[mid+1:end]

# # Fit least squares model

include("rbfLeastSquares.jl")
let minError = Inf
    for sigma in 2.0.^(-20:20)
        for lambda in 2.0.^(-20:20)
            model = leastSquaresRBFL2(Xtrain, ytrain, sigma, lambda)

            # Report the error on the test set
            using Printf
            t = size(Xvalid,1)
            yhat = model.predict(Xvalid)
            testError = sum((yhat - yvalid).^2)/t
            if testError < minError
                minError = testError
                global optimalLambda = lambda
                global optiomalSigma = sigma
            end

        end
    end
    
    model = leastSquaresRBFL2(Xtrain, ytrain, optiomalSigma, optimalLambda)
    
    # Report the error on the test set
    using Printf
    # print(minError,"/n")
    # print(optimalLambda,"/n")
    print(optiomalSigma)
    t = size(Xvalid,1)
    yhat = model.predict(Xvalid)
    testError = sum((yhat - yvalid).^2)/t
    # Plot model
    using PyPlot
    figure()
    plot(X,y,"b.")
    plot(Xtest,ytest,"g.")
    Xhat = minimum(X):.1:maximum(X)
    Xhat = reshape(Xhat,length(Xhat),1) # Make into an n by 1 matrix
    yhat = model.predict(Xhat)
    plot(Xhat[:],yhat,"r")
    ylim((-300,400))
    
    display(gcf())
    @printf("TestError = %.2f\n",testError)
end
