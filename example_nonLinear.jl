# Load X and y variable
using JLD
data = load("nonLinear.jld")
(X,y,Xtest,ytest) = (data["X"],data["y"],data["Xtest"],data["ytest"])

# Compute number of training examples and number of features
(n,d) = size(X)

# # Fit least squares model
# include("leastSquares.jl")
# model = leastSquares(X,y)

include("rbfLeastSquares.jl")
sigma = 1
lambda = 1
model = leastSquaresRBFL2(X, y, sigma, lambda)

# Report the error on the test set
using Printf
t = size(Xtest,1)
yhat = model.predict(Xtest)
testError = sum((yhat - ytest).^2)/t
@printf("TestError = %.2f\n",testError)

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
