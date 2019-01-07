# Load X and y variable
using JLD
data = load("multiData.jld")
(X,y,Xtest,ytest) = (data["X"],data["y"],data["Xtest"],data["ytest"])

# Data is already roughly standardized, but let's add bias
n = size(X,1)
X = [ones(n,1) X]

# Do the same transformation to the test data
t = size(Xtest,1)
Xtest = [ones(t,1) Xtest]

# Fit one-vs-all logistic regression model
include("logReg.jl")
model = logRegOnevsAll(X,y)

# Compute training and validation error
using Statistics
yhat = model.predict(X)
trainError = mean(yhat .!= y)
@show(trainError)
yhat = model.predict(Xtest)
validError = mean(yhat .!= ytest)
@show(validError)

# Plot results
k = maximum(y)
include("plot2Dclassifier.jl")
plot2Dclassifier(X,y,model,Xtest=Xtest,ytest=ytest,biasIncluded=true,k=5)
