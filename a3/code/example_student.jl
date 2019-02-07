using LinearAlgebra, PyPlot

# Generate data from a Gaussian with outlier
n = 250
d = 2
nOutliers = 25
mu = randn(d)
Sigma = randn(d,d)
Sigma = (1/2)*(Sigma+Sigma') # Make symmetric
sd = eigen(Sigma)
Sigma += (1-minimum(sd.values))*I # Make positive-definite
R = cholesky(Sigma) # Get a matrix acting like the "standard deviation": Sigma = A*A'
A = R.L
X = zeros(n,d)
for i in 1:n
    xi = randn(d) # Sample from multivariate standard normal
    X[i,:] = A*xi + mu # Sample from multivariate Gausian (by affine property)
end
X[rand(1:n,nOutliers),:] = abs.(10*rand(nOutliers,d)) # Add some crazy points

include("studentT.jl")
model = studentT(X)

# Plot data and densities (you can ignore the code below)
plot(X[:,1],X[:,2],".")

increment = 100
(xmin,xmax) = xlim()
xDomain = range(xmin,stop=xmax,length=increment)
(ymin,ymax) = ylim()
yDomain = range(ymin,stop=ymax,length=increment)

xValues = repeat(xDomain,1,length(xDomain))
yValues = repeat(yDomain',length(yDomain),1)

z = model.pdf([xValues[:] yValues[:]])

@assert(length(z) == length(xValues),"Size of model function's output is wrong");

zValues = reshape(z,size(xValues))

contour(xValues,yValues,zValues)
