# Load X and y variable
using JLD, PyPlot
data = load("MNIST_images.jld")
(X,Xtest) = (data["X"],data["Xtest"])

m = size(X,1)
n = size(X,3)

# Train an independent Bernoulli model
p_ij = zeros(m,m)
for i in 1:m
    for j in 1:m
        p_ij[i,j] = sum(X[i,j,:] .== 1)/n
    end
end

# Show Bernoulli parameters
figure(1)
imshow(p_ij)

# Fill-in some random test images
t = size(Xtest,3)
figure(2)
for image in 1:4
    subplot(2,2,image)

    # Grab a random test example
    ind = rand(1:t)
    I = Xtest[:,:,ind]

    # Fill in the bottom half using the model
    for i in 1:m
        for j in 1:m
            if isnan(I[i,j])
                I[i,j] = rand() < p_ij[i,j]
            end
        end
    end
    imshow(I)
end
