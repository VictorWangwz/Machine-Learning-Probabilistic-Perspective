# Load X and y variable
using JLD, PyPlot
data = load("MNIST_images.jld")
(X,Xtest) = (data["X"],data["Xtest"])

m = size(X,1)
n = size(X,3)

#Train a inhomogeneous markov chain 
 p_ij = zeros(m, m, 2)
 for j in 1:m
     p_ij[1, j, 1] = sum(X[1, j, :] .== 1)/n
 end
 for i in 2: m
     for j in 1:m
         n1 = sum(X[i-1, j, :] .==1)
         p_ij[i, j, 1] = sum((X[i-1, j, :] .== 1) .& (X[i, j, :] .== 1))/n1
         p_ij[i, j, 2] = sum((X[i-1, j, :] .== 0) .& (X[i, j, :] .== 1))/(n-n1)
     end
 end

t = size(Xtest,3)
figure(1)
for image in 1:4
    subplot(2,2,image)

    # Grab a random test example
    ind = rand(1:t)
    I = Xtest[:,:,ind]

    # Fill in the bottom half using the model of Markov Chain
     for i in 2:m
         for j in 1:m
             if isnan(I[i,j])
		 if(i==1)
		     I[i, j] = rand() < p_ij[i,j, 1]
                 elseif(I[i-1, j]==1)
                     I[i, j] = rand() < p_ij[i,j, 1]
                 else
                     I[i, j] = rand() < p_ij[i,j, 2]
                 end
                
             end
         end
     end

    imshow(I)
    display(gcf())
end
