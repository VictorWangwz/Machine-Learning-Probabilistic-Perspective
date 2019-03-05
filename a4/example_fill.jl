# Load X and y variable
using JLD, PyPlot
data = load("MNIST_images.jld")
(X,Xtest) = (data["X"],data["Xtest"])

m = size(X,1)
n = size(X,3)

# Train an independent Bernoulli model
# p_ij = zeros(m,m)
# for i in 1:m
#     for j in 1:m
#         p_ij[i,j] = sum(X[i,j,:] .== 1)/n
#     end
# end

#Train a inhomogeneous markov chain 
# p_ij = zeros(m, m, 2)
# for j in 1:m
#     p_ij[1, j, 1] = sum(X[1, j, :] .== 1)/n
# end
# for i in 2: m
#     for j in 1:m
#         n1 = sum(X[i-1, j, :] .==1)
#         p_ij[i, j, 1] = sum((X[i-1, j, :] .== 1) .& (X[i, j, :] .== 1))/1
#         p_ij[i, j, 2] = sum((X[i-1, j, :] .== 0) .& (X[i, j, :] .== 1))/(n-n1)
#     end
# end


# Train a tabular DAG model
model = Array{Any}(undef, m, m)
include("tabular.jl")
for i in 1:m
    for j in 1: m
        h_l = max(1, j-2)
        w_l = max(1, i-2)
        h = j - h_l + 1
        w = i - w_l + 1
        d = h*w
        X_patch = zeros(n, d-1)
        y_patch = zeros(n, 1)
        k = 1
        for a in w_l: i
            for b in h_l: j
                if (a==i)&&(b==j)
                    y_patch[:] = X[i, j, :]
                    continue
                end
                X_patch[:, k] = X[i, j, :]
                k = k + 1 
            end
        end
        model[i, j] = tabular(X_patch, y_patch)
    end
end

# Show Bernoulli parameters
figure(1)
# imshow(p_ij[:,:,2])
# display(gcf())
# Fill-in some random test images
t = size(Xtest,3)
figure(2)
for image in 1:4
    subplot(2,2,image)

    # Grab a random test example
    ind = rand(1:t)
    I = Xtest[:,:,ind]
    # Fill in the bottom half using the model of Bernoulli
    # for i in 1:m
    #     for j in 1:m
    #         if isnan(I[i,j])
    #             I[i,j] = rand() < p_ij[i,j]
    #         end
    #     end
    # end

    # Fill in the bottom half using the model of Markov Chain
    # for i in 2:m
    #     for j in 1:m
    #         if isnan(I[i,j])
    #             if(I[i-1, j]==1)
    #                 I[i,j] = rand() < p_ij[i,j, 1]
    #             else
    #                 I[i,j] = rand() < p_ij[i,j, 2]
    #             end
                
    #         end
    #     end
    # end

    # Fill in the bottom half using the model of tabular
    for i in 1:m
        for j in 1:m
            if isnan(I[i,j])
                h_l = max(1, j-2)
                w_l = max(1, i-2)
                h = j - h_l + 1
                w = i - w_l + 1
                d = h*w
                X_patch = zeros(n, d-1)
                k = 1
                for a in w_l: i
                    for b in h_l: j
                        if (a==i)&&(b==j)
                            continue
                        end
                        X_patch[:, k] = X[i, j, :]
                        k = k + 1 
                    end
                end
                I[i,j] = model[i, j].sample(X_patch)
            end
        end
    end
    imshow(I)
    display(gcf())
end
