# Load X and y variable
using JLD, PyPlot
data = load("MNIST_images.jld")
(X,Xtest) = (data["X"],data["Xtest"])

m = size(X,1)
n = size(X,3)

model = Array{Any}(undef, m, m)
include("logReg.jl")
for i in 15:m
    for j in 1: m
        d = i*j
	X_tab = zeros(n, d)
	k = 1
	for ii in 1:i
		for jj in 1:j
			X_tab[:, k] = X[ii, jj, :]
			k += 1
		end
	end
        model[i, j] = logReg(X_tab[:, 1:d-1], X_tab[:, d])
    end
end

t = size(Xtest,3)
figure(2)
for image in 1:4
    subplot(2,2,image)

    # Grab a random test example
    ind = rand(1:t)
    I = Xtest[:,:,ind]


    for i in 1:m
        for j in 1:m
            if isnan(I[i,j])
                d = i*j
                I_patch = zeros(d-1)
                k = 1
                for a in 1: i
                    for b in 1: j
                        if (a==i)&&(b==j)
                            continue
                        end
                        I_patch[k] = I[a, b]
                        k = k + 1 
                    end
                end
                I[i,j] = model[i, j].sample(I_patch)
            end
        end
    end
    imshow(I)
    display(gcf())
end
