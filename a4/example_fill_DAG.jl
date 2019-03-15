# Load X and y variable
using JLD, PyPlot
data = load("MNIST_images.jld")
(X,Xtest) = (data["X"],data["Xtest"])

m = size(X,1)
n = size(X,3)

model = Array{Any}(undef, m, m)
include("tabular.jl")
for i in 1:m
    for j in 1: m
        i1 = max(1, i-2)
        j1 = max(1, j-2)
	d = (i-i1+1)*(j-j1+1)
	X_tab = zeros(n, d)
	k = 1
	for ii in i1:i
		for jj in j1:j
			X_tab[:, k] = X[ii, jj, :]
			k += 1
		end
	end
        model[i, j] = tabular(X_tab[:, 1:d-1], X_tab[:, d])
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
                h_l = max(1, j-2)
                w_l = max(1, i-2)
                h = j - h_l + 1
                w = i - w_l + 1
                d = h*w
                I_patch = zeros(d-1)
                k = 1
                for a in w_l: i
                    for b in h_l: j
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
