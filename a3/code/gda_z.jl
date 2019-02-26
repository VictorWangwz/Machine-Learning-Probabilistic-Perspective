using LinearAlgebra
include("misc.jl") # Includes mode function and GenericModel typedef
function gda_predict(Xhat, theta, mu, sigma)
    (n, d) = size(Xhat)
    Yhat = zeros(n, 1)
    k = size(mu,1)

    for i in 1:n
        log_p = zeros(k)
        for c in 1: k
            temp = Xhat[i, :] - mu[c, :]
            log_p[c] = -1/2 *temp' *inv(sigma[c ,:, :])*temp - 1/2*log(det(sigma[c,:,:])) + log(theta[c])
        end
        (~, Yhat[i]) = findmax(log_p)
    end

    return Yhat

end

function gda(X, y)
    (n, d) = size(X)
    k = maximum(y)
    
    n_c = zeros(k)
    mu = zeros(k,d)
    theta  = zeros(k)
    sigma = zeros(k, d, d)

    # mu
    for i in 1: n
        mu[y[i], :] += X[i, :]
        n_c[y[i]] += 1
    end
    for c in 1:k
        mu[c, :] = mu[c, :] ./ n_c[c]
        theta[c] = n_c[c]/n
    end 

    # sigma
    for i in 1: n
        sigma[y[i], :, :] += (X[i,:]-mu[y[i],:])*(X[i,:]-mu[y[i],:])'
    end
    for c in 1: k
        sigma[c,:,:]  = sigma[c,:,:] ./n_c[c]
    end
    predict(Xhat) = gda_predict(Xhat,theta, mu, sigma)
	return GenericModel(predict)



end