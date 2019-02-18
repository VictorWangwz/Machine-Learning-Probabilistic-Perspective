using LinearAlgebra

function GDA(X, Y)

	(n, d) = size(X)
	k = maximum(Y)

	theta_c = zeros(k)	#pi_c in class
	n_c = zeros(k)
	mu_c = zeros(k, d)
	sigma_c = zeros(k, d, d)


	for i in 1:n
		mu_c[y[i], :] += X[i, :]
		n_c[y[i]] += 1
	end
	# mu_c, n_c
	for c in 1:k
		mu_c[c, :] ./= n_c[c]
		theta_c[c] = n_c[c]/n
	end

	for i in 1:n
		c = y[i]
		sigma_c[c, :, :] += (X[i, :] - mu_c[c, :])*(X[i, :] - mu_c[c, :])'
	end

	for c in 1:k
		sigma_c[c, :, :] ./= n_c[c]
	end

	predict(Xhat) = GDA_predict(Xhat,theta_c, mu_c, sigma_c)
	return GenericModel(predict)
end

function GDA_predict(Xhat, theta, mu, sigma)
	(n, d) = size(Xhat)
	Yhat = zeros(n,1)
	k = size(mu, 1)

	for i in 1:n
		log_p = zeros(k)
		for c in 1:k
			temp = Xhat[i, :] - mu[c, :]
			log_p[c] = log(theta[c]) - (1/2)*temp'*inv(sigma[c,:,:])*temp - (1/2)*log(det(sigma[c,:,:]))
		end
		(~, Yhat[i]) = findmax(log_p)
	end

	return Yhat
end
