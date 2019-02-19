using LinearAlgebra
include("misc.jl") # Includes mode function and GenericModel typedef

function gaussianMixture(X)
	(n,d) = size(X)

	K = 3
	phi_c = 1/K * ones(K,1)
	mu = 10*rand(K, d)
	Sigma = 2*rand(d,d,K) - rand(d,d,K)
	iter = 100
	for i in 1:K
		Sigma[:,:,i] = [zeros(d) I][:,2:end]
		while det(Sigma[:,:,i])<0.001
			Sigma[:,:,i] = 2*rand(d,d) - rand(d,d)
		end
	end

	function PDFc(Xhat, c)
		(t,d) = size(Xhat)
		PDFs = zeros(t)
		SigmaInv = Sigma[:,:,c]^-1
		logZ = (d/2)log(2pi) + (1/2)logdet(Sigma[:,:,c])  
		for i in 1:t
			xc = Xhat[i,:] - mu[c,:]
			loglik = -(1/2)dot(xc,SigmaInv*xc) - logZ
			PDFs[i] = exp(loglik)
		end
		return PDFs
	end

	function PDF(Xhat)
		(t,d) = size(Xhat)
		PDFs = zeros(t)
		logZ = zeros(K)
		SigmaInv = zeros(d,d,K)

		for c in 1:K 
			logZ[c] = (d/2)log(2pi) + (1/2)logdet(Sigma[:,:,c])
			SigmaInv[:,:,c] = Sigma[:,:,c]^-1
		end
		  
		for i in 1:t
			for c in 1:K
				xc = Xhat[i,:] - mu[c,:]
				loglik = -(1/2)dot(xc,SigmaInv[:,:,c]*xc) - logZ[c]
				PDFs[i] += phi_c[c]*exp(loglik)
			end
		end
		return PDFs
	end

	for t in 1:iter
		rc = zeros(n, K)
		for c in 1:K
			rc[:,c] = phi_c[c]*PDFc(X, c)./PDF(X)
			phi_c[c] = sum(rc[:, c])/n
		end
		for c in 1:K
			temp_mu = zeros(d)
			temp_sigma = zeros(d,d)
			for i in 1:n
				temp_mu += rc[i, c].*X[i, :]
				xc = X[i, :] - mu[c, :]
				temp_sigma += rc[i, c].*(xc*xc')
			end
			mu[c, :] = temp_mu./(n*phi_c[c])
			Sigma[:,:,c] = temp_sigma./(n*phi_c[c])
		end
	end

	return DensityModel(PDF)
end
