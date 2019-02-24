using LinearAlgebra
include("misc.jl") # Includes mode function and GenericModel typedef

function gaussianMixture(X)
    (n,d) = size(X)
    k = 3
    p_c = 1/k * ones(k,1)
    iter = 100
    mu = 10*rand(k, d)
    Sigma = rand(k, d, d)
    SigmaInv = zeros(k, d, d)
    for i in 1:k
		Sigma[i,:,:] = [zeros(d) I][:,2:end]
		while det(Sigma[i,:,:])<0.001
			Sigma[i,:,:] = 2*rand(d,d) - rand(d,d)
		end
	end
    
	function PDFc(Xhat, c)
		(t,d) = size(Xhat)
		PDFs = zeros(t)
        SigmaInv = Sigma[c, :, :]^-1
        logZ = (d/2)log(2pi) + (1/2)logdet(Sigma[c, :,:])  
		for i in 1:t
			xc = Xhat[i,:] - mu[c, :]
			loglik = -(1/2)dot(xc,SigmaInv*xc) - logZ
			PDFs[i] = exp(loglik)
		end
		return PDFs
    end
    
    function PDF(Xhat)
        (t,d) = size(Xhat)
        PDFs = zeros(t)
        logZ = zeros(k)
        SigmaInv = zeros(k, d, d)
        for c in 1: k
            logZ[c] = (d/2)log(2pi) + (1/2)logdet(Sigma[c, :,:]) 
            SigmaInv[c,:,:] = Sigma[c,:,:]^-1 
        end 
		
        for i in 1:t
            for c in 1:k
                xc = Xhat[i,:] - mu[c, :]
                loglik = -(1/2)dot(xc,SigmaInv[c,:,:]*xc) - logZ[c]
                PDFs[i] += p_c[c]*exp(loglik)
            end 
			
		end
		return PDFs

    end

    for i in 1: iter
        rc = zeros(n, k)
        for c in 1: k
            rc[:, c] = p_c[c]*PDFc(X, c)./PDF(X)
            p_c[c] = sum(rc[:,c])/n
        end
        for c in 1:k
            temp_mu = zeros(d)
            temp_sigma = zeros(d, d)
            for i in 1:n
                temp_mu += rc[i,c]*X[i, :]
                temp_sigma += rc[i, c]*(X[i,:]-mu[c,:])*(X[i,:]-mu[c,:])'
            end
            mu[c,:] = temp_mu ./ (n*p_c[c])
            Sigma[c, :, :] = temp_sigma ./(n *p_c[c])
        end
    end

	return DensityModel(PDF)
end