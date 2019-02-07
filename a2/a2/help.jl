function getfNew(X, wNew, alpha, g_j, j, r, lambda)
	Xj = X[:,j]
	Xj = reshape(Xj, length(Xj), 1)
	rNew = r - alpha*g_j*Xj
	return ((1/2)norm(rNew)^2 + (lambda/2)norm(wNew)^2, rNew)
end

function getNewGj(X, wNew, alpha, g_j, j, r, lambda)
	Xj = X[:,j]
	Xj = reshape(Xj, length(Xj), 1)
	rNew = r - alpha*g_j*Xj
	Xj = X'[j,:]
    	g_j = (reshape(Xj, 1, length(Xj))*rNew)[1,1] + lambda*wNew[j]
	return g_j
end
