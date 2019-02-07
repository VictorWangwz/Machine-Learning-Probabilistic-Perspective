include("misc.jl") # Includes mode function and GenericModel typedef

function knn_predict(Xhat,X,y,k)
  (n,d) = size(X)
  (t,d) = size(Xhat)
  k = min(k,n)

  # Let's just pre-compute all the squared distances
  # (increases memory to n^2, which isn't necessary)
  D = distancesSquared(X,Xhat)

  yhat = zeros(t)
  for i in 1:t
    # Sort the distances to the other points
    nearest = sortperm(D[:,i])

    # Use mode of the labels among the neighbours
    yhat[i] = mode(y[nearest[1:k]])
  end

  return yhat
end

function knn(X,y,k)
	# Implementation of k-nearest neighbour classifier
  predict(Xhat) = knn_predict(Xhat,X,y,k)
  return GenericModel(predict)
end
