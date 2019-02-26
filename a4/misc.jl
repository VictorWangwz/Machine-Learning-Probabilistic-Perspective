
# Define a "model" type, that just needs a predict function
mutable struct GenericModel
	predict # Function that makes predictions
end

mutable struct LinearModel
	predict # Funcntion that makes predictions
	w # Weight vector
end

mutable struct DensityModel
	pdf # Function that gives PDF
end

# Function to compute the mode of a vector
function mode(x)
	# Returns mode of x
	# if there are multiple modes, returns the smallest
	x = sort(x[:]);

	commonVal = [];
	commonFreq = 0;
	x_prev = NaN;
	freq = 0;
	for i in 1:length(x)
		if(x[i] == x_prev)
			freq += 1;
		else
			freq = 1;
		end
		if(freq > commonFreq)
			commonFreq = freq;
			commonVal = x[i];
		end
		x_prev = x[i];
	end
	return commonVal
end


# Return squared Euclidean distance all pairs of rows in X1 and X2
function distancesSquared(X1,X2)
	(n,d) = size(X1)
	(t,d2) = size(X2)
	@assert(d==d2)
	return X1.^2*ones(d,t) + ones(n,d)*(X2').^2 - 2X1*X2'
end

### A function to compute the gradient numerically
function numGrad(func,x)
	n = length(x);
	delta = 2*sqrt(1e-12)*(1+norm(x));
	g = zeros(n);
	e_i = zeros(n)
	for i = 1:n
		e_i[i] = 1;
		(fxp,) = func(x + delta*e_i)
		(fxm,) = func(x - delta*e_i)
		g[i] = (fxp - fxm)/2delta;
		e_i[i] = 0
	end
	return g
end

### Check if number is a real-finite number
function isfinitereal(x)
	return (imag(x) == 0) & (!isnan(x)) & (!isinf(x))
end

### For vector p given discrete probabilities, generates a random sample
function sampleDiscrete(p)
	minimum(find(cumsum(p[:]).> rand()))
end
