include("marginalCK.jl")
function calculateCDF(p, d)
    for i in 2: d
        p[i] = p[i-1] + p[i]
    end
    return p
end

function sampleBackwards(p1, pT, t, xd, d)
	mcs = ones(Int64, t, d)
	M = marginalCK(p1, pT, d)
	for i in 1:t
		mcs[i, d] = xd
		for j in d-1:-1:1
			pT_b = pT[:,mcs[i, j+1]].*M[:, j]
			pT_b = pT_b ./ sum(pT_b)
			cdf = calculateCDF(pT_b, size(pT_b, 1))
			r = rand()
			index = findfirst(x->(x>=r),cdf)
			mcs[i, j] = index
		end
	end
	return mcs
end
