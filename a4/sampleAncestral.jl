function calculateCDF(p, d)
    for i in 2: d
        p[i] = p[i-1] + p[i]
    end
    return p
end

function sampleAncestral(p0, pt, t, d)
    n = size(pt, 1) ;
    mcs = ones(Int64, t, d)
    prob = ones(Float64, n, d)
    for i in 1:t
        p0_new = p0
        for j in 1:d
            cdf = calculateCDF(p0_new, n)
            r = rand()
            index = findfirst(x->(x>=r),cdf)
            mcs[i, j] = index
            p0_new = pt[index, :]
        end
    end
    for i in 1:d
        for j in 1:n
            prob[j, i] = size(findall(x -> x==j, mcs[:, i]))[1]/t
        end
    end
    return prob
end