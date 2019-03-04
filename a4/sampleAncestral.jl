function calculateCDF(CDF, p, d)
    for i in 2: d
        CDF[i] = CDF[i-1] + p[i]
    end
    return CDF
end

function sampleAncestral(p0, pt, t)
    d = size(pt, 1) ;
    X = ones(Int64, t, d)
    CDF = ones(d)
    prob = ones(Float64, d, d)
    for i in 1:t
        CDF[1] = p0[1]
        CDF = calculateCDF(CDF, p0, d)
        random_prob = rand()
        X[i, 1] = findfirst(x->(x>=random_prob), CDF)
        CDF[1] = pt[X[i, 1]]
        for j in 2:d
            CDF = calculateCDF(CDF, pt[X[i, 1],:], d)
            random_prob = rand()
            X[i, j] = findfirst(x->(x>=random_prob), CDF)
        end
    end
    prod2 = ones(Float64, d)
    
    for i in 1:d
        prod2[i] = float(size(findall(x->x==i, X))[1]/(t*d))
        for j in 1:d
            prob[i, j] = float(size(findall(x->x==j, X[:, i]))[1]/t)
        end
    end

    print(prod2)
    return prob
end