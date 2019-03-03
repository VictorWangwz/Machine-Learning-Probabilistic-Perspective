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
    return X
end
    # for i in 1: t
    #     if rand() > p0[1]
    #         X[i, 1] = 2
    #     end
    #     for j in 2:d
    #         if rand() > pt[X[i, j-1], j-1]
    #             X[i, j] = 2
    #         end 
    #     end

    # end