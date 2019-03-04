function marginalCK(p1, pt, t)
    d = size(p1,1)
    M = ones(d, t)
    M[:, 1] = p1
    for i in 2:t

        M[:,i] = pt' * M[:, i-1]
    end
    return M

end