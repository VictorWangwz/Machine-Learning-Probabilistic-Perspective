function marginalCK(p1, pt, d)
    t = size(p1,1)
    M = ones(t, d)
    M[:, 1] = p1
    for i in 2:d
        p1 = (p1' * pt)'
        M[:,i] = p1
    end
    return M

end