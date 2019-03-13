function viterbiDecoding(p1, pt, d)
    t = size(p1,1)
    M = ones(t, d)
    index = ones(t, d)
    for i in 1:d
        new_p = (p1 .* pt)
        p1, idx= findmax(new_p, dims=1)
        for j in 1: t
            index[j, i] = idx[j][1] 
        end
        
        M[:,i] = p1
    end
    return M, index

end