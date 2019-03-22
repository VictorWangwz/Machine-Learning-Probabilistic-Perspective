include("misc.jl") # Includes iterator over states

function suffStat(X,E,k)
    (n,d) = size(X)

    nodeStats = zeros(d,k)
    for i in 1:n
        for j in 1:d
            nodeStats[j,X[i,j]] += 1
        end
    end
    nodeStats /= n

    nEdges = size(E,1)
    edgeStats = zeros(k,k,nEdges)
    for i in 1:n
        for e in 1:nEdges
            j1 = E[e,1]
            j2 = E[e,2]
            edgeStats[X[i,j1],X[i,j2],e] += 1
        end
    end
    edgeStats /= n

    return (nodeStats,edgeStats)
end

function unnormalizedProb(x,w,v,E)
    (d,k) = size(w)

    pTilde = 1
    for j in 1:d
        pTilde *= exp(w[j,x[j]])
    end

    # Loops like this one would look nicer if I could do:
    # for (j1,j2) in E (and somehow get the edge number)
    for e in 1:nEdges
        j1 = E[e,1]
        j2 = E[e,2]
        pTilde *= exp(v[x[j1],x[j2],e])
    end

    return pTilde
end

function UGM_Decode(w,v,E)

    (d,k) = size(w)

    xDecode = zeros(d)
    decodeProb = 0
    for x in DiscreteStates(d,k)
        pTilde = unnormalizedProb(x,w,v,E)

        if pTilde > decodeProb
            decodeProb = pTilde
            xDecode = x
        end
    end
    return xDecode
end


function UGM_Infer(w,v,E)

    (d,k) = size(w)
    nEdges = size(E,1)

    # Compute normalizing constant and marginals
    Z = 0
    nodeMarg = zeros(d,k)
    edgeMarg = zeros(k,k,nEdges)
    for x in DiscreteStates(d,k)

        pTilde = unnormalizedProb(x,w,v,E)
        Z = Z + pTilde

        for j in 1:d
            nodeMarg[j,x[j]] += pTilde
        end
        for e in 1:nEdges
            j1 = E[e,1]
            j2 = E[e,2]
            edgeMarg[x[j1],x[j2],e] += pTilde
        end
    end
    nodeMarg ./= Z
    edgeMarg ./= Z

    return (Z,nodeMarg,edgeMarg)
end

function UGM_Sample(w,v,E,nSamples)

    (d,k) = size(w)

    (Z,~,~) = UGM_Infer(w,v,E)

    samples = zeros(nSamples,d)
    for s in 1:nSamples
        u = rand()
        CDF = 0
        for x in DiscreteStates(d,k)
            CDF += unnormalizedProb(x,w,v,E)

            if CDF/Z > u
                samples[s,:] = x
                break
            end
        end
    end
    return samples
end


function UGM_NLL(wv,E,nodeStats,edgeStats)

    (d,k) = size(nodeStats)
    nEdges = size(E,1)

    w = reshape(wv[1:d*k],d,k)
    v = reshape(wv[d*k+1:end],k,k,nEdges)

    # Compute normalizing constant and marginals
    (Z,nodeMarg,edgeMarg) = UGM_Infer(w,v,E)

    # Compute NLL
    NLL = - dot(w[:],nodeStats[:]) - dot(v[:],edgeStats[:]) + log(Z)

    # Compute gradient
    nodeGrad = zeros(d,k)
    for j in 1:d
        for s in 1:k
            nodeGrad[j,s] = -nodeStats[j,s] + nodeMarg[j,s]
        end
    end
    edgeGrad = zeros(k,k,nEdges)
    for e in 1:nEdges
        j1 = E[e,1]
        j2 = E[e,2]
        for s1 in 1:k
            for s2 in 1:k
                edgeGrad[s1,s2,e] = -edgeStats[s1,s2,e] + edgeMarg[s1,s2,e]
            end
        end
    end
    return (NLL,[nodeGrad[:];edgeGrad[:]])
end


function UGM_Infer_Cond(w,v,E,xc)

    (d,k) = size(w)
    nEdges = size(E,1)

    # This is NOT an efficient way to do conditional inference
    # (the conditional inference should be faster than unconditional)

    # Compute normalizing constant and marginals
    Z = 0
    nodeMarg = zeros(d,k)
    edgeMarg = zeros(k,k,nEdges)
    for x in DiscreteStates(d,k)

        satisfied = true
        for j in 1:d
            if (xc[j] != 0) & (xc[j] != x[j])
                # Example not satisfying conditioning
                satisfied = false
            end
        end

        if satisfied
            pTilde = unnormalizedProb(x,w,v,E)
            Z = Z + pTilde

            for j in 1:d
                nodeMarg[j,x[j]] += pTilde
            end
            for e in 1:nEdges
                j1 = E[e,1]
                j2 = E[e,2]
                edgeMarg[x[j1],x[j2],e] += pTilde
            end
        end
    end
    nodeMarg ./= Z
    edgeMarg ./= Z

    return (Z,nodeMarg,edgeMarg)
end
