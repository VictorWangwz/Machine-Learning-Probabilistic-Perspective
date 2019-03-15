# Load X and y variable
using JLD, PyPlot
data = load("gradChain.jld")
(p1,pt) = (data["p1"],data["pt"])
k = length(p1)
include("sampleAncestral.jl")
prob, mcs = sampleAncestral(p1, pt, 1000000, 10);
n = size(pt, 1)
cnt = zeros(n+1)
for i in 1:1000000
	if mcs[i, 10]==6
		cnt[n+1] +=1
		cnt[mcs[i, 5]] += 1
	end
end
show(cnt./cnt[8])
#[1.0, 15.0, 28.0, 3.0, 2.0, 9.0, 0.0, 58.0]
