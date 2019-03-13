# Load X and y variable
using JLD, PyPlot
data = load("gradChain.jld")
(p1,pt) = (data["p1"],data["pt"])
k = length(p1)
time = 500
num_sample = 1000
include("sampleAncestral.jl")
# prob = sampleAncestral(p1, pt, num_sample, time)
n = size(p1)[1]

print("\n")
include("marginalCK.jl")
# prob = marginalCK(p1, pt, time)

include("viterbiDecoding.jl")
prob, index = viterbiDecoding(p1, pt, time)
for i in 1:n
    plot(range(1, stop=time, length=time), prob[i,:])
end
display(gcf())