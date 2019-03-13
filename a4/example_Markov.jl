# Load X and y variable
using JLD, PyPlot
data = load("gradChain.jld")
(p1,pt) = (data["p1"],data["pt"])
k = length(p1)
include("sampleAncestral.jl")
prob = sampleAncestral(p1, pt, 1000, 500)
n = size(p1)[1]
for i in 1:n
    plot(range(1, stop=500, length=500), prob[i,:])
end
display(gcf())
# print(sampleAncestral(p1, pt, 50, 1000))
# print("\n")
# include("marginalCK.jl")
# print(marginalCK(p1, pt, 50))