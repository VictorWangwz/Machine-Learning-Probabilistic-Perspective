# Load X and y variable
using JLD
data = load("gradChain.jld")
(p1,pt) = (data["p1"],data["pt"])
k = length(p1)
include("sampleAncestral.jl")
print(sampleAncestral(p1, pt, 50))
