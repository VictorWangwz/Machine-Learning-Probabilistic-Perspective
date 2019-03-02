# Load X and y variable
using JLD
data = load("gradChain.jld")
(p1,pt) = (data["p1"],data["pt"])
k = length(p1)
