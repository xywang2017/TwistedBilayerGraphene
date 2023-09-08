using PyPlot
using Printf
using DelimitedFiles
using JLD
fpath = pwd()
include(joinpath(fpath,"libs/HofstadterLL_modv2.jl"))

##
params = Params(dθ=1.38π/180)  #chiral limit
# ϕs = [2//7; 2//5; 3//7;4//9; 1//2; 6//13;5//13;7//15;8//17]
ϕs = [1//2 .- 1 .// collect(3:18); 1//2]
data = Dict()
for iϕ in eachindex(ϕs)
    println("ϕ=$(ϕs[iϕ])")
    p = numerator(ϕs[iϕ])
    q = denominator(ϕs[iϕ])
    hof = constructHofstadterLL(params,q=q,p=p,nLL=max(50,6q),lk=1)
    data["$iϕ"] =  hof.spectrum[:]
end
save("Angle1.38/LL_results_psequence.jld","data",data)

##
data = load("Angle1.38/LL_results_psequence.jld","data")
idx = 14
fig = figure(figsize=(4,3))
ϵ = reshape(data["$(idx)"],:,1)
q = denominator(ϕs[idx])
mid_index = length(ϵ) ÷ 2
ϵflat = ϵ[:,1]
plot(ϵflat[(mid_index-q+1):(mid_index+q)],"r.",label="7/16")
legend()
tight_layout()
display(fig)
savefig("psequences_7_16.pdf")
close(fig)
##
fig = figure(figsize=(6,4))

data = load("Angle1.38/LL_results_psequence.jld","data")
for iϕ in eachindex(ϕs)
    p = numerator(ϕs[iϕ])
    q = denominator(ϕs[iϕ])
    ϵ = data["$iϕ"]
    mid_index = length(ϵ) ÷ 2
    ϵflat = ϵ[(mid_index-q+1):(mid_index+q)]
    ϕ = ones(size(ϵflat)) * p/q
    plot(ϕ,ϵflat,"b.",ms=3,markeredgecolor="none")
end

ylabel(L"ϵ/ϵ_0")
xlabel(L"ϕ/ϕ_0")
ylim([-0.15,0.15])
tight_layout()
display(fig)
savefig("psequences.pdf")
close(fig)


## 
q = 15
p = 6
params = Params(dθ=1.38π/180,w0=0.0)  #chiral limit
hof = constructHofstadterLL(params,q=q,p=p,nLL=max(50,6q),lk=4)

##
