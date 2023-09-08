using PyPlot
using JLD
fpath = pwd()
include(joinpath(fpath,"libs/HofstadterLL_modv3.jl"))

# This code tests out the Hofstadter spectra, using LL states at 1/q as variational ansatz; specifically we choose q=2
##
params = Params(dθ=1.38π/180)  #chiral limit
q = 2
p = 1
δϕ = collect(0.4:0.02:0.6) .- 0.5
factors = δϕ ./ 0.5
data = Dict()
σz = []
for iq in eachindex(factors)
    # println("factor=$(factors[iq])")
    lk = 32
    hof = constructHofstadterLL(params,q=q,p=p,nLL=max(50,10q),lk=lk,factor = factors[iq])
    data["$iq"] =  hof.spectrum[:]
    push!(σz,hof.σz)
end
save("Angle1.38test/LL_results.jld","data",data,"σz",σz)

##

fig = figure(figsize=(4,3))

ϕs = [2//7; 2//5; 3//7;4//9; 1//2; 6//13;5//13;7//15;8//17]
data = load("Angle1.38/LL_results_psequence.jld","data")
for iϕ in eachindex(ϕs)
    p = numerator(ϕs[iϕ])
    q = denominator(ϕs[iϕ])
    ϵ = data["$iϕ"]
    ϕ = ones(size(ϵ)) * p/q
    plot(ϕ,ϵ,"b.",ms=1,markeredgecolor="none")
end

d=load("Angle1.38/LL_results.jld")
data = d["data"]
σz = d["σz"]
qs = collect(2:20)
for iq in eachindex(qs)
    p = 1
    q = qs[iq]
    ϵ = data["$iq"] 
    ϕ = ones(size(ϵ)) * p/q
    plot(ϕ,ϵ,"r.",ms=1,markeredgecolor="none")
end

d=load("Angle1.38test/LL_results.jld")
data = d["data"]
σz = d["σz"]
q= 2 
p = 1
for iq in eachindex(factors)
    ϵ = data["$iq"] 
    ϕ = ones(size(ϵ)) * (factors[iq] * 0.5 + 0.5)
    plot(ϕ,ϵ,"g.",ms=1,markeredgecolor="none")
end

ylabel(L"ϵ/ϵ_0")
xlabel(L"ϕ/ϕ_0")
ylim([-0.2,0.2])
tight_layout()
display(fig)
close(fig)
