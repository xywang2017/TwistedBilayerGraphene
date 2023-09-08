using PyPlot
using JLD
fpath = pwd()
include(joinpath(fpath,"libs/HofstadterLL_modv3.jl"))

##
params = Params(dθ=1.38π/180,ϵ=1e-3,φ=0π/180)  #chiral limit
initParamsWithStrainv2(params)
# qs = [collect(2:20);25;30;35]
qs = collect(2:20)
p = 1
data = Dict()
σz = []
for iq in eachindex(qs)
    println("q=$(qs[iq])")
    q = qs[iq]
    lk = 12
    if (q<4)
        lk = 12
    elseif (q>10)
        lk = 4
    end
    hof = constructHofstadterLL(params,q=q,p=p,nLL=max(50,6q),lk=lk)
    data["$iq"] =  hof.spectrum[:]
    push!(σz,hof.σz)
end
save("Angle1.38Strain/LL_results_strain1e-3test.jld","data",data,"σz",σz)

##
fig = figure(figsize=(4,3))

d=load("Angle1.38Strain/LL_results_nostrain.jld")
data = d["data"]
σz = d["σz"]
qs = [collect(2:20);25;30;35]
for iq in eachindex(qs)
    p = 1
    q = qs[iq]
    ϵ = data["$iq"] 
    ϕ = ones(size(ϵ)) * p/q
    plot(ϕ,ϵ,"g.",ms=1,markeredgecolor="none")
end

d=load("Angle1.38Strain/LL_results_strain1e-3test.jld")
data = d["data"]
σz = d["σz"]
# qs =  [collect(2:20);25;30;35]
qs = collect(2:20)
for iq in eachindex(qs)
    p = 1
    q = qs[iq]
    ϵ = data["$iq"] 
    ϕ = ones(size(ϵ)) * p/q
    plot(ϕ,ϵ,"r.",ms=1,markeredgecolor="none")
end

# d=load("Angle1.38Strain/HybridW_Q32results_strain.jld")
# data = d["data"]
# σz = d["σz"]
# ps = collect(1:10)
# q = 32
# for ip in eachindex(ps)
#     p = ps[ip]
#     ϵ = data["$ip"] *1.5
#     ϕ = ones(size(ϵ)) * p/q
#     plot(ϕ,ϵ,"b.",ms=2,markeredgecolor="none")
# end

ylabel(L"ϵ/ϵ_0")
xlabel(L"ϕ/ϕ_0")
ylim([-0.4,0.4])
tight_layout()
display(fig)
savefig("Angle1.38Strain/Hofstadter LL.png",dpi=330)
close(fig)

##
# here I show the sublattice polarization as a function of ϕ/ϕ0
fig = figure(figsize=(4,3))
d=load("Angle1.38Strain/LL_results.jld")
data = d["data"]
σz = d["σz"]
ϕ = p./qs
nσz = [sum(σz[i])/length(σz[i]) for i in eachindex(σz)]
plot(ϕ,nσz,"go",markerfacecolor="none",ms=3)

xlabel(L"ϕ/ϕ_0")
ylabel(L"σ_z")
ylim([0,2.2])
tight_layout()
display(fig)
close(fig)

##
params = Params(dθ=1.38π/180,ϵ=0e-3);
initParamsWithStrain(params)
hof = constructHofstadterLL(params,q=4,p=1,nLL=40,lk=64);

kmesh = ( reshape(hof.k1,:,1)*params.g1 .+ reshape(hof.k2,1,:)*params.g2 ) / abs(params.g2);
kx = real(kmesh);
ky = imag(kmesh);

##
fig,ax = subplots(2,2,figsize=(8,4))
cnt = -1
for r in 1:2 
    for c in 1:2
        pl = ax[r,c].contourf(kx,ky,hof.spectrum[hof.nH+cnt,:,:],cmap="Blues_r")
        colorbar(pl,ax=ax[r,c])
        ax[r,c].axis("equal")
        cnt = cnt + 1
    end
end
tight_layout()
savefig("Angle1.38Strain/lowest_LLsq4_unstrained.pdf")
display(fig)
close(fig)

## 
# cut 
fig = figure(figsize=(4,3)) 
maxLL = 4
idx = 1
ϵcut = hof.spectrum[(hof.nH-maxLL):(hof.nH+maxLL+1),1:(hof.lk÷2+1),idx]
ϵcut = [ϵcut  hof.spectrum[(hof.nH-maxLL):(hof.nH+maxLL+1),hof.lk÷2+1,(idx+1):end]]

for n in 1:size(ϵcut,1)
    plot(ϵcut[n,:],"-",lw=0.5)
end
tight_layout()
# savefig("Angle1.38/lowest_LLs_cutq3.pdf")
display(fig)
close(fig)
