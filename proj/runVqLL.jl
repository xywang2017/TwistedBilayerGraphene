using PyPlot
using JLD
using DelimitedFiles
fpath = pwd()
include(joinpath(fpath,"libs/HofstadterLL_modv3.jl"))
include(joinpath(fpath,"libs/HofstadterVqLL_fullgrid.jl"))

##
w1=96.056
α=0.71
params = Params(dθ=1.05π/180,w0=α*w1,w1=w1,ϵ=0.0)
ϕ = 1//1
q = denominator(ϕ)
p = numerator(ϕ)

nLL = 20
hof = constructHofstadterLL(params,q=q,p=p,nLL=nLL,lk=10);
@time hof = constructHofstadterVqLL(params,q=q,p=p,nLL=nLL,lk=10);

save("2piFlux/fullfluxVq.jld","hof",hof)

## 
data = load("2piFlux/fullfluxVq.jld","hof")
kx = real(data.kvec) / abs(params.g1)
ky = imag(data.kvec) / abs(params.g1)
ϵ_hl = data.spectrum[:,:,:,1]
ϵ_cnp = data.spectrum[:,:,:,2]
ϵ_el = data.spectrum[:,:,:,3]
# V0 = 1/abs(params.a1)

ϵ = ϵ_cnp
valmin = minimum(ϵ)
valmax = maximum(ϵ)
fig,ax=subplots(2,1,figsize=(4,6))
pl = ax[1].pcolormesh(kx,ky,ϵ[1,:,:],vmin=valmin,vmax=valmax,cmap="Spectral_r")
ax[1].plot([0],[1/sqrt(3)],"r+")
ax[1].axis("equal")
colorbar(pl,ax=ax[1])
pl = ax[2].pcolormesh(kx,ky,ϵ[2,:,:],vmin=valmin,vmax=valmax,cmap="Spectral_r")
ax[2].plot([0],[1/sqrt(3)],"r+")
ax[2].axis("equal")
colorbar(pl,ax=ax[2])
tight_layout()
display(fig)
close(fig)

# ycut for x = 0 
kcut = [imag(data.kvec[i,2(i-1)+1]) for i in 1:(size(data.kvec,1)÷2)]
e1cut = [ϵ[1,i,2(i-1)+1] for i in 1:(size(data.kvec,1)÷2)]
e2cut = [ϵ[2,i,2(i-1)+1] for i in 1:(size(data.kvec,1)÷2)]
fig = figure()
plot(kcut,e1cut,"r.")
plot(kcut,e2cut,"b.")
# ylim([0,2])
tight_layout()
display(fig)
close(fig)
##

ϵBM = hof.spectrum[(hof.nH+1-hof.q):(hof.nH+hof.q),:,:]
valmin = minimum(ϵBM)
valmax = maximum(ϵBM)
kvec = reshape(hof.k1,:,1)* params.g1 .+ reshape(hof.k2,1,:)*params.g2
kx = real(kvec) / abs(params.g1)
ky = imag(kvec) / abs(params.g1)
fig,ax=subplots(2,1,figsize=(4,6))
pl = ax[1].pcolormesh(kx,ky,ϵBM[1,:,:],cmap="Spectral_r")
ax[1].plot([0],[1/sqrt(3)],"r+")
ax[1].axis("equal")
colorbar(pl,ax=ax[1])
pl = ax[2].pcolormesh(kx,ky,ϵBM[2,:,:],cmap="Spectral_r")
ax[2].plot([0],[1/sqrt(3)],"r+")
ax[2].axis("equal")
colorbar(pl,ax=ax[2])
tight_layout()
display(fig)
close(fig)

