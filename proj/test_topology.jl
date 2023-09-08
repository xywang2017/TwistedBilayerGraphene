using PyPlot
using FFTW
using Printf
using DelimitedFiles
fpath = "/Users/xiaoyu/Code/TwistedBilayerGraphene"
include(joinpath(fpath,"libs/HamiltonianTopology_mod.jl"))

##
A = initHamiltonian(nsites=32,Δ=0.2,t=1.0);
##
fig, ax = subplots(2,2,figsize=(8,8))
to_plot = zeros(Float64,A.nsites,A.nsites,2,2)
to_plot[:,:,1,1] = abs.(A.H[1:2:(2A.nsites),1:2:(2A.nsites),1])
to_plot[:,:,1,2] = abs.(A.H[1:2:(2A.nsites),2:2:(2A.nsites),1])
to_plot[:,:,2,1] = abs.(A.H[2:2:(2A.nsites),1:2:(2A.nsites),1])
to_plot[:,:,2,2] = abs.(A.H[2:2:(2A.nsites),2:2:(2A.nsites),1])
v0 = 0
v1 = maximum(abs.(A.H)) 
titles = ["⟨+|H|+⟩" "⟨+|H|-⟩";"⟨-|H|+⟩" "⟨-|H|-⟩"]
for i in 1:2
    for  j in 1:2 
        ax[i,j].imshow(to_plot[:,:,i,j],origin="lower",vmin=v0,vmax=v1)
        ax[i,j].set_title(titles[i,j])
    end
end
display(fig)
close(fig)

## 
F = eigen(A.H)
spectrum = F.values
fig  = figure()
plot(spectrum,"b.",ms=1)
savefig("testH.pdf")
display(fig)
close(fig)

## 
u = F.vectors 
fig  = figure()
idx = A.nsites-1
vec = abs.(u[:,idx]).^2
plot(vec[1:2:size(A.H,1)],"r.")
plot(vec[2:2:size(A.H,1)],"g.")
display(fig)
close(fig)