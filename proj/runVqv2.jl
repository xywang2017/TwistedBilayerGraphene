using PyPlot
using JLD
fpath = pwd()
include(joinpath(fpath,"libs/HofstadterVq_mod.jl"))

##
@time begin 
ϕ = 1//16
p = numerator(ϕ)
q = denominator(ϕ)
params = Params(dθ=1.05π/180,ϵ=0.0,w0=0.96056,w1=96.056);
hof = HofstadterVq()
# Hnq = Dict()
# for lk in [1,5,9,13,17]
# for lk in [9]
   blk, basis= initHofstadterVq(hof,params,p=p,q=q,lk=5);
    # Hnq["$(lk)"] = hof.H; 
# end
# save("tmpH_grid_comparison_flatCoulomb.jld","data",Hnq)
end

##
fig = figure(figsize=(4,3))
plot(eigvals(Hermitian(hof.O[:,:,1])),"g.")
tight_layout()
display(fig)
close(fig)



save("StrongCoupling/strongcoupling_q$(q)_p$(p)_chiralv2.jld","spectrum",hof.H,"sigmaz",hof.Σz)

##
Hnq = load("tmpH_grid_comparison_flatCoulomb.jld","data")
energies = Dict()
for lk in [1,5,9,13,17]
    H = Hnq["$(lk)"];
    vals = zeros(Float64,size(H,1),size(H,3))
    for ik in 1:size(H,3)
        vals[:,ik] = eigvals(Hermitian(H[:,:,ik]))
    end
    energies["$(lk)"] = vals
end

##
fig = figure(figsize=(5,4))
for lk in [1,5,9,13,17]
    ϵ= energies["$(lk)"][:]
    nq = (lk-1)÷4+1
    plot(ϵ,ones(length(ϵ)) .+ (lk-1)/20,".",markeredgecolor="none",label="nq=$(nq)")
end
yticks([])
xlabel("ϵ/ϵ0")
legend()
tight_layout()
savefig("finite_gridsize_effect_q4p1_flatCoulomb.pdf")
display(fig)
close(fig)
##
q = 16
p = 3
data = load("StrongCoupling/strongcoupling_q$(q)_p$(p)_chiralv2.jld")
H = data["spectrum"]
Pz = data["sigmaz"]
energies = zeros(Float64,size(H,2),size(H,3))
for i1 in 1:size(H,3)
    F = eigen(Hermitian(Pz[:,:,i1]))
    Hnew = F.vectors' * H[:,:,i1] * F.vectors
    # energies[1:(q+1),i1] = eigvals(Hermitian(Hnew[1:(q+1),1:(q+1)]))
    # energies[(q+2):(2q),i1] = eigvals(Hermitian(Hnew[(q+2):(2q),(q+2):(2q)]))
    energies[:,i1] = eigvals(Hermitian(Hnew))
end

##
Pz11 = Pz[:,:,4]
F = eigen(Hermitian(Pz11))
fig,ax = subplots(1,2,figsize=(8,3))
pl=ax[1].imshow(real(Pz11),origin="lower",extent=(1,2q,1,2q))
colorbar(pl,ax=ax[1])
H11 = F.vectors' * H[:,:,4] * F.vectors
pl=ax[2].imshow(log10.(abs.(H11)),origin="lower",extent=(1,2q,1,2q))
colorbar(pl,ax=ax[2])
tight_layout()
# savefig("StrongCoupling/strongcoupling_q10p1_chiral_H_Pz.pdf")
display(fig)
close(fig)

##
Pz11 = Pz[:,:,4]
fig = figure()
pl=imshow(real.(Pz11),origin="lower",extent=(1,2q,1,2q))
colorbar(pl)
display(fig)
close(fig)

##
fig = figure()
data = load("StrongCoupling/strongcoupling_q10_p1_chiralv2.jld")
H = data["spectrum"]
Pz = data["sigmaz"]
q = 10
energies = zeros(Float64,size(H,2),size(H,3))
for i1 in 1:size(H,3)
    F = eigen(Hermitian(Pz[:,:,i1]))
    Hnew = F.vectors' * H[:,:,i1] * F.vectors
    energies[1:(q+1),i1] = eigvals(Hermitian(Hnew[1:(q+1),1:(q+1)]))
    energies[(q+2):(2q),i1] = eigvals(Hermitian(Hnew[(q+2):(2q),(q+2):(2q)]))
end
ϵ1 = energies[1:11,:]
ϵ2 = energies[12:20,:]
plot(ones(length(ϵ1))/10,ϵ1[:],"bo",ms=1,markerfacecolor="none")
plot(ones(length(ϵ2))/10.1,ϵ2[:],"ro",ms=1,markerfacecolor="none")
data = load("StrongCoupling/strongcoupling_q16_p1_chiralv2.jld")
H = data["spectrum"]
Pz = data["sigmaz"]
q=16
energies = zeros(Float64,size(H,2),size(H,3))
for i1 in 1:size(H,3)
    F = eigen(Hermitian(Pz[:,:,i1]))
    Hnew = F.vectors' * H[:,:,i1] * F.vectors
    energies[1:(q+1),i1] = eigvals(Hermitian(Hnew[1:(q+1),1:(q+1)]))
    energies[(q+2):(2q),i1] = eigvals(Hermitian(Hnew[(q+2):(2q),(q+2):(2q)]))
end
ϵ1 = energies[1:17,:]
ϵ2 = energies[18:32,:]
plot(ones(length(ϵ1))/16,ϵ1[:],"bo",ms=1,markerfacecolor="none")
plot(ones(length(ϵ2))/16.1,ϵ2[:],"ro",ms=1,markerfacecolor="none")
xlabel("p/q")
ylabel("Spectrum")
tight_layout()
display(fig)
savefig("StrongCoupling/hofstadter_chiralv2.pdf")
close(fig)

##
fig = figure(figsize=(4,3))
# plot(1:(q-1),energies[(q+2):(2q),:].-energies[q+2,1],"bo",ms=2,markerfacecolor="none")
# plot(1:(q+1),energies[1:(q+1),:].-energies[1,1],"ro",ms=2,markerfacecolor="none")
plot(1:(2q),energies,"go",ms=1,markerfacecolor="none")
# xlim([0,21])
# xticks(1:2:20)
ylabel("ϵ")
tight_layout()
savefig("StrongCoupling/hofstadter_chiral_q$(q)_p$(p)_kv2.pdf")

display(fig)
close(fig)


###
## testing discrepancy between nq=1 vs nq =2 
q = 4
p = 1
data = load("StrongCoupling/strongcoupling_q$(q)_p$(p)_chiralv2.jld")
H = data["spectrum"]
Pz = data["sigmaz"]
energies1 = zeros(Float64,size(H,2),size(H,3))
for i1 in 1:size(H,3)
    F = eigen(Hermitian(Pz[:,:,i1]))
    Hnew = F.vectors' * H[:,:,i1] * F.vectors
    energies1[1:(q+1),i1] = eigvals(Hermitian(Hnew[1:(q+1),1:(q+1)]))
    energies1[(q+2):(2q),i1] = eigvals(Hermitian(Hnew[(q+2):(2q),(q+2):(2q)]))
end

data = load("StrongCoupling/strongcoupling_q$(q)_p$(p)_chiralv2_test.jld")
H = data["spectrum"]
Pz = data["sigmaz"]
energies2 = zeros(Float64,size(H,2),size(H,3))
for i1 in 1:size(H,3)
    F = eigen(Hermitian(Pz[:,:,i1]))
    Hnew = F.vectors' * H[:,:,i1] * F.vectors
    energies2[1:(q+1),i1] = eigvals(Hermitian(Hnew[1:(q+1),1:(q+1)]))
    energies2[(q+2):(2q),i1] = eigvals(Hermitian(Hnew[(q+2):(2q),(q+2):(2q)]))
end

##
fig = figure(figsize=(3,3))
plot(ones(length(energies1))/q,energies1[:].- minimum(energies1),"r.",ms=2,markeredgecolor="none")
plot(ones(length(energies2))/q.+0.001,energies2[:] .- minimum(energies2),"b.",ms=2,markeredgecolor="none")
xlim([0.24,0.26])
tight_layout()
savefig("tmp.pdf")
display(fig)
close(fig)


##

q = 16
p = 1
data = load("StrongCoupling/strongcoupling_q$(q)_p$(p)_chiralv2.jld","spectrum")
H = data
l1 = 32
l2 = 2
energies = zeros(Float64,size(H,2),size(H,3))
for i1 in 1:size(H,3)
    energies[:,i1] = eigvals(Hermitian(H[:,:,i1]))
end

fig = figure(figsize=(3,3))
ϵ = reshape(energies,:,l1,l2)
ϵ1 = ϵ[:,:,1]
plot(ones(length(ϵ1))/q,ϵ1[:],"r.",markeredgecolor="none")
# xlim([0.24,0.26])
tight_layout()
# savefig("tmp.pdf")
display(fig)
close(fig)


##

fig = figure(figsize=(4,3))
plot(1:size(ϵ1,1),ϵ1,"rx",ms=3)
plot(1:size(ϵ1,1),data,"b+",ms=3)
# xlim([0.24,0.26])
tight_layout()
# savefig("tmp1.pdf")
display(fig)
close(fig)


##
using DelimitedFiles

data1 = readdlm("tbSpecq16.txt")