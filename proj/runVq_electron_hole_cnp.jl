using PyPlot
using DelimitedFiles
fpath = pwd()
include(joinpath(fpath,"libs/HofstadterVq_mod.jl")) # single k calculation
# include(joinpath(fpath,"libs/HofstadterVq_modv3_kselect.jl")) # multi k calculation

# ϕ = [1 // parse(Int,ARGS[1])]
ϕ = [1//6]
w1 = 96.056
w0=0.6
dθ = 1.05
params = Params(dθ=dθ*π/180,ϵ=0.0,w0=w0*w1,w1=w1);
##
for iϕ in eachindex(ϕ)
    @time begin
        p = numerator(ϕ[iϕ])
        q = denominator(ϕ[iϕ])
        println(ϕ[iϕ]," w0=",w0," w1=",w1)
        hof = HofstadterVq()
        lk = 8
        initHofstadterVq(hof,params,p=p,q=q,lk=lk);
        ϵ = zeros(Float64,size(hof.H,2),size(hof.H,3))
        σz = zeros(Float64,size(hof.H,2),size(hof.H,3))
        for ik in 1:size(hof.H,3)
            F = eigen(Hermitian(hof.H[:,:,ik]))
            σz[:,ik] = diag(real(F.vectors'*hof.Σz[:,:,ik]*F.vectors))
            ϵ[:,ik] = F.values
        end
        fname = "q$(q)_w0$(w0)_electron.txt"
        writedlm(fname,[ϵ[:] σz[:]])      
    end
end

#
q=6
data = readdlm("q$(q)_w0$(w0)_electron.txt")
ϵ_electron = data[:,1]
σz_electron = data[:,2]
data = readdlm("q$(q)_w0$(w0)_hole.txt")
ϵ_hole = data[:,1]
σz_hole = data[:,2]

#
fig = figure(figsize=(3,3))
scatter(ones(length(ϵ_electron)),ϵ_electron,c=σz_electron,marker=".",s=4,cmap="Spectral",vmin=-1,vmax=1)
scatter(ones(length(ϵ_hole))*2,ϵ_hole,c=σz_hole,marker=".",s=4,cmap="Spectral",vmin=-1,vmax=1)
xticks([1,2],["electron","hole"])
xlim([-1,4])
title("charge neutrality w0=0.6w1")
tight_layout()
savefig("electron_hole_cnp.pdf")
display(fig)
close(fig)