# using PyPlot
using DelimitedFiles
using JLD
fpath = pwd()
include(joinpath(fpath,"libs/HofstadterVq_mod.jl"))

p = 1
q = parse(Int,ARGS[1])
ϕ = p//q
# q = 13
# ps = collect(1:3)
# ϕ = ps .// q 
# M = Dict()

w1=96.056
w0=0.0
dθ = 1.05
nLL= 25
params = Params(dθ=dθ*π/180,ϵ=0.0,w0=w0*w1,w1=w1);
# M = Dict()
# for iϕ in eachindex(ϕ)
    @time begin
        # p = ps[iϕ]
        println("ϕ= ",ϕ)
        hof = HofstadterVq()
        lk = 13
        initHofstadterVq(hof,params,p=p,q=q,lk=lk,nLL=nLL);
        # fname = "BMresults/HW_dtheta$(dθ)_w0$(w0)_p$(p)q$(q)_overlap.jld"
        fname = "VqLL/HW_dtheta$(dθ)_w0$(w0)_p$(p)q$(q)nLL$(nLL)_overlap_Asymptotic.jld"
        save(fname,"overlap",hof.Λ)
        # M["$iϕ"] = hof.O[:,:,1]
    end
# end
# save("overlapM_w00.7_n0_1.jld","overlap",M)

# fig,ax  = subplots(1,2,figsize=(4.5,2.5))
# # plot overlap eigenvalues if only one n0 chosen 
# M = load("overlapM_w00.7_n0_1.jld","overlap")
# for iϕ in eachindex(ϕ)
#     p = ps[iϕ]
#     ax[1].plot(1:(2q),reverse(eigvals(Hermitian(M["$iϕ"]))),".",ms=3,label="$(p)/$(q)")
# end
# ax[1].set_xticks([1,2q])
# ax[1].set_yticks([0,1,2])
# ax[1].set_xticklabels([1,"2q"])
# ax[1].legend()

# # plot overlap eigenvalues if 2 n0 chosen 
# M = load("overlapM_w00.7_n0_2.jld","overlap")
# for iϕ in eachindex(ϕ)
#     p = ps[iϕ]
#     ax[2].plot(1:(4q),reverse(eigvals(Hermitian(M["$iϕ"]))),".",ms=3,label="$(p)/$(q)")
# end
# ax[2].set_xticks([1,2q,4q])
# ax[2].set_xticklabels([1,"2q","4q"])
# ax[2].set_yticks([0,1,2,3])
# ax[2].legend()
# ax[2].axvline(2q+0.5,ls=":",c="gray")
# tight_layout(pad=1.3)
# savefig("overlap_eigvals_w00.7.pdf",transparent=true)
# display(fig)
# close(fig)