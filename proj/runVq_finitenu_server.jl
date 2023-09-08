using DelimitedFiles
using JLD
fpath = pwd()
include(joinpath(fpath,"libs/HofstadterVq_modv3.jl")) # single k calculation
# include(joinpath(fpath,"libs/HofstadterVq_modv3_kselect.jl")) # multi k calculation

ϕ = [1 // parse(Int,ARGS[1])]
# ϕ = [1//7]
w1=96.056
w0=0.7w1
dθ = 1.05
params = Params(dθ=dθ*π/180,ϵ=0.0,w0=w0,w1=w1);
##
for iϕ in eachindex(ϕ)
    @time begin
        p = numerator(ϕ[iϕ])
        q = denominator(ϕ[iϕ])
        println(ϕ[iϕ]," w0=",w0," w1=",w1)
        hof = HofstadterVq()
        lk = 10
        initHofstadterVq(hof,params,p=p,q=q,lk=lk);
        # ϵ = zeros(Float64,size(hof.H,2),size(hof.H,3))
        # σz = zeros(Float64,size(hof.H,2),size(hof.H,3))
        # for iν in eachindex(hof.ν)
        #     for ik in 1:size(hof.H,3)
        #         F = eigen(Hermitian(hof.H[:,:,ik,iν]))
        #         if size(hof.H,3) == 1
        #             σz[:,ik] = diag(real(F.vectors'*hof.Σz[:,:,ik]*F.vectors))
        #         else
        #             σz[:,ik] = diag(real(F.vectors'*hof.Σz[:,:,hof.n_ext[ik]]*F.vectors))
        #         end
        #         ϵ[:,ik] = F.values
        #     end
        #     νval = hof.ν[iν]
            # fname = "finitenu/q$(q)_nu$(νval)_w00.7_n0.txt"
            # fname = "VqLL/q$(q)_nu$(νval)_w00.0.txt"
            # writedlm(fname,[ϵ[:] σz[:]])
        # end
    end
end
