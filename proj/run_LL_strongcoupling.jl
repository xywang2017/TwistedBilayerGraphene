using PyPlot
using JLD
using DelimitedFiles
fpath = pwd()
include(joinpath(fpath,"libs/HofstadterLL_modv3.jl"))
include(joinpath(fpath,"libs/HofstadterLL_modv4.jl"))

##
w1=96.056
w0=0.0
params = Params(dθ=1.05π/180,w0=w0,w1=w1,ϵ=0.0)
# ϕ = 1// parse(Int,ARGS[1])
ϕ = 1//1
q = denominator(ϕ)
p = numerator(ϕ)

# nLL = parse(Int,ARGS[2]) * q 
nLL = 20
hof = constructHofstadterLL(params,q=q,p=p,nLL=nLL,lk=10);
@time hof = constructHofstadterVqLL(params,q=q,p=p,nLL=nLL,lk=10);


nLL = nLL ÷ q
w00 = w0/w1
# writedlm("VqLL/LLdensitymatrix_w0$(w00)_q$(q)_nLL$(nLL).txt",abs.(hof.ρq[:,1:q:end])/(2q*hof.l1*hof.l2))
PΣz = load("VqLL/NarrowBandEigenstates_w0$(w00)_q$(q)_nLL$(nLL).jld","σz")
for iν in eachindex(hof.ν)
    H = hof.H[:,:,iν]
    F = eigen(Hermitian(H))
    σz = real(diag(F.vectors' * PΣz * F.vectors))
    νval = hof.ν[iν]
    fname = "VqLL/LLStrongCouplingSpectrum_w0$(w00)_q$(q)_nLL$(nLL)_nu$(νval).txt"
    writedlm(fname,[real(F.values) σz])
end
#         

# function plot_densitycomparison(w0::Float64)
#     fig = figure(figsize=(3,3))
#     ϕLL = 1 .// collect(4:12)
#     nLLs = [10;15;20;25]
#     ρLL = zeros(Float64,length(ϕLL),length(nLLs))
#     for iLL in eachindex(nLLs)
#         for iϕ in eachindex(ϕLL)
#             q = denominator(ϕLL[iϕ])
#             nLL = nLLs[iLL]
#             fname = "VqLL/LLdensitymatrix_w0$(w0)_q$(q)_nLL$(nLL).txt"
#             data = readdlm(fname)
#             ρLL[iϕ,iLL] = data[3,3]
#         end
#     end

#     if w0 == 0.0
#         ϕHW = 1 .// collect(4:17)
#     else
#         ϕHW = 1 .// collect(7:17)
#     end
#     ρHW = []
#     for iϕ in eachindex(ϕHW)
#         q = denominator(ϕHW[iϕ])
#         fname = "VqLL/HWSrhoq_w0$(w0)_q$(q).txt"
#         data = readdlm(fname)
#         ρHW = [ρHW; data[3,3]]
#     end

#     ρB0 = readdlm("VqLL/B0density_w0$(w0).txt")[3,3]
#     plot(0, ρB0/ρB0,"ko",ms=4)

#     plot(ϕHW,ρHW./ρB0,".",color="gray",label="hybrid Wannier")
    
#     # colors = [[0.2;cos(π/6*i);sin(π/6*i)] for i in 0:3]
#     for ic in 1:4
#         plot(ϕLL,ρLL[:,ic]./ρB0,"^",markersize=3,label=L"$n_{LL}=%$(nLLs[ic])q$")
#     end
    
    
#     # legend()
#     xlabel(L"ϕ/ϕ_0")
#     ylabel(L"$ρ_{g_1}/ρ_{g_1}(B=0)$")
#     ylim([0.85,1.02])
#     yticks([0.85,0.9,0.95,1])
#     tight_layout()
#     display(fig)
#     savefig("DensityComparison_w0$(w0).pdf",transparent=true)
#     close(fig)
# end
# plot_densitycomparison(0.7)

# a=3
#
function plot_spectrum_comparison(w0::Float64,q::Int,ν::Int)
    H = load("VqLL/LLStrongCouplingSpectrum_w0$(w0)_q$(q).jld","H")
    iν = (ν÷2)+2
    F = eigen(Hermitian(H[:,:,iν]))
    PΣz = load("VqLL/NarrowBandEigenstates_w0$(w0)_q$(q).jld","σz")
    σz = real(diag(F.vectors' * PΣz * F.vectors))

    fig = figure(figsize=(3,3)) 
    scatter(ones(2q)*1,F.values,c=σz,s=6,marker=".",cmap="Spectral",vmin=-1,vmax=1)

    data = readdlm("finitenu/q$(q)_nu$(ν)_w0$(w0)_svec4.txt")
    ϵ = data[1:(2q),1]
    σz = data[1:(2q),2]
    # ϵ = data[:,1]
    # σz = data[:,2]
    scatter(ones(length(ϵ))*2,ϵ,c=σz,marker=".",s=6,cmap="Spectral",vmin=-1,vmax=1)

    # ylim([0.7,1.8])
    xticks([1,2],["LL","hWS"])
    xlim([-1,4])
    tight_layout()
    # savefig("VqLL/comparison_w0$(w0)_q$(q)_nu$(ν).pdf",transparent=true)
    display(fig)
    close(fig)
end

plot_spectrum_comparison(0.0,5,0)


function plot_spectra_sequence_LL(w0::Float64,ν::Int)
    nLL=25
    ϕ = 1 .// collect(4:8)
    fig,ax = subplots(1,2,sharex=true,sharey=true,figsize=(4.2,2.4)) 
    for iϕ in eachindex(ϕ)
        q = denominator(ϕ[iϕ])
        fname = "VqLL/LLStrongCouplingSpectrum_w0$(w0)_q$(q)_nLL$(nLL)_nu$(ν).txt"
        data = readdlm(fname)
        ϵ = data[:,1]
        σz = data[:,2]
        ax[2].scatter(ones(2q)*ϕ[iϕ],ϵ,c=σz,s=4,marker=".",cmap="Spectral",vmin=-1,vmax=1)
    end
    # ax[1].set_xlabel(L"ϕ/ϕ_0")
    # ax[1].set_ylabel(L"E")
    ax[1].set_xlim([0,0.28])
    # ax[1].set_ylim([0.7,1.8])
    ax[1].set_yticks(collect(-0.5:0.5:0.7))

    if w0 == 0.0
        ϕ = 1 .// collect(4:25)
    else
        ϕ = 1 .// collect(8:25)
    end
    for iϕ in eachindex(ϕ)
        q = denominator(ϕ[iϕ])
        data = readdlm("finitenu/q$(q)_nu$(ν)_w0$(w0)_svec4.txt")
        ϵ = data[1:(2q),1]
        σz = data[1:(2q),2]
        ax[1].scatter(ones(length(ϵ))*ϕ[iϕ],ϵ,c=σz,marker=".",s=4,cmap="Spectral",vmin=-1,vmax=1)
    end
    tight_layout(w_pad=0)
    display(fig)
    savefig("VqLL/comparison_w0$(w0)_nu$(ν).pdf",transparent=true)
    close(fig)
    return nothing
end

plot_spectra_sequence_LL(0.7,-2)



function plot_spectra_sequence_LL_in_one(w0::Float64,ν::Int)
    nLL=25
    
    fig,ax = subplots(figsize=(3,3)) 

    if w0 == 0.0
        ϕ = 1 .// collect(4:25)
    else
        ϕ = 1 .// collect(8:25)
    end
    for iϕ in eachindex(ϕ)
        q = denominator(ϕ[iϕ])
        data = readdlm("finitenu/q$(q)_nu$(ν)_w0$(w0)_svec4.txt")
        ϵ = data[:,1]
        σz = data[:,2]
        println(q," ",length(ϵ)/(2q))
        ax.scatter(ones(length(ϵ))*ϕ[iϕ],ϵ,c="b",marker=".",s=6,)
    end

    ϕ = 1 .// collect(4:8)
    for iϕ in eachindex(ϕ)
        q = denominator(ϕ[iϕ])
        fname = "VqLL/LLStrongCouplingSpectrum_w0$(w0)_q$(q)_nLL$(nLL)_nu$(ν).txt"
        data = readdlm(fname)
        ϵ = data[:,1]
        σz = data[:,2]
        ax.scatter(ones(2q)*ϕ[iϕ],ϵ,c="r",s=6,marker=".")
    end
    ax.set_xlabel(L"ϕ/ϕ_0")
    ax.set_ylabel(L"E")
    ax.set_xlim([0,0.28])
    # ax[1].set_ylim([0.7,1.8])
    # ax.set_yticks(collect(-0.5:0.5:0.7))

    ax.set_title("w0/w1=0.")
    tight_layout(w_pad=0)
    display(fig)
    savefig("VqLL/comparison_w0$(w0)_nu$(ν).pdf",transparent=true)
    close(fig)
    return nothing
end

plot_spectra_sequence_LL_in_one(0.,0)