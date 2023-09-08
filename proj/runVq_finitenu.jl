using PyPlot
using DelimitedFiles
using JLD
fpath = pwd()
# include(joinpath(fpath,"libs/HofstadterVq_modv3.jl")) # single k calculation
# include(joinpath(fpath,"libs/HofstadterVq_modv3_kselect.jl")) # multi k calculation
include(joinpath(fpath,"libs/HofstadterVq_modv3_old.jl")) # multi k calculation

ϕ = 1 .// collect(8:25)
# ϕ = [1 // parse(Int,ARGS[1])]
# ϕ = [1//18] 
w1=96.056
# w1 = 110
w0=0.0
dθ = 1.05
params = Params(dθ=dθ*π/180,ϵ=0.0,w0=w0,w1=w1);
hof = 0
ϵ = 0
##
for iϕ in eachindex(ϕ)
    @time begin
        p = numerator(ϕ[iϕ])
        q = denominator(ϕ[iϕ])
        println(ϕ[iϕ]," w0=",w0," w1=",w1)
        hof = HofstadterVq()
        lk = 8
        initHofstadterVq(hof,params,p=p,q=q,lk=lk);
        # ϵ = zeros(Float64,size(hof.H,2),size(hof.H,3))
        # σz = zeros(Float64,size(hof.H,2),size(hof.H,3))
        ϵ = zeros(Float64,size(hof.H0,2),size(hof.H0,3),length(hof.ν))
        for iν in eachindex(hof.ν)
            # for ik in 1:size(hof.H,3)
            for ik in 1:size(hof.H0,3)
                # F = eigen(Hermitian(hof.H[:,:,ik,iν]))
                F = eigen(Hermitian(hof.H0[:,:,ik,iν]))
                # σz[:,ik] = diag(real(F.vectors'*hof.Σz[:,:,ik]*F.vectors))
                ϵ[:,ik,iν] = F.values
            end
            νval = hof.ν[iν]
            # fname = "finitenu/q$(q)_nu$(νval)_w00.6.txt"
            # writedlm(fname,[ϵ[:] σz[:]])
            # fname = "finitenu/q$(q)_nu$(νval)_w00.0_H0.txt"
            # open(fname, "a") do io
            #     writedlm(io, reshape(ϵ[:,:,iν],:))
            # end                  
        end
    end
end

##
# this code plots B=0 dispersion, works with v3_old.jl code
##
function plot_energy_cut_B0(iν::Int,w0::Float64)
    # only works for q =18, l1 = 36
    iΓ = [1;1]
    iM = [19;1]
    iK = [25;13]
    cutKΓ = [iK - [2i;i] for i in 0:12]
    cutΓM = [iΓ + [i;0] for i in 1:18]
    cutMK = [iM + [i;2i] for i in 1:6]
    
    kvec = hof.kvec
    # plot energy cut along Γ -> M 
    l1 = size(kvec,1)
    energies = reshape(ϵ[:,:,iν],2,l1,l1)
    ϵcut = zeros(2,length(cutΓM)+length(cutMK)+length(cutKΓ))
    cnt = 1
    for i1 in eachindex(cutKΓ)
        ϵcut[:,cnt] = energies[:,cutKΓ[i1][1],cutKΓ[i1][2]]
        cnt +=1
    end
    for i1 in eachindex(cutΓM)
        ϵcut[:,cnt] = energies[:,cutΓM[i1][1],cutΓM[i1][2]]
        cnt +=1
    end
    for i1 in eachindex(cutMK)
        ϵcut[:,cnt] = energies[:,cutMK[i1][1],cutMK[i1][2]]
        cnt +=1
    end
    
    fig = figure(figsize=(2.5,2))
    plot(1:size(ϵcut,2),ϵcut[1,:],"k-")
    plot(1:size(ϵcut,2),ϵcut[2,:],"k-")
    # ylim([0.7,1.8])
    yticks([])
    xticks([1,length(cutKΓ),length(cutKΓ)+length(cutΓM),size(ϵcut,2)],["K","Γ","M","K"])
    # axis("off")
    tight_layout()
    νval = hof.ν[iν]
    savefig("energycut_nu$(νval)_w0$(w0).pdf",transparent=true)
    display(fig)
end

plot_energy_cut_B0(1,0.7)
##
function plot_strongcoupling_dispersion_dos(ν::Int,w0::Float64)
    LL = []
    if w0==0.0
        # fig,ax = subplots(1,2,sharey=true,figsize=(2.8,2.8),gridspec_kw=Dict("width_ratios" => [1,1.1]))  #(4,3)[1,2.5]
        fig,ax = subplots(1,2,sharey=true,figsize=(4,3),gridspec_kw=Dict("width_ratios" => [1,2.5]))
    else
        # fig,ax = subplots(1,2,sharey=true,figsize=(2.8,2.8),gridspec_kw=Dict("width_ratios" => [1,1.1])) 
        fig,ax = subplots(1,2,sharey=true,figsize=(2.8,3),gridspec_kw=Dict("width_ratios" => [1,1.1]))
    end
    #dos 
    energies = readdlm("finitenu/q18_nu$(ν)_w0$(w0)_H0.txt")[:];
    dos = computeDensityofStatesB0v2(energies,γ=0.01);
    ax[1].plot(dos[:,2],dos[:,1],"-",c="gray")
    ax[1].invert_xaxis()
    if w0 == 0.7
        # ax[1].set_ylim([-0.6,4])
        # ax[1].set_yticks([0,1,2,3,4])
    else
        ax[1].set_ylim([0.7,1.8])
        # ax[1].set_yticks([0,1,2])
    end
    ax[1].set_xticks([])
    pl = 0
    for iϕ in eachindex(ϕ)
        p = numerator(ϕ[iϕ])
        q = denominator(ϕ[iϕ])

        data = readdlm("finitenu/q$(q)_nu$(ν)_w0$(w0)_svec4.txt")
        ϵ = data[1:(2q),1]
        σz = data[1:(2q),2]
        push!(LL,ϵ[1])
        pl=ax[2].scatter(ones(length(ϵ))*p/q,ϵ,c=σz,marker=".",s=4,cmap="Spectral",vmin=-1,vmax=1)

    end

    if w0 == 0.0
        ax[2].set_xlim([0.0,0.28])
        ax[2].set_xticks(collect(0.05:0.05:0.25))
        # ax[2].set_xticks(collect(0.1:0.1:0.2))
    else
        ax[2].set_xlim([0,0.13])
        # ax[2].set_xticks([0.05,0.1])
    end
    ax[2].axes.get_yaxis().set_visible(false)
    ax[1].spines["right"].set_visible(false)
    tight_layout(w_pad=-0.3)
    display(fig)
    # savefig("finitenu/hofstadter_nu$(ν)_w0$(w0)_singleK.pdf",transparent=true)
    close(fig)

    # return nothing 
    return LL 
end
using DataFrames
df = DataFrame()
LL=plot_strongcoupling_dispersion_dos(0,0.7)
df[!,"ϕ"] = ϕ
df[!,"w0_07_ν_0"] = LL

fig = figure(figsize=(3,3))
styles = ["ro","r^","bo","b^"]
labels = ["w0=0,ν=0","w0=0,ν=2","w0=0.7,ν=0","w0=0.7,ν=2"]
cnt = 1
for w0 in ["00","07"], ν in ["0","2"]
    plot(df[!,"ϕ"],df[!,"w0_"*w0*"_ν_"*ν].-df[length(ϕ),"w0_"*w0*"_ν_"*ν],styles[cnt],ms=2,label=labels[cnt])
    cnt = cnt + 1
end
legend()
xlim([0,0.1])
xticks([0,0.06,0.12])
ylim([-0.1,0.2])
xlabel("ϕ/ϕ0")
ylabel("lowest LL")
tight_layout()
display(fig)
savefig("LLL.pdf",transparent=true)
close(fig)



fig = figure(figsize=(3,3))
styles = ["ro","r^","bo","b^"]
labels = ["w0=0,ν=0","w0=0,ν=2","w0=0.7,ν=0","w0=0.7,ν=2"]
cnt = 1
for w0 in ["00","07"], ν in ["0","2"]
    plot((df[1:(length(ϕ)-1),"ϕ"]+df[2:(length(ϕ)),"ϕ"])/2,
            diff(df[!,"w0_"*w0*"_ν_"*ν]) ./ diff(df[!,"ϕ"]),
            styles[cnt],ms=3,label=labels[cnt])
    cnt = cnt + 1
end
legend()
xlim([0,0.1])
xticks([0,0.06,0.12])
xlabel("ϕ/ϕ0")
ylabel("slope from finite diff")
tight_layout()
display(fig)
savefig("LLL_slope.pdf",transparent=true)
close(fig)
## # charge gap in field
function plot_strongcoupling_dispersion_gap(w0::Float64)
    
    Δ = zeros(Float64,length(ϕ)+1)
    flux = [0; ϕ]
    # B = 0
    ϵ_el = minimum(readdlm("finitenu/q18_nu2_w0$(w0)_H0.txt")[:]);
    ϵ_hl = minimum(readdlm("finitenu/q18_nu-2_w0$(w0)_H0.txt")[:]);
    Δ[1] = ϵ_el + ϵ_hl
    for iϕ in eachindex(ϕ)
        p = numerator(ϕ[iϕ])
        q = denominator(ϕ[iϕ])
        ϵ_el = minimum(readdlm("finitenu/q$(q)_nu2_w0$(w0).txt")[:,1])
        ϵ_hl = minimum(readdlm("finitenu/q$(q)_nu-2_w0$(w0).txt")[:,1])
        Δ[iϕ+1] = ϵ_el + ϵ_hl
    end
    
    fig = figure(figsize=(3,3))
    plot(flux,Δ,"g.")
    xlabel(L"ϕ/ϕ_0")
    ylabel(L"$Δ_{ph}$")
    title(L"$w_0/w_1=%$(w0)$")
    tight_layout()
    display(fig)
    close(fig)
    return nothing 
end
# plot_strongcoupling_dispersion_gap(0.)
