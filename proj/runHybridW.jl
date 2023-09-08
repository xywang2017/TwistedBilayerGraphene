using PyPlot
using DelimitedFiles
using JLD
fpath = pwd()
include(joinpath(fpath,"libs/Hofstadter_mod.jl"))

##
# w1=96.056
# w0=0.0
# params = Params(dθ=1.05π/180,w0=w0,w1=w1,ϵ=0.0)  #chiral limit
params = Params() # Weizmann group data
q = 101
ps = collect(1:2)

##
function plot_spectrum_comparison(dθ::Float64,w0::Float64)
    fig = figure(figsize=(4,3))

    d=load("BMresults/theta$(dθ)_LL_w0$(w0).jld")
    data = d["data"]
    σz = d["σz"]
    qs = collect(2:18)
    for iq in eachindex(qs)
        p=1
        q = qs[iq]
        lk = 10 
        if (q>=10)
            lk = 4 
        end
        if (q>20)
            lk = 1
        end
        l1,l2 = lk,lk
        ϵ = sort(data["$iq"],by=abs)[1:(2q*l1*l2)]
        ϕ = ones(size(ϵ)) * p/q
        plot(ϕ,100ϵ,".",c="gray",ms=2,markeredgecolor="none")
    end

    q = 83
    ps = collect(1:20)
    ϕ = ps .// q 
    for iϕ in eachindex(ϕ)
        p = ps[iϕ]
        data = readdlm("BMresults/theta$(dθ)_q$(q)p$(p)_w0$(w0).txt")
        ϵ = data[:,1]
        σz = data[:,2]
        plot(ones(length(ϵ))*p/q,100ϵ,"b.",ms=2,markeredgecolor="none")
    end

    ylabel(L"ϵ/ϵ_0")
    xlabel(L"ϕ/ϕ_0")
    xlim([0,0.51])
    ylim([-0.2,0.2])
    yticks(collect(-1:0.5:1))
    tight_layout()
    display(fig)
    savefig("BMresults/theta$(dθ)_w0$(w0)_comparison.png",transparent=true,dpi=300)
    close(fig)
end

plot_spectrum_comparison(1.05,0.7)

##
function sublattice_polarization(dθ::Float64,w0::Float64)
    fig = figure(figsize=(3,3))

    d=load("BMresults/theta$(dθ)_LL_w0$(w0).jld")
    σz = d["σz"]
    # nσz = [sum(σz[i])/length(σz[i]) for i in eachindex(σz)]
    nσz = [σz[i][1,1] for i in eachindex(σz)]
    qs = collect(2:25)
    plot(1 .//qs,nσz,".",c="gray",ms=6,markeredgecolor="none",label="Landau Level")

    q = 83
    ps = collect(1:20)
    ϕ = ps .// q 
    σz = zeros(length(ϕ))
    for iϕ in eachindex(ϕ)
        p = ps[iϕ]
        data = readdlm("BMresults/theta$(dθ)_q$(q)p$(p)_w0$(w0).txt")
        σz[iϕ] = sum(data[:,2])
    end
    
    plot(ϕ,σz ./ps ,"b.",ms=6,markeredgecolor="none",label="Hybrid Wannier")
    legend()
    ylabel(L"⟨σ_z⟩")
    xlabel(L"ϕ/ϕ_0")
    xlim([0,0.51])
    ylim([0,2.2])
    tight_layout()
    display(fig)
    savefig("BMresults/theta$(dθ)_w0$(w0)_sublatticepolarization.pdf",transparent=true)
    close(fig)
end

sublattice_polarization(1.05,0.)


##

##
function plot_spectrum_comparison(dθ::Float64,w0::Float64)
    fig = figure(figsize=(4,3))

    d=load("BMresults/theta$(dθ)_LL_w0$(w0).jld")
    data = d["data"]
    σz = d["σz"]
    qs = collect(2:25)
    for iq in eachindex(qs)
        q = qs[iq]
        p = 1
        l1 = 1
        l2 = 1
        # ϵ = sort(data["$iq"],by=abs)[1:(2q*l1*l2)]
        nH = 2*(10q)-1
        idx_flat = (nH+1-q):(nH+q)
        ϵ= data["$iq"][idx_flat]
        ϕ = ones(size(ϵ)) * p/q
        fname2 = "BMresults/LL_eigenstates_dtheta$(dθ)_w0$(w0)_q$(q).jld"
        σz = load(fname2,"sigmaz")[:]
        scatter(ϕ,ϵ,c=σz,s=1,cmap="coolwarm",vmin=-1,vmax=1)
    end

    q = 83
    ps = collect(1:20)
    ϕ = ps .// q 
    for iϕ in eachindex(ϕ)
        p = ps[iϕ]
        data = readdlm("BMresults/theta$(dθ)_q$(q)p$(p)_w0$(w0).txt")
        ϵ = data[:,1]
        σz = data[:,2]
        scatter(ones(length(ϵ))*p/q,ϵ,c=σz,s=1,cmap="coolwarm",vmin=-1,vmax=1)
    end

    ylabel(L"ϵ/ϵ_0")
    xlabel(L"ϕ/ϕ_0")
    xlim([0,0.51])
    ylim([-0.2,0.2])
    tight_layout()
    display(fig)
    savefig("BMresults/theta$(dθ)_w0$(w0)_comparison.pdf",transparent=true)
    close(fig)
end

plot_spectrum_comparison(1.38,0.7)



function plot_sigmaz_vs_energy(dθ::Float64,w0::Float64)
    fig = figure(figsize=(4,3))

    d=load("BMresults/theta$(dθ)_LL_w0$(w0).jld")
    data = d["data"]
    σz = d["σz"]
    qs = collect(2:25)
    for iq in eachindex(qs)
        q = qs[iq]
        p = 1
        l1 = 1
        l2 = 1
        # ϵ = sort(data["$iq"],by=abs)[1:(2q*l1*l2)]
        nH = 2*(10q)-1
        idx_flat = (nH+1-q):(nH+q)
        ϵ= data["$iq"][idx_flat]
        ϕ = ones(size(ϵ)) * p/q
        fname2 = "BMresults/LL_eigenstates_dtheta$(dθ)_w0$(w0)_q$(q).jld"
        σz = load(fname2,"sigmaz")[:]
        if q in [25]
            plot(ϵ,σz,"o",ms=3,label="LL")
        end
    end

    # q = 83
    # ps = collect(1:20)
    p = 1
    qs = [15;20;25]
    ϕ = p .// qs
    for iϕ in eachindex(ϕ)
        q = qs[iϕ]
        data = readdlm("BMresults/theta$(dθ)_q$(q)p$(p)_w0$(w0).txt")
        ϵ = data[:,1]
        σz = data[:,2]
        if q in [25]
            plot(ϵ,σz,"x",ms=3,label="Hybrid")
        end
    end
    legend()
    title("1/25")
    tight_layout()
    display(fig)
    close(fig)
end

plot_sigmaz_vs_energy(1.38,0.7)




function plot_sigmaz_vs_energyv2(dθ::Float64,w0::Float64)
    fig = figure(figsize=(4,3))

    d=load("BMresults/theta$(dθ)_LL_w0$(w0).jld")
    data = d["data"]
    σz = d["σz"]
    qs = collect(2:25)
    for iq in eachindex(qs)
        q = qs[iq]
        p = 1
        l1 = 1
        l2 = 1
        # ϵ = sort(data["$iq"],by=abs)[1:(2q*l1*l2)]
        nH = 2*(10q)-1
        idx_flat = (nH+1-q):(nH+q)
        ϵ= data["$iq"][idx_flat]
        ϕ = ones(size(ϵ)) * p/q
        fname2 = "BMresults/LL_eigenstates_dtheta$(dθ)_w0$(w0)_q$(q).jld"
        σz = load(fname2,"sigmaz")[:]
        if q in [15,20,25]
            data1 = readdlm("BMresults/theta$(dθ)_q$(q)p$(p)_w0$(w0).txt")
            ϵ1 = data1[:,1]
            σz1 = data1[:,2]
            plot(1/q,norm(σz.-σz1)/sqrt(q),"o",ms=3)
        end
    end
    xlim([0,0.13])
    ylim([0,0.02])
    tight_layout()
    display(fig)
    close(fig)
end

plot_sigmaz_vs_energyv2(1.38,0.7)