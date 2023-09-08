using PyPlot
using DelimitedFiles
using JLD
fpath = pwd()
include(joinpath(fpath,"libs/HofstadterLL_modv3.jl")) # multi k calculation

##
ϕ = 1 .// collect(4:18)
w1=96.056
w0=0.7w1
dθ = 1.05
params = Params(dθ=dθ*π/180,ϵ=0.0,w0=w0,w1=w1);

# compare flat band projector using hybrid wannier method vs LL exact method
function figure2(dθ::Float64,w0::Float64)
    qs = collect(2:18)
    ϕ = 1 .// qs
    d=load("BMresults/theta$(dθ)_LL_w0$(w0).jld")
    data = d["data"]
    # getting data
    fig = figure(figsize=(2.5,2))
    for iϕ in eachindex(ϕ)
        q = denominator(ϕ[iϕ])
        p = numerator(ϕ[iϕ])
        if q in [4,6,8,10]
            fname1 = "VqLL/HW_dtheta$(dθ)_w0$(w0)_p$(p)q$(q)nLL25_overlap_Asymptotic.jld"
            fname2 = "BMresults/LL_eigenstates_dtheta$(dθ)_w0$(w0)_q$(q).jld"

            Λ = load(fname1,"overlap")
            vec = load(fname2,"LLvec")
            Λfinal = Λ[:,:,1] * vec
            vals = real(diag(Λfinal'*Λfinal))
            
            lk = 10 
            if (q>=10)
                lk = 4 
            end
            if (q>20)
                lk = 1
            end
            l1,l2 = lk, lk
            iq = findall(qs .== q)[1]
            ϵ = reshape(data["$iq"],:,l1,l2)
            nH = 2*(25q) -1 
            mid_index = nH 
            ϵ = ϵ[:,1,1]
            irange = (mid_index-2q+1):(mid_index+2q)
            plot(vals[irange],(irange .- mid_index .-0.5)./maximum(irange.-mid_index.-0.5),".",ms=4,label=L"$1/%$(q)$")
        end
    end
    axhline(-0.5,color="gray",ls=":")
    axhline(0.5,color="gray",ls=":")
    xlim([-0.1,2])
    xticks([0,0.5,1])
    yticks([])
    legend(labelspacing=0.1,borderpad=0.2)
    tight_layout()
    # savefig("BMresults/overlapw00.7.pdf",transparent=true)
    display(fig)
    close(fig)
end

figure2(1.05,0.0)


##
function plot_svd_vals(dθ::Float64,w0::Float64)
    ϕ = 1 .// collect(4:9)
    fig = figure(figsize=(2,2))
    nLL = 25
    for iϕ in eachindex(ϕ)
        q = denominator(ϕ[iϕ])
        p = numerator(ϕ[iϕ])
        # fname1 = "BMresults/HW_dtheta$(dθ)_w0$(w0)_p$(p)q$(q)_overlap.jld"
        # fname2 = "BMresults/LL_eigenstates_dtheta$(dθ)_w0$(w0)_q$(q).jld"

        fname1 = "VqLL/HW_dtheta$(dθ)_w0$(w0)_p$(p)q$(q)nLL$(nLL)_overlap_Asymptotic.jld"
        fname2 = "VqLL/NarrowBandEigenstates_w0$(w0)_q$(q)_nLL$(nLL).jld"

        Λ = load(fname1,"overlap")
        vec = load(fname2,"vec")
        Λfinal = Λ[:,:,1] * vec[:,:,1,1]
        vals = svdvals(Λfinal*Λfinal')
        plot(ones(length(vals))*p/q,vals,".",ms=2)
    end
    
    ylim([0,1.1])
    yticks(collect(0:0.2:1))
    xlim([0,0.28])
    xticks(collect(0.05:0.1:0.25))
    # title("dθ=$dθ, w0/w1=$w0")
    tight_layout()
    # savefig("BMresults/overlap_svd_dθ$(dθ)_w0$(w0).pdf",transparent=true)
    display(fig)
    close(fig)
end

plot_svd_vals(1.05,0.7)


function plot_svd_weights(dθ::Float64,w0::Float64)
    qs = collect(2:25)
    p = 1 
    ϕ = p .// qs
    d=load("BMresults/theta$(dθ)_LL_w0$(w0).jld")
    data = d["data"]
    # getting data
    fig = figure(figsize=(2,2))
    pl = 0
    for iϕ in eachindex(ϕ)
        q = qs[iϕ]
        if q in 4:14
            fname1 = "BMresults/HW_dtheta$(dθ)_w0$(w0)_p$(p)q$(q)_overlap.jld"
            fname2 = "BMresults/LL_eigenstates_dtheta$(dθ)_w0$(w0)_q$(q).jld"

            Λ = load(fname1,"overlap")
            vec = load(fname2,"LLvec")
            
            # lk = 32 
            # if (q>=10)
            #     lk = 15 
            # end
            # if (q>=20)
            #     lk = 4 
            # end
            # l1,l2 = lk,lk
            l1,l2 = 1,1
            ϵfull = data["$iϕ"]
            nH = size(ϵfull,1) ÷ 2
            idx_flat = (nH+1-q):(nH+q)
            ϵ = ϵfull[idx_flat,1,1]
            Λfinal = Λ[:,:,1] * vec[:,idx_flat]
            
            plot(p/q,real(tr(Λfinal*Λfinal')) / (2q),"b.",ms=2)
            # pl=imshow(abs2.(Λfinal),origin="lower",vmin=0,vmax=0.5)
            # colorbar(pl) 
        end
    end
    ylim([0,1.1])
    yticks(collect(0:0.2:1))
    xlim([0,0.28])
    xticks(collect(0.05:0.1:0.25))
    tight_layout()
    savefig("BMresults/overlap_weight_dθ$(dθ)_w0$(w0).pdf",transparent=true)
    display(fig)
    close(fig)
end

plot_svd_weights(1.38,0.7)


##
function plot_overlap_combined()
    nLL=25
    qs = collect(4:10)
    p = 1 
    ϕ = p .// qs
    toplot = [[1.05;0.0] [1.05;0.7] [1.38;0.7]]
    fig, ax = subplots(2,3,sharex=true,sharey=true,figsize=(5,3))
    for c in 1:size(toplot,2)
        dθ = toplot[1,c]
        w0 = toplot[2,c]
        for iϕ in eachindex(ϕ)
            q = qs[iϕ]
            if q in 4:10
                fname1 = "VqLL/HW_dtheta$(dθ)_w0$(w0)_p$(p)q$(q)nLL$(nLL)_overlap_Asymptotic.jld"
                if c <3
                    fname2 = "VqLL/NarrowBandEigenstates_w0$(w0)_q$(q)_nLL$(nLL).jld"
                else
                    fname2 = "VqLL/NarrowBandEigenstates_dtheta1.38_w0$(w0)_q$(q)_nLL$(nLL).jld"
                end

                Λ = load(fname1,"overlap")
                vec = load(fname2,"vec")
                Λfinal = Λ[:,:,1] * vec[:,:,1,1]
                vals = svdvals(Λfinal*Λfinal')
                ax[1,c].plot(ones(length(vals))*p/q,vals,".",ms=2)
                ax[2,c].plot(p/q,real(tr(Λfinal*Λfinal')) / (2q),"b.",ms=2)
            end
        end
    end
    ax[1,1].set_ylim([0,1.1])
    ax[1,1].set_yticks(collect(0:0.2:1))
    ax[1,1].set_xlim([0,0.28])
    ax[1,1].set_xticks(collect(0.05:0.1:0.25))
    for c in 1:3 
        ax[2,c].set_xlabel(L"ϕ/ϕ_0")
    end
    
    tight_layout()
    savefig("Overlap_summary.pdf",transparent=true)
    display(fig)
    close(fig)
end

plot_overlap_combined()