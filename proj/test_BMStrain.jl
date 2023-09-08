using PyPlot
using Printf
using DelimitedFiles
fpath = pwd()
include(joinpath(fpath,"libs/BM_Weizmann_mod.jl"))


##
params = Params(ϵ=0.005,φ=0.0)
initParamsWithStrain(params)
Latt = Lattice()
initLattice(Latt,params;lk=60)
blk = HBM()
initHBM(blk,Latt,params;lg=9)
writedlm(joinpath(fpath,"Stanford/strain_005.txt"),blk.Hk)
## 
# enegy jmap 
function plot_map(ϵ::Matrix{Float64},Latt::Lattice)
    fig,ax = subplots(2,figsize=(4,5))
    ϵ1 = reshape(ϵ[1,:],Latt.lk,Latt.lk)
    ϵ2 = reshape(ϵ[2,:],Latt.lk,Latt.lk)
    kvec = reshape(Latt.kvec,Latt.lk,Latt.lk) ./(params.kb)
    pl=ax[1].contourf(real(kvec),imag(kvec),ϵ1,20,cmap="Greens")
    colorbar(pl,ax=ax[1])
    ax[1].axis("equal")
    ax[1].plot([0;sqrt(3)/2],[1;1/2],"r+")

    pl=ax[2].contourf(real(kvec),imag(kvec),ϵ2,20,cmap="Blues_r")
    colorbar(pl,ax=ax[2])
    ax[2].axis("equal")
    ax[2].plot([0;sqrt(3)/2],[1;1/2],"r+")
    tight_layout()
    display(fig)
    savefig("Stanford/BM_map_strain_005.pdf",transparent=true)
    close(fig)
end 

plot_map(readdlm(joinpath(fpath,"Stanford/strain_001.txt")),Latt)

## filling fraction map 
function plot_map_filling(ϵ::Matrix{Float64},Latt::Lattice)
    fig,ax = subplots(2,figsize=(4,5))
    ϵ1 = reshape(ϵ[1,:],Latt.lk,Latt.lk) 
    ϵ2 = reshape(ϵ[2,:],Latt.lk,Latt.lk)
    levels1 = collect(-0.9:0.1:-0.1) * maximum(abs.(ϵ))
    # levels2 = collect(0.1:0.1:0.9) * maximum(abs.(ϵ))
    levels2 = [0.247;0.328] * maximum(abs.(ϵ))
    ν1 = [8*sum( (sign.(levels1[i] .- ϵ[:] ) .+1 )./2 ) / length(ϵ[:]) - 4 for i in eachindex(levels1)] 
    ν2 = [8*sum( (sign.(levels2[i] .- ϵ[:] ) .+1)./2 ) / length(ϵ[:]) - 4 for i in eachindex(levels2)]
    
    # note that this definition of filling fraction is incorrect if maximum(ϵ1) > minimum(ϵ2)
    kvec = reshape(Latt.kvec,Latt.lk,Latt.lk) ./(params.kb)
    pl = ax[1].pcolor(real(kvec),imag(kvec),ϵ1,cmap="Blues_r")
    colorbar(pl,ax=ax[1])
    pl=ax[1].contour(real(kvec),imag(kvec),ϵ1,levels=levels1,cmap="tab10")
    ν1str = Dict(pl.levels[i]=> @sprintf("%1.1f",ν1[i]) for i in eachindex(pl.levels))
    ax[1].clabel(pl,pl.levels,fmt=ν1str,inline=true,fontsize=6)
    ax[1].axis("equal")
    ax[1].plot([0;sqrt(3)/2],[1;1/2],"r+")

    pl = ax[2].pcolor(real(kvec),imag(kvec),ϵ2,cmap="Greens")
    colorbar(pl,ax=ax[2])
    pl=ax[2].contour(real(kvec),imag(kvec),ϵ2,levels=levels2,cmap="tab10")
    ν2str = Dict(pl.levels[i]=> @sprintf("%1.1f",ν2[i]) for i in eachindex(pl.levels))
    
    ax[2].clabel(pl,pl.levels,fmt=ν2str,inline=true,fontsize=6)
    ax[2].axis("equal")
    ax[2].plot([0;sqrt(3)/2],[1;1/2],"r+")
    tight_layout()
    display(fig)
    # savefig("Stanford/BM_map_strain_003_director45.pdf",transparent=true)
    close(fig)
end 

plot_map_filling(readdlm(joinpath(fpath,"Stanford/strain_001.txt")),Latt)


## filling fraction bounds van Hove 
function plot_map_filling_vanhove(ϵ::Matrix{Float64},Latt::Lattice)
    fig,ax = subplots(figsize=(4,2.5))
    ϵ2 = reshape(ϵ[2,:],Latt.lk,Latt.lk)
    levels2 = [0.2636;0.557] * maximum(abs.(ϵ))
    ν2 = [8*sum( (sign.(levels2[i] .- ϵ[:] ) .+1)./2 ) / length(ϵ[:]) - 4 for i in eachindex(levels2)]
    
    # note that this definition of filling fraction is incorrect if maximum(ϵ1) > minimum(ϵ2)
    kvec = reshape(Latt.kvec,Latt.lk,Latt.lk) ./(params.kb)

    pl = ax.pcolor(real(kvec),imag(kvec),ϵ2,cmap="Greens")
    colorbar(pl,ax=ax)
    pl=ax.contour(real(kvec),imag(kvec),ϵ2,levels=levels2,cmap="tab10")
    ν2str = Dict(pl.levels[i]=> @sprintf("%1.2f",ν2[i]) for i in eachindex(pl.levels))
    
    ax.clabel(pl,pl.levels,fmt=ν2str,inline=true,fontsize=8)
    ax.axis("equal")
    # ax.plot([0;sqrt(3)/2],[1;1/2],"r+")
    tight_layout()
    display(fig)
    savefig("Stanford/BM_vanhove_strain_005.pdf",transparent=true)
    close(fig)
end 

plot_map_filling_vanhove(readdlm(joinpath(fpath,"Stanford/strain_005.txt")),Latt)

##
# cut 
function plot_cuts(ϵ::Matrix{Float64},Latt::Lattice)
    lk = Latt.lk 
    ϵ = reshape(ϵ,2,lk,lk)
    # KΓ = [idK .- [2;1]*i for i in 0:(lk÷3)]
    # ΓM = [idΓ .+ [1;0]*i for i in 1:(lk÷2)]
    # MK = [idM .+ [1;2]*i for i in 1:(lk÷6)]
    idM = [lk÷2+1;1]
    idK = [2lk÷3+1;lk÷3+1]
    idK1 = [lk÷3+1;2lk÷3+1]
    idΓ = [1;1]
    ϵ1K1K = [ϵ[1,idK1[1]+i,idK1[2]-i] for i in 0:(lk÷3)]
    ϵ1KΓ = [ϵ[1,idK[1]-2i,idK[2]-i] for i in 1:(lk÷3)]
    ϵ1ΓM = [ϵ[1,idΓ[1]+i,idΓ[2]] for i in 1:(lk÷2)]
    # ϵ1MK = [ϵ[1,idM[1]+i,idM[2]+2i] for i in 1:(lk÷6)]
    ϵ1 = [ϵ1K1K;ϵ1KΓ;ϵ1ΓM]
    ϵ2K1K = [ϵ[2,idK1[1]+i,idK1[2]-i] for i in 0:(lk÷3)]
    ϵ2KΓ = [ϵ[2,idK[1]-2i,idK[2]-i] for i in 1:(lk÷3)]
    ϵ2ΓM = [ϵ[2,idΓ[1]+i,idΓ[2]] for i in 1:(lk÷2)]
    # ϵ2MK = [ϵ[2,idM[1]+i,idM[2]+2i] for i in 1:(lk÷6)]
    ϵ2 = [ϵ2K1K;ϵ2KΓ;ϵ2ΓM]

    fig = figure(figsize=(3,2.5))
    plot(eachindex(ϵ1), ϵ1,"b-")
    plot(eachindex(ϵ2),ϵ2,"g-")
    # axvline(1)
    # axvline(length(ϵ1K1K))
    # axvline(length(ϵ1KΓ)+length(ϵ1K1K))
    # axvline(length(ϵ1KΓ)+length(ϵ1K1K)+length(ϵ1ΓM))
    xticks([1,length(ϵ1K1K),length(ϵ1K1K)+length(ϵ1KΓ),length(ϵ1K1K)+length(ϵ1KΓ)+length(ϵ1ΓM)],
            [L"K_+",L"K_-",L"Γ",L"M"])
    # ylim([-6,6])
    tight_layout()
    display(fig)
    savefig("Stanford/BM_cuts_strain_003.pdf",transparent=true)
    close(fig)
end
plot_cuts(blk.Hk,Latt)

