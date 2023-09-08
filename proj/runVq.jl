using PyPlot
using DelimitedFiles
using Roots
using JLD
fpath = pwd()
# include(joinpath(fpath,"libs/HofstadterVq_modv1.jl")) # single k calculation
include(joinpath(fpath,"libs/HofstadterVq_mod.jl")) # multi k calculation
# include(joinpath(fpath,"libs/HofstadterVq_mod_ksample.jl")) # select k points calculation

# ϕ = 1 .// collect(4:25)
# ϕ = [1//18]
w1=96.056
w0=0.0
if w0 == 0.7w1 
    ϕ = 1 .// collect(8:25)
else
    ϕ = 1 .// collect(4:25)
end
# ϕ = [1//12;1//13;1//14;1//15]
dθ = 1.05
params = Params(dθ=dθ*π/180,ϵ=0.0,w0=w0,w1=w1);

if !isdir("StrongCoupling/dtheta$(dθ)_w0$(w0)_w1$(w1)")
    mkdir("StrongCoupling/dtheta$(dθ)_w0$(w0)_w1$(w1)")
end

ϕ = [1//20]
# getting data
energies = []
kvec = []
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
        # for ik in 1:size(hof.H,3)
        #     F = eigen(Hermitian(hof.H[:,:,ik]))
        #     σz[:,ik] = diag(real(F.vectors'*hof.Σz[:,:,hof.n_ext[ik]]*F.vectors))
        #     ϵ[:,ik] = F.values
        # end
        # fname = "StrongCoupling/dtheta$(dθ)_w0$(w0)_w1$(w1)/dtheta$(dθ)_w0$(w0)_w1$(w1)_p$(p)q$(q)_fullgrid.txt"
        # writedlm(fname,[ϵ[:] σz[:]])

        fname = "StrongCoupling/dtheta$(dθ)_w0$(w0)_w1$(w1)/dtheta$(dθ)_w0$(w0)_w1$(w1)_dos0.txt"
        open(fname, "a") do io
            writedlm(io, hof.dos[:])
        end   
        # energies = hof.dos[:]
        # kvec = hof.kvec
        # fname = "StrongCoupling/dtheta$(dθ)_w0$(w0)_w1$(w1)/dtheta$(dθ)_w0$(w0)_w1$(w1)_p$(p)q$(q)_overlap.jld"
        # save(fname,"overlap",hof.Λ)
        
    end
end

##
function plot_energy_cut_B0()
    # plot energy cut along Γ -> M 
    l1 = size(kvec,1)
    ϵ = reshape(energies,2,l1,l1)
    ϵΓM = circshift(ϵ[:,:,1],(0,l1÷2))
    kvecΓM = circshift( real(kvec[:,1]) ./abs(params.g1), l1÷2)
    kvecΓM = [kvecΓM; kvecΓM[1]]
    ϵΓM = [ϵΓM ϵΓM[:,1]]
    fig = figure(figsize=(2.5,2))

    plot(eachindex(kvecΓM),ϵΓM[1,:],"b-")
    plot(eachindex(kvecΓM),ϵΓM[2,:],"r-")
    ylim([0.7,1.8])
    yticks([])
    xticks([1,l1÷2+1,l1+1],["M","Γ","M"])
    # axis("off")
    tight_layout()
    savefig("energycut_w0.7.pdf",transparent=true)
    display(fig)
end

# plot_energy_cut_B0()

##
function plot_strongcoupling_dispersion_dos(ν::Int)
    fig,ax = subplots(1,2,sharey=true,figsize=(4.,3),gridspec_kw=Dict("width_ratios" => [1,2.5])) 
    # fig,ax = subplots(1,2,sharey=true,figsize=(2.8,3),gridspec_kw=Dict("width_ratios" => [1,1.1])) 

    #dos 
    energies = readdlm("StrongCoupling/dtheta$(dθ)_w0$(w0)_w1$(w1)/dtheta$(dθ)_w0$(w0)_w1$(w1)_dos0.txt")[:];
    dos = computeDensityofStatesB0v2(energies,γ=0.015);
    ax[1].plot(dos[:,2],dos[:,1],"-",c="gray")
    ax[1].invert_xaxis()
    # ax[1].set_ylim([0.7,1.8])
    ax[1].set_xticks([])
    pl = 0
    Eplus = zeros(Float64,length(ϕ))
    Eminus = zeros(Float64,length(ϕ))
    for iϕ in eachindex(ϕ)
        p = numerator(ϕ[iϕ])
        q = denominator(ϕ[iϕ])

        # if q==30
        #     data = readdlm("StrongCoupling/office_data/dtheta$(dθ)_w0$(w0)_w1$(w1)_p$(p)q$(q).txt")
        # elseif (q>13)
        #     data = readdlm("StrongCoupling/dtheta$(dθ)_w0$(w0)_w1$(w1)/dtheta$(dθ)_w0$(w0)_w1$(w1)_p$(p)q$(q).txt")
        # elseif q in 7:11
        #     data = readdlm("StrongCoupling/office_data/dtheta$(dθ)_w0$(w0)_w1$(w1)_p$(p)q$(q)_fullgrid.txt")
        # else
        #     data = readdlm("StrongCoupling/dtheta$(dθ)_w0$(w0)_w1$(w1)/dtheta$(dθ)_w0$(w0)_w1$(w1)_p$(p)q$(q)_fullgrid.txt")
        # end 
        data = readdlm("finitenu/q$(q)_nu$(ν)_w00.0_svec4.txt")
        
        ϵ = reshape(data[:,1],2q,:)
        lk = size(ϵ,2)
        ϵ = ϵ[:]
        σz = data[:,2]
        println(minimum(abs.(σz)))
        ϵplus = ϵ[σz .> 0]
        ϵminus = ϵ[σz .< 0]
        Eplus[iϕ] = sum(ϵplus) / (lk*q)
        Eminus[iϕ] = sum(ϵminus) / (lk*q)
        # println(length(ϵplus)/lk-length(ϵminus)/lk)
        pl=ax[2].scatter(ones(length(ϵ))*p/q,ϵ,c=σz,marker=".",s=4,cmap="Spectral",vmin=-1,vmax=1)
    end
    ax[2].set_xlim([0.0,0.28])
    # ax[2].set_xticks([0.05,0.1,0.15,0.2,0.25])
    # ax[2].set_xlim([0,0.12])
    # ax[2].set_xticks([0.05,0.1])
    ax[2].axes.get_yaxis().set_visible(false)
    ax[1].spines["right"].set_visible(false)
    # colorbar(pl,location="bottom")
    tight_layout(w_pad=-0.3)
    display(fig)
    # savefig("hofstadter_w00..pdf",transparent=true)
    close(fig)

    return Eplus,Eminus
end

Eplus = zeros(Float64,length(ϕ),3)
Eminus = zeros(Float64,length(ϕ),3)
for iν in 1:3   
    ν = [0,-2,2][iν]
    Eplus[:,iν], Eminus[:,iν] = plot_strongcoupling_dispersion_dos(ν)
end

fig = figure(figsize=(3.5,3))
p1 = 0
p2 = 0
for iν in 1:3   
    val = 1 - 0.2
    # ,c=[0;0;val]
    # ,c=[val;0;0]
    p1 , = plot(ϕ,Eplus[:,iν],".")
    p2 , = plot(ϕ,Eminus[:,iν],".")
end

# legend([p1,p2],["+","-"])
ylabel("E")
xlabel(L"ϕ/ϕ_0")
xlim([0,0.29])
tight_layout()
savefig("magnetic_moment.pdf",transparent="true")
display(fig)
close(fig)

##
function semiclassical_quantization()
    # non interacting
    h = readdlm("StrongCoupling/dtheta$(dθ)_w0$(w0)_w1$(w1)/dtheta$(dθ)_w0$(w0)_w1$(w1)_dos0.txt")[:];
    num_of_energies = length(h)
    hmin = minimum(h)
    hmax = maximum(h)
    @inline function area(x::Float64)
        return sum((sign.(x .- h) .+1)./2) / (num_of_energies) 
    end 

    # use value at p/q = 1/25 to gauge semiclassics
    p = numerator(ϕ[end])
    q = denominator(ϕ[end])
    data = readdlm("StrongCoupling/dtheta$(dθ)_w0$(w0)_w1$(w1)/dtheta$(dθ)_w0$(w0)_w1$(w1)_p$(p)q$(q).txt")
    ϵtmp = sort(data[:,1])
    ϵ0 = (ϵtmp[1]+ϵtmp[2])/2   # global energy offset
    S0 = area(ϵ0)
    
    fig = figure(figsize=(4,4))

    for iϕ in eachindex(ϕ)
        p = numerator(ϕ[iϕ])
        q = denominator(ϕ[iϕ])
        # data = readdlm("StrongCoupling/dtheta$(dθ)_w0$(w0)_w1$(w1)/dtheta$(dθ)_w0$(w0)_w1$(w1)_p$(p)q$(q).txt")
        
        Sn = S0*25/q .+ collect(0:(q-1)) .* p/q   # keep q/2 LLs counting from the bottom 
        ϵLL = zeros(Float64,length(Sn))
        for is in eachindex(Sn)
            ϵLL[is] = fzero(x->area(x)-Sn[is],hmin,hmax)
        end
        plot(ones(length(ϵLL))*p/q,ϵLL,"k.",ms=4)

        data = load("BMCoulomb/")
        data = readdlm("StrongCoupling/office_data/dtheta$(dθ)_w0$(w0)_w1$(w1)_p$(p)q$(q).txt")
        
        ϵ = data[:,1]
        σz = data[:,2]
        scatter(ones(length(ϵ))*p/q,ϵ,c=σz,marker=".",s=10,cmap="Spectral",vmin=-1,vmax=1)
    end
    ylim([0.7,1.8])
    xlim([0,0.26])
    tight_layout() 
    savefig("semiclassics_w00.0.pdf",transparent=true)
    display(fig)

    return nothing
end

semiclassical_quantization()

##

function semiclassical_quantizationv2(w0::Float64)
    w00 = w0 * w1
    h = readdlm("StrongCoupling/dtheta$(dθ)_w0$(w00)_w1$(w1)/dtheta$(dθ)_w0$(w00)_w1$(w1)_dos0.txt")[:];
    h = reshape(h,2,:)
    hminus = h[1,:]
    hplus = h[2,:]
    num_of_energies = length(h) ÷ 2
    
    p = numerator(ϕ[end])
    q = denominator(ϕ[end])
    data = readdlm("finitenu/q$(q)_nu0_w0$(w0)_svec4.txt")
    ϵ0plus = data[1,1]
    ϵ0minus = data[2,1]
    

    @inline function area_plus(x::Float64)
        return sum((sign.(x .- hplus) .+1)./2) / (num_of_energies) 
    end 
    @inline function area_minus(x::Float64)
        return sum((sign.(x .- hminus) .+1)./2) / (num_of_energies) 
    end 

    S0plus = area_plus(ϵ0plus)
    S0minus = area_minus(ϵ0minus)

    fig = figure(figsize=(3,3))
    for iϕ in eachindex(ϕ)
        p = numerator(ϕ[iϕ])
        q = denominator(ϕ[iϕ])
        Sn = S0plus*25/q .+ collect(0:(q÷2)) .* p/q
        ϵLLplus = zeros(Float64,length(Sn))
        for is in eachindex(Sn)
            ϵLLplus[is] = fzero(x->area_plus(x)-Sn[is],minimum(hplus),maximum(hplus))
        end

        Sn = S0minus*25/q .+ collect(0:q÷2) .* p/q
        ϵLLminus = zeros(Float64,length(Sn))
        for is in eachindex(Sn)
            ϵLLminus[is] = fzero(x->area_minus(x)-Sn[is],minimum(hminus),maximum(hminus))
        end
    
        plot(ones(length(ϵLLplus))*p/q,ϵLLplus,".",c="b",ms=4)
        plot(ones(length(ϵLLminus))*p/q,ϵLLminus,".",c="r",ms=4)

        data = readdlm("finitenu/q$(q)_nu0_w0$(w0)_svec4.txt")
        ϵ = data[1:(2q),1]
        σz = data[1:(2q),2]
        scatter(ones(length(ϵ))*p/q,ϵ,c=σz,marker=".",s=10,cmap="Spectral",vmin=-1,vmax=1)
    end
    ylim([0.7,1.8])
    xlim([0,0.15])
    tight_layout() 
    savefig("semiclassics_w00.7.pdf",transparent=true)

    display(fig)
    close(fig)

    return nothing
end

semiclassical_quantizationv2(0.7)