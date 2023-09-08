using PyPlot
using JLD
fpath = pwd()
include(joinpath(fpath,"libs/Hofstadter_mod.jl"))
include(joinpath(fpath,"libs/HofstadterVq_modv3.jl"))

dθ = 1.06
println(dθ)
w1=96.056
w0=0.7w1
params = Params(dθ=dθ*π/180,w0=w0,w1=w1,ϵ=0.0)  #chiral limit
q = 18
p = 1

# non-interacting part - BM 
hof = initHofstadterHoppingElements(params;q=q,p=p)
ϵ0 = 2π * params.vf / abs(params.a1)
n0dim = size(hof.M,3)
H = reshape(hof.H,4hof.l2,4hof.l2)
H = (H + H') * ϵ0

println("End of non-interacting calculation")
# interaction part - Vq 
hof_Vq = HofstadterVq()
initHofstadterVq(hof_Vq,params,p=p,q=q,lk=10);
println("End of interacting calculation")
save("benfeldman/dtheta$(dθ)_p$(q)q1_w00.7_energetics.jld","BM",H,
                        "Vq",hof_Vq.H[:,:,1,:],"Σz",hof_Vq.Σz[:,:,1],
                        "Uort",hof_Vq.Uort[:,:,1])


## load data and plot at ν = -4
function plot_field(cntr::Int)
    ϵr_list = [1;2;5;10;20;30;40;50;60;70;80;90;100;200;500;1000;2000]
    # coulomb energy 
    qs = collect(8:17)
    # energy_bounds = load("BMCoulomb/energy_bounds.jld","energy_bounds")
    fig = figure(figsize=(4,3))
    pl = 0
    cnt = 1
    dθ = 1.38
    ϵ0 = 8.854e-12 
    ϵr = ϵr_list[cntr]
    charge = 1.6e-19 
    params = Params(dθ=dθ*π/180,w0=w0,w1=w1,ϵ=0.0)
    Lm = abs(params.a1) * 2.46e-10
    V0 = charge/(4π * ϵ0 * ϵr * Lm) * 1000 # meV
    for q in qs
        data = load("benfeldman/dtheta$(dθ)_p$(q)q1_w00.7_energetics.jld")
        H_Vq = data["Vq"][:,:,1] * V0 # 10meV coulomb energy
        Σz = data["Σz"]
        Uort = data["Uort"]
        H_BM = Uort' * data["BM"] * Uort
        F = eigen(Hermitian(H_Vq+H_BM))
        pl = scatter(ones(size(H_Vq,1))*p/q,F.values,c=real(diag(F.vectors'*Σz*F.vectors)),s=8,marker=".",cmap="coolwarm_r",vmin=-1,vmax=1)
        # plot(energy_bounds[:,cnt],[1;1]*dθ,"k|")
        cnt = cnt +1 
    end
    xlabel(L"ϕ/ϕ_0")
    ylabel("E (meV)")
    axhline(0,ls=":",lw=1,c="grey")
    yticks([0])
    title(L"$ϵ_r=%$(ϵr)$")
    colorbar(pl)
    tight_layout()
    # display(fig)
    if cntr<10
        savefig("benfeldman/00$cntr.png")
    else
        savefig("benfeldman/0$cntr.png")
    end
    close(fig)
end

for i in 1:17
 plot_field(i)
end



## load data and plot at ν = -4
function wannier_density(ϵsource::Vector{Float64})
    ϵmax = maximum(abs.(ϵsource))
    γ = 0.002 * ϵmax
    # ϵ = range(-1.02ϵmax,1.02ϵmax,length=500)
    ϵ = ϵsource
    ρϵ = zeros(Float64,length(ϵ))
    for i in eachindex(ρϵ)
        # ρϵ[i] = sum(1 ./(γ^2 .+ (ϵ[i] .- ϵsource).^2)) /π
        # ρϵ[i] = sum(atan.( (ϵ[i] .- ϵsource)./γ ) ) /π /length(ϵ)
        ρϵ[i] = sum(sign.( (ϵ[i] .- ϵsource .+ γ) ) ) /2 /length(ϵ)
    end
    return ρϵ
end

function plot_field_wannier(cntr::Int)
    ϵr_list = [1;2;5;10;20;30;40;50;60;70;80;90;100;200;500;1000;2000]
    # coulomb energy 
    qs = collect(8:17)
    # energy_bounds = load("BMCoulomb/energy_bounds.jld","energy_bounds")
    fig = figure(figsize=(3,4))

    dθ = 1.38
    ϵ0 = 8.854e-12 
    ϵr = ϵr_list[cntr]
    charge = 1.6e-19 
    params = Params(dθ=dθ*π/180,w0=w0,w1=w1,ϵ=0.0)
    Lm = abs(params.a1) * 2.46e-10
    V0 = charge/(4π * ϵ0 * ϵr * Lm) * 1000 # meV
    for iq in eachindex(qs)
        q = qs[iq]
        data = load("benfeldman/dtheta$(dθ)_p$(q)q1_w00.7_energetics.jld")
        H_Vq = data["Vq"][:,:,1] * V0
        Uort = data["Uort"]
        H_BM = Uort' * data["BM"] * Uort
        F = eigen(Hermitian(H_Vq+H_BM))
        ρϵ = wannier_density(F.values)
        plot(ones(size(H_Vq,1))*p/q,ρϵ*8,"b.",markersize=2)
    end

    for nu in -4:4
        plot([0,0.25],[0,nu],":",c="gray",lw=1)
    end
    for nu in 0:4
        plot([0,0.25],[-4,nu-4],":",c="gray",lw=1)
    end
    for nu in 0:4
        plot([0,0.25],[4,4-nu],":",c="gray",lw=1)
    end
    xlabel(L"ϕ/ϕ_0")
    ylabel(L"n/n_0")
    xlim([0,0.5])
    ylim([-4,4])
    title(L"$ϵ_r=%$(ϵr)$")
    tight_layout()
    # display(fig)
    if cntr<10
        savefig("benfeldman/wannier_00$cntr.png")
    else
        savefig("benfeldman/wannier_0$cntr.png")
    end
    close(fig)
end

for i in 1:17
 plot_field_wannier(i)
end