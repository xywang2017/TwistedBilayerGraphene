using PyPlot
using JLD
fpath = pwd()
include(joinpath(fpath,"libs/HofstadterVq_modv3_old.jl"))

# dθ = parse(Float64,ARGS[1])
dθ = 1.38
println(dθ)
w1=96.056
w0=0.7w1
params = Params(dθ=dθ*π/180,w0=w0,w1=w1,ϵ=0.0)  #chiral limit
q = 18
p = 1

# non and interaction part - Vq 
hof_Vq = HofstadterVq()
hof_BM = initHofstadterVq(hof_Vq,params,p=p,q=q,lk=18);

save("BMCoulomb/B0_dtheta$(dθ)_w00.7_energetics.jld",
                        "BM",reshape(hof_BM,2,2,:),"Vq",hof_Vq.H0)


save("BMCoulomb/B0_dtheta$(ARGS[1])_w00.7_energetics.jld",
                        "BM",reshape(hof_BM,2,2,:),"Vq",hof_Vq.H0)


## Plot
function plot_energy_cut_B0(iν::Int,w0::Float64,dθ::Float64,cntr::Int)
    ϵr_list = [2;3;4;5;6;7;8;9;10;15;20;25;30;35;40;45;50;60;70;80;90;100]
    # only works for q =18, l1 = 36
    params = Params(dθ=dθ*π/180,w0=w0,w1=w1,ϵ=0.0)  #chiral limit
    k1 = collect(0:35) ./ 36
    iΓ = [1;1]
    iM = [19;1]
    iK = [25;13]
    cutKΓ = [iK - [2i;i] for i in 0:12]
    cutΓM = [iΓ + [i;0] for i in 1:18]
    cutMK = [iM + [i;2i] for i in 1:6]
    
    # coulomb energy 
    ϵ0 = 8.854e-12 
    ϵr =reverse(ϵr_list)[cntr]
    charge = 1.6e-19 
    Lm = abs(params.a1) * 2.46e-10
    V0 = charge/(4π * ϵ0 * ϵr * Lm) * 1000 # meV
    
    kvec = reshape(k1,:,1) * params.g1 .+ reshape(k1,1,:) * params.g2
    # plot energy cut along Γ -> M 
    l1 = size(kvec,1)
    data = load("BMCoulomb/B0_dtheta$(dθ)_w00.7_energetics.jld")
    hBM = data["BM"]
    hVq = data["Vq"]
    ϵ = zeros(Float64,2,l1^2)
    for ik in 1:size(hBM,3)
        ϵ[:,ik] = eigvals(Hermitian(hBM[:,:,ik]+V0*hVq[:,:,ik,iν]))
    end
    energies = reshape(ϵ,2,l1,l1)
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
    
    fig = figure(figsize=(3,2.5))
    plot(1:size(ϵcut,2),ϵcut[1,:],"k-")
    plot(1:size(ϵcut,2),ϵcut[2,:],"k-")
    # ylim([0.7,1.8])
    # yticks([])
    ylabel("E (meV)")
    xticks([1,length(cutKΓ),length(cutKΓ)+length(cutΓM),size(ϵcut,2)],["K","Γ","M","K"])
    # title(L"$θ=%$(dθ)^\circ$")
    title(L"$ϵ_r=%$(ϵr)$")
    # axis("off")
    tight_layout()
    if cntr<10
        savefig("benfeldman/B0_00$cntr.png")
    else
        savefig("benfeldman/B0_0$cntr.png")
    end
    close(fig)
end

for i in 1:22
    plot_energy_cut_B0(1,0.7,1.38,i)
end

###

dθs = [0.95,0.97,1,1.03,1.05,1.1,1.15,1.2,1.25,1.3,1.35]
energy_bounds = zeros(Float64,2,length(dθs))
cnt = 1
for dθ in dθs
    w1=96.056
    w0=0.7w1
    params = Params(dθ=dθ*π/180,w0=w0,w1=w1,ϵ=0.0)  #chiral limit
    # coulomb energy 
    ϵ0 = 8.854e-12 
    ϵr = 20.0 
    charge = 1.6e-19 
    Lm = abs(params.a1) * 2.46e-10
    V0 = charge/(4π * ϵ0 * ϵr * Lm) * 1000 # meV
    println(V0)
    k1 = collect(0:35) ./ 36
    kvec = reshape(k1,:,1) * params.g1 .+ reshape(k1,1,:) * params.g2
    # plot energy cut along Γ -> M 
    l1 = size(kvec,1)
    data = load("BMCoulomb/B0_dtheta$(dθ)_w00.7_energetics.jld")
    hBM = data["BM"]
    hVq = data["Vq"]
    ϵ = zeros(Float64,2,l1^2)
    for ik in 1:size(hBM,3)
        ϵ[:,ik] = eigvals(Hermitian(hBM[:,:,ik]+V0*hVq[:,:,ik,1]))
    end
    energy_bounds[:,cnt] = [minimum(ϵ); maximum(ϵ)]
    cnt = cnt + 1
end
save("benfeldman/energy_bounds.jld","energy_bounds",energy_bounds)