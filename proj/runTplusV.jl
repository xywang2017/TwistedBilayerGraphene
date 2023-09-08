# using PyPlot
using JLD
fpath = pwd()
include(joinpath(fpath,"libs/Hofstadter_mod.jl"))
include(joinpath(fpath,"libs/HofstadterVq_modv3.jl"))

# dθ = parse(Float64,ARGS[1])
dθ = 1.38
println(dθ)
w1=96.056
w0=0.7w1
params = Params(dθ=dθ*π/180,w0=w0,w1=w1,ϵ=0.0)  #chiral limit
q = parse(Int,ARGS[1])
p = 1

# non-interacting part - BM 
hof = initHofstadterHoppingElements(params;q=q,p=p);
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
# function plot_field()
#     # coulomb energy 
#     dθs = [0.95,0.97,1,1.03,1.05,1.1,1.15,1.2,1.25,1.3,1.35]
#     energy_bounds = load("BMCoulomb/energy_bounds.jld","energy_bounds")
#     fig = figure(figsize=(4,3))
#     pl = 0
#     cnt = 1
#     for dθ in dθs
#         ϵ0 = 8.854e-12 
#         ϵr = 20.0 
#         charge = 1.6e-19 
#         params = Params(dθ=dθ*π/180,w0=w0,w1=w1,ϵ=0.0)
#         Lm = abs(params.a1) * 2.46e-10
#         V0 = charge/(4π * ϵ0 * ϵr * Lm) * 1000 # meV
        
#         data = load("BMCoulomb/dtheta$(dθ)_p12q1_w00.7_energetics.jld")
#         H_Vq = data["Vq"][:,:,1] * V0 # 10meV coulomb energy
#         Σz = data["Σz"]
#         Uort = data["Uort"]
#         H_BM = Uort' * data["BM"] * Uort

#         F = eigen(Hermitian(H_Vq+H_BM))
#         pl = scatter(F.values,ones(size(H_Vq,1))*dθ,c=real(diag(F.vectors'*Σz*F.vectors)),s=4,cmap="coolwarm_r",vmin=-1,vmax=1)
#         plot(energy_bounds[:,cnt],[1;1]*dθ,"k|")
#         cnt = cnt +1 
#     end
#     xlabel("E")
#     ylabel("θ")
#     colorbar(pl)
#     tight_layout()
#     display(fig)
#     savefig("BMCoulomb/nu_-4_exact_solutions.pdf",transparent=true)
#     close(fig)
# end

# plot_field()