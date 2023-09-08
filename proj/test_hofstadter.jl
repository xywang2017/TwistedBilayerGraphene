using PyPlot
using Printf
using DelimitedFiles
using JLD
fpath = pwd()
include(joinpath(fpath,"libs/Hofstadter_mod_v3.jl"))

##
params = Params(dθ = 1.38π/180)
initParamsWithStrain(params)
q = 64
data = Dict()
ps = collect(1:(q÷2))
for ip in eachindex(ps)
    @printf("p/q=%d/%d\n",ps[ip],q)
    hof,blk,basis = initHofstadterHoppingElements(q,ps[ip],params);
    data["$ip"] = hof.spectrum
end
##
save("Q$(q)_n0_9_results.jld","data",data)


##
fig = figure(figsize=(4,3))
colors = ["b","b","b","b","b"]
qs = [32,64]
for iq in eachindex(qs)
    q = qs[iq]
    data = load("Q$(q)_n0_9_results.jld","data")
    ps = collect(1:(q÷2))
    for ip in eachindex(ps)
        ϵ = data["$ip"]
        plot(ones(size(ϵ))*ps[ip]/q,1.5*ϵ,".",c=colors[iq],ms=1.5,markerfacecolor="b",markeredgecolor="none")
    end
end
ylabel(L"ϵ/v_Fk_θ")
xlabel(L"ϕ/ϕ_0")
xlim([0,0.4])
# axhline(0)
tight_layout()
savefig("energy_flux.pdf")
display(fig)
close(fig)

##
# Wannier plot
# function wannier_plot()
energies = []
ϕ = []
for q in qs 
    # data = load(joinpath(fpath,"Angle1.83/Sep 22/Q$(q)_n0_9_results.jld"),"data")
    data = load("Q$(q)_n0_9_results.jld","data")
    tmp = zeros(Float64,2q,q÷2)
    for ip in 1:(q÷2)
        tmp[:,ip] = data["$ip"]
    end
    push!(energies,tmp)
    tmp = collect(1:(q÷2)) ./ q 
    ϕ = [ϕ;tmp]
end


γ = 0.001
ϵ = range(-0.35,0.35,length=1000)
ρ = zeros(Float64,length(ϕ),length(ϵ))
ν= zeros(Float64,length(ϕ),length(ϵ))
perm = sortperm(ϕ)
ϕ = repeat(ϕ[perm],1,length(ϵ))

# filling 
cnt = 0
for iq in eachindex(qs)
    tmp1 = energies[iq]
    for ip in 1:size(tmp1,2)
        for iϵ in eachindex(ϵ)
            ν[ip+cnt,iϵ] =  1/π * sum( atan.( (ϵ[iϵ].-tmp1[:,ip])./γ ) ) /length(tmp1[:,ip]) *2 #*ϕ[ip,iϵ] 
            ρ[ip+cnt,iϵ] =  1/π * sum(γ./((tmp1[:,ip] .- ϵ[iϵ]).^2 .+γ^2)) /length(tmp1[:,ip]) *2 # *ϕ[ip,iϵ] 
        end
    end
    cnt = cnt + size(tmp1,2)
end

ν .= ν[perm,:]
ρ .= ρ[perm,:]

fig = figure(figsize=(4,3))
for ip in 1:size(ϕ,1)
    plot(ν[ip,:],ϕ[ip,:],".",ms=1.5,markerfacecolor="b",markeredgecolor="none")
end
# ylim([0,0.5])
xlim([-1,1])
ylabel(L"ϕ/ϕ0")
xlabel(L"ν/ν0")
tight_layout()
display(fig)
savefig("wannier_plot.pdf")
close(fig)
# end