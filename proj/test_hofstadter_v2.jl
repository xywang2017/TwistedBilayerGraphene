using PyPlot
using Printf
using DelimitedFiles
fpath = "/Users/xiaoyu/Code/TwistedBilayerGraphene"
include(joinpath(fpath,"libs/Hoftstadter_mod_v2.jl"))

##
# qs = [23,37,53,64]
qs = [32]
@time begin
q = qs[1]
params = Params(dθ = 1.83π/180)

# for q in qs
    data = Dict()
    ϵ0 = params.vf * params.kb
    @printf("q=%d\n",q)
    hof,blk,basis = initHofstadterHoppingElements(q,params);
    data["1"] = hof.Spectrum./ϵ0;
    writedlm("Angle1.83v2/spectrumdata_q$(q).txt",data["1"][:])
    # data["$iq"] = readdlm("flux_test/spectrumdata_q$(q[iq]).txt")
# end
end
##
# Wannier plot
# function wannier_plot()
    energies = []
    ϕ = []
    for q in qs 
        tmp = reshape(readdlm("Angle1.83v2/spectrumdata_q$(q).txt"),(2q)*10,:)
        push!(energies,tmp)
        tmp = collect(0:q) ./ q 
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

    fig = figure(figsize=(4,5))
    pl =contourf(ν,ϕ,ρ,cmap="Blues_r")
    # pl = pcolormesh(ν,ϕ,ρ,cmap="Blues_r",shading="auto")
    colorbar(pl)
    # ylim([0,0.5])
    xlim([-1,1])
    ylabel(L"ϕ/ϕ0")
    xlabel(L"ν/ν0")
    tight_layout()
    # display(fig)
    savefig("Angle1.83v2/angle1.83_wannier.pdf")
    close(fig)
# end

##
fig = figure(figsize=(4,5))
for ip in 1:size(ϕ,1)
    plot(ν[ip,:],ϕ[ip,:],"b+",ms=0.1)
end
# ylim([0,0.5])
xlim([-1,1])
ylabel(L"ϕ/ϕ0")
xlabel(L"ν/ν0")
tight_layout()
# display(fig)
savefig("Angle1.83v2/angle1.83_nu_phi.pdf")
close(fig)

##
#Energy Plotx
# function energy_plot()
dimR = 2
    fig = figure(figsize=(4,5))
    # for q in qs
        tmp = reshape(readdlm("Angle1.83v2/spectrumdata_q$(q).txt"),dimR*2q,hof.l1,:)
        p = collect(0:q)
        for ip in eachindex(p)
            ϵ = reshape(tmp[:,:,ip],:)
            ϕ = ones(Float64,size(ϵ)) /q * p[ip]
            plot(ϵ,ϕ,"g+",ms=0.8)
        end
    # end
    # xlim([-0.4,0.4])
    ylabel(L"ϕ/ϕ_0")
    xlabel(L"ϵ/v_Fk_θ")
    tight_layout()
    display(fig)
    # savefig("Angle1.83v2/angle1.83_energy.png")
    close(fig)
# end


##
k1s = eachindex(hof.k1)
dimR = 2

hof.Ham = zeros(ComplexF64,dimR*2*hof.l2,dimR*2*hof.l2,length(k1s),length(hof.p))
    hof.Vec = zeros(ComplexF64,dimR*2*hof.l2,dimR*2*hof.l2,length(k1s),length(hof.p))  # orthonormalized eigenvectors
    hof.D = zeros(Float64,dimR*2*hof.l2,length(k1s),length(hof.p))
    Hmag = hof.Term1-hof.Term2
    for ip in eachindex(hof.p)
        for m in eachindex(k1s)
            tmpU = reshape(view(hof.M,:,:,:,:,m,ip),dimR*2*hof.l2,dimR*2*hof.l2)
            tmpH = reshape(view(Hmag,:,:,:,:,m,ip),dimR*2*hof.l2,dimR*2*hof.l2)
            tmpH = (tmpH + tmpH') / 2
            F = eigen(Hermitian(tmpU))
            F.values[F.values .< 0.1] .= Inf
            # F.values[1:(end-2*hof.l2)] .= Inf
            hof.D[:,m,ip] = F.values
            hof.Vec[:,:,m,ip] = F.vectors'
            vals = 1 ./ sqrt.(F.values)
            U = Diagonal(vals) * hof.Vec[:,:,m,ip] 
            hof.Ham[:,:,m,ip] = U * tmpH * U'
        end
    end

    # ------------------------ Spectrum ------------------------ #

    # hof.Spectrum = zeros(Float64,2*hof.l2,length(k1s),length(hof.p))
    hof.Spectrum = zeros(Float64,dimR*2*hof.l2,length(k1s),length(hof.p))
    for ip in 1:size(hof.Spectrum,3)
        for m in 1:size(hof.Spectrum,2)
            hof.Spectrum[:,m,ip] = eigvals(Hermitian(hof.Ham[:,:,m,ip]))
        end
    end

    data["1"] = hof.Spectrum./ϵ0;
    writedlm("Angle1.83v2/spectrumdata_q$(q).txt",data["1"][:])