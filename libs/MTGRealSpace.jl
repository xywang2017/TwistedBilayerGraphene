# include("Hoftstadter_mod_v3.jl")

mutable struct MTG
    γ::Int
    k1::Float64
    k2::Float64
    nc::Int
    nvec::Vector{Int}
    p::Int 
    q::Int

    Ψr::Array{ComplexF64,3}  # layer,sublattice, z1 and z2

    MTG() = new()
end

function computeMTGRealSpace(hof::HofstadterVq,params::Params,blk::HBM,basis::HybridWannier)
    A = MTG()
    A.γ = 1 
    A.k1 = 0.0
    A.k2 = 0.0 
    A.nc = 10
    A.nvec = collect((-A.nc):A.nc)
    A.p = hof.p 
    A.q = hof.q 

    lg = blk.lg 
    gq = collect((-(lg-1)÷2):((lg-1)÷2))
    l2 = hof.l2*hof.q
    tmp = reshape(basis.WγLS,4,lg,lg,2,hof.l1,l2)
    k2 = reshape(reshape(hof.k2,:,1) .+ reshape(hof.r,1,:)./q,:)
    U = tmp[:,:,:,A.γ,:,:]

    z1s = collect(-2:0.1:2)
    z2s = collect(-2:0.1:2)
    A.Ψr = zeros(ComplexF64,4*l2,length(z1s),length(z2s))

    for i2 in eachindex(z2s)
        z2 = z2s[i2]
        for i1 in eachindex(z1s)
            z1 = z1s[i1]
            for n in eachindex(A.nvec)
                θ1 = exp.(1im * 2π*(A.k1 .-  hof.k1) * A.nvec[n]) * exp(-1im * π * A.nvec[n] * (A.nvec[n]-1) * A.p/(2A.q) )
                θ2 = exp(1im * 2π * n * A.p/A.q * (z1/2+z2))
                θ3 = exp.(1im * 2π * hof.k1 * z1)
                θ4 = exp.(1im * 2π * k2 * z2)
                θ5 = exp.(1im * 2π * gq * z1)
                θ6 = exp.(1im * 2π * gq * z2)

                θ1 = reshape(θ1,1,1,1,hof.l1,1)
                # θ2 = reshape(θ2,1,1,1,1,1)
                θ3 = reshape(θ3,1,1,1,hof.l1,1)
                θ4 = reshape(θ4,1,1,1,1,l2)
                θ5 = reshape(θ5,1,lg,1,1,1)
                θ6 = reshape(θ6,1,1,lg,1,1) 
                θ = θ1 .* θ2 .* θ3 .* θ4 .* θ5 .* θ6
                A.Ψr[:,i1,i2] .+= reshape(sum(θ .* U,dims=(2,3,4)),:) ./ sqrt(hof.l1)
            end
        end
    end
    return A
end

function plotMTGRealSpace(A::MTG,params::Params)
    fig,ax = subplots(2,2,figsize=(8,8))
    weight = abs.(A.Ψr).^2 
    weight = reshape(weight,4,:,size(weight,2),size(weight,3))
    z1s = collect(-2:0.1:2)
    z2s = collect(-2:0.1:2)
    zmesh =  ( reshape(z1s,:,1)*params.a1  .+ reshape(z2s,1,:)*params.a2 )./abs(params.a1)
    
    zgrid = ( reshape(-2:2,:,1)*params.a1  .+ reshape(-2:2,1,:)*params.a2 )./abs(params.a1)
    maxweight = maximum(weight)
    cnt = 1
    for r in 1:2
        for c in 1:2
            pl = ax[r,c].contourf(real(zmesh),imag(zmesh),weight[cnt,2,:,:]/maxweight)
            ax[r,c].axis("equal")
            colorbar(pl,ax=ax[r,c])
            cnt = cnt + 1
            ax[r,c].plot(real(zgrid),imag(zgrid),"k.")
        end
    end 
    
    display(fig)
    # savefig("mtg_r/mtgr_k1$(A.k1)_k2$(A.k2).pdf")
    close(fig)
end