include("hybridWannier_mod.jl")
using LinearAlgebra
using ClassicalOrthogonalPolynomials

mutable struct HofstadterVq 
    # This module contains methods for calculating Hofstadter at strong coupling
    # implement calculation of O_{j1k,j2p}(g_q) instead, alternative scheme
    p::Int
    q::Int 

    lk::Int  # 2nq q
    nq::Int 
    l1::Int
    l2::Int 
    k1::Vector{Float64} # grid in the range of [0,1)
    k2::Vector{Float64} # grid in the range of [0,1/q) 
    kvec::Matrix{ComplexF64} # grid of k points in the magnetic Brillouin zone
    r::Vector{Int} # 0...q-1
    g1::ComplexF64 
    g2::ComplexF64

    nvec::Vector{Int} # nvec = [0,1]
    svec::Vector{Int} # svec = -nc:nc

    δg::Array{Complex{Int},2} # a g1 + b g2; stores a+ib 
    M::Array{ComplexF64,4}  # (γ1, k), (γ2, p), δg1, δg2 
    Mz::Array{ComplexF64,4} # polarization σz
    O::Array{ComplexF64,3}  # (nγr) x (nγr) x k1k2
    Oz::Array{ComplexF64,3} # (nγr) x (nγr) x k1k2
    # coefficients relating trial MTG and orthonormalized MTG eigenstates
    Uort::Array{ComplexF64,3}  # (nγr) x (j) x k1k2 

    # comparison to Landau Level based approach
    Λ::Array{ComplexF64,3}  # (nγr) x(n layer LL) x k1k2
    nγ::Int # = 2 sublattice
    nLL::Int  # number of harmonic oscillator states
    nH::Int  # = nγ nLL - 1, number of Landau levels per layer

    
    # Strong coupling energetics
    gq::Array{Complex{Int},2}  # m g1 + n g2/q; store m + i n
    OΨ::Array{ComplexF64,4}    # (j1 k1,k2) x (j2 p1 p2) for a given m,n 
    Ot::Array{ComplexF64,4}    # (n1 r1 γ1 k1,k2) x (n2 r2 γ2 p1 p2) for a given m,n 
    
    #
    Σz::Array{ComplexF64,3} # (γr) x (γr) x k1 k2 # eigenvalues of σz μ_0
    ν::Array{Int}  # 0, ± 1, ± 2, all electron side (but particle-hole symmetry relates two sides? care)
    H::Array{ComplexF64,4} # (γr) x (γr) x k1 k2 x ν-- for every k in magnetic strip, γr x γr matrix

    #
    H0::Array{ComplexF64,4} # zeros field Hamiltonian: 2x2xgrid
    dos::Array{Float64,2} # density of states
 
    HofstadterVq() = new()

end

function initHofstadterVq(A::HofstadterVq, params::Params;p::Int=1,q::Int=5,lk::Int=4)
    
    # A = HofstadterVq()
    ϕ = p//q 

    A.p = numerator(ϕ)
    A.q = denominator(ϕ)
    A.g1 = params.g1 
    A.g2 = params.g2
    A.nvec = collect(0:1)
    A.svec = collect(-4:4)
    
    gc = 3
    A.δg = reshape((-gc):(gc),:,1) .+ 1im * reshape((-gc):(gc),1,:)
    A.gq = reshape(-gc:gc,:,1) .+ 1im * reshape((-gc*A.q):(gc*A.q),1,:)

    # gc0 = 3
    # A.δg = reshape((-gc0):(gc0),:,1) .+ 1im * reshape((-gc0):(gc0),1,:)
    # ------ lattice initialization ------ #
    if (A.q>=lk)
        A.nq = 1
        A.lk = 2*A.q * A.nq
    else
        A.nq = (lk-1) ÷ A.q + 1
        A.lk = 2*A.q * A.nq
    end

    println("q= ",A.q," nq= ",A.nq)
    A.l1 = 2 * A.q * A.nq 
    A.k1 = collect(0:(A.l1-1)) ./ (A.l1)
    A.l2 = 2*A.nq
    A.k2 = collect(0:(A.l2-1)) ./ (A.l2*A.q)
    A.r = collect(0:(A.q-1))

    # A.k1 .+= 1/(2A.l1)
    # A.k2 .+= 1/(2A.l1)

    k2mbz = reshape( reshape(A.k2,:,1) .+ reshape(A.r ./A.q,1,:) , : )
    A.kvec = reshape(A.k1,:,1) * A.g1 .+ reshape(k2mbz,1,:) * A.g2

    A.ν = Int[-4,0,4]

    
    # --------- Bloch part of the story ------- # 
    blk = ConstructHBM_Hoftstadter(A,params)
    basis = ConstructHybridWannier_Hofstadter(A,blk)
    computeM(A, blk,basis)
    
    # blk = 0 
    # basis = 0
    # GC.gc()

    # --------- Matrix elements at strong coupling B=0 --------- #
    computeSingleParticleSpectrumB0(A)
    # computeSingleParticleSpectrumB0_hole(A)
    # computeDensityofStatesB0(A)

    # --------- Overlap matrix and orthonormalization ---------- # 
    # O, Oz
    # computeOverlapMatrix(A)

    # Uort, Σz 
    # computeOrtNormMTG(A)

    # ---------- Landau level comparison ----------- #
    # computeLLOverlapSingleK(A,basis,params)

    # --------- Matrix elements at strong coupling ------- # 
    # computeSingleParticleSpectrum(A)

    return basis.Hγk
end

@inline function Coulomb(q::ComplexF64)
    return abs(q)>1e-5 ? 2π*tanh(0.5*abs(q)*abs(params.a1))/abs(q) : 0.0
    # return abs(q)>1e-5 ? 4π/abs(q) : 0.0
end


function computeM(A::HofstadterVq,blk::HBM,basis::HybridWannier)
    # Given hybrid wannier states from blk.Uγk[4lg^2,2,lk1,lk2], compute overlap matrix 
    # M(γ1k,γ2p,q) = U_g(γ1k)^† U_g+δg(γ2p)
    Uγk = reshape(basis.WγLS,4blk.lg^2,2A.l1*A.l2*A.q)
    A.M = zeros(ComplexF64,2A.l1*A.l2*A.q,2A.l1*A.l2*A.q,size(A.δg,1),size(A.δg,2))
    A.Mz = zeros(ComplexF64,2A.l1*A.l2*A.q,2A.l1*A.l2*A.q,size(A.δg,1),size(A.δg,2))
    Oσz = kron(Array{ComplexF64}(I,blk.lg^2,blk.lg^2),kron(ComplexF64[1 0;0 1],ComplexF64[1 0;0 -1]))
    for ig in CartesianIndices(A.δg)
        Uγp = reshape(Uγk,4,blk.lg,blk.lg,:)
        Uγp = circshift(Uγp,(0,-real(A.δg[ig]),-imag(A.δg[ig]),0))
        Uγp = reshape(Uγp,4blk.lg^2,:)
        A.M[:,:,ig[1],ig[2]] = Uγk' * Uγp
        A.Mz[:,:,ig[1],ig[2]] = Uγk' * Oσz * Uγp
    end
    return nothing
end


function computeSingleParticleSpectrumB0(A::HofstadterVq)
    # kvec in Moire Brillouin zone
    kvec = A.kvec
    A.H0 = zeros(ComplexF64,2,2,A.l1*A.l2*A.q,length(A.ν))
    ρkp = reshape(A.M,2,A.l1*A.l2*A.q,2,A.l1*A.l2*A.q,size(A.δg,1),size(A.δg,2))
    ρgq = zeros(ComplexF64,size(A.δg,1),size(A.δg,2))
    for ig in CartesianIndices(A.δg)
        gq = real(A.δg[ig]) * A.g1 + imag(A.δg[ig]) * A.g2
        Vq = reshape( Coulomb.( reshape(kvec,:,1) .- reshape(kvec,1,:) .- gq), (1,1,A.l1*A.l2*A.q,1,A.l1*A.l2*A.q) )
        Λkp = view(ρkp,:,:,:,:,ig[1],ig[2])
        Λ1 = reshape(Λkp,2,1,A.l1*A.l2*A.q,2,A.l1*A.l2*A.q)
        Λ2 = reshape(Λkp,1,2,A.l1*A.l2*A.q,2,A.l1*A.l2*A.q)
        A.H0 .+= reshape( sum(Vq .* Λ1 .* conj.(Λ2), dims=(4,5)), (2, 2, A.l1*A.l2*A.q, 1))
        
        ρkp_gq = 2* tr(reshape(Λkp,2*A.l1*A.l2*A.q,2*A.l1*A.l2*A.q))

        ρgq[ig] = tr(reshape(Λkp,2*A.l1*A.l2*A.q,2*A.l1*A.l2*A.q)) / (2A.q*A.l1*A.l2)

        for ik in 1:(A.l1*A.l2*A.q)
            tmp = ( Coulomb(gq) * ρkp_gq ) * Λ1[:,1,ik,:,ik]'
            for iν in eachindex(A.ν)
                A.H0[:,:,ik,iν] .+= (A.ν[iν]/2) * tmp 
            end
        end
    end

    # writedlm("VqLL/B0density.txt",abs.(ρgq))
    
    Lm = (4π)/(sqrt(3)*abs(A.g1))
    V0 = 1/Lm
    # V0 = 1.0
    # ξ = 0.744947*abs(params.a1)
    # Uξ = 26.0
    A.H0 .*= (1/(sqrt(3)*Lm^2*A.l1^2) / V0) 
    return nothing
end


function computeSingleParticleSpectrumB0_hole(A::HofstadterVq)
    # compute spectrum for holes
    # kvec in Moire Brillouin zone
    kvec = A.kvec
    A.H0 = zeros(ComplexF64,2,2,A.l1*A.l2*A.q,length(A.ν))
    ρkp = reshape(A.M,2,A.l1*A.l2*A.q,2,A.l1*A.l2*A.q,size(A.δg,1),size(A.δg,2))

    for ig in CartesianIndices(A.δg)
        gq = real(A.δg[ig]) * A.g1 + imag(A.δg[ig]) * A.g2
        Vq = reshape( Coulomb.( reshape(kvec,:,1) .- reshape(kvec,1,:) .- gq), (1,A.l1*A.l2*A.q,1,1,A.l1*A.l2*A.q) )
        Λkp = view(ρkp,:,:,:,:,ig[1],ig[2])
        Λ1 = reshape(Λkp,2,A.l1*A.l2*A.q,2,1,A.l1*A.l2*A.q)
        Λ2 = reshape(Λkp,2,A.l1*A.l2*A.q,1,2,A.l1*A.l2*A.q)
        A.H0 .+= reshape( sum(Vq .* Λ1 .* conj.(Λ2), dims=(1,2)), (2, 2, A.l1*A.l2*A.q, 1))
        
        ρkp_gq = 2* tr(reshape(Λkp,2*A.l1*A.l2*A.q,2*A.l1*A.l2*A.q))

        for ik in 1:(A.l1*A.l2*A.q)
            tmp = ( Coulomb(gq) * ρkp_gq ) * conj(Λkp[:,ik,:,ik])
            for iν in eachindex(A.ν)
                A.H0[:,:,ik,iν] .-= (A.ν[iν]/2) * tmp 
            end
        end
    end

    Lm = (4π)/(sqrt(3)*abs(A.g1))
    V0 = 1/Lm
    # V0 = 1.0
    # ξ = 0.744947*abs(params.a1)
    # Uξ = 26.0
    A.H0 .*= (1/(sqrt(3)*Lm^2*A.l1^2) / V0) 
    return nothing
end

function computeDensityofStatesB0v2(h::Vector{Float64};γ::Float64=0.03)
    num_of_energies = length(h)
    hmax = maximum(h)
    hmin = minimum(h)
    ϵ = range(hmin-0.02,hmax+0.02,length=500)
    dos = zeros(Float64,length(ϵ),2)
    dos[:,1] = ϵ
    for iϵ in eachindex(ϵ)
        dos[iϵ,2] = sum(1 ./(γ^2 .+ (h[:] .- ϵ[iϵ]).^2)) * γ/ (π * num_of_energies) 
    end
    dos[ϵ .> hmax,2] .=0 
    dos[ϵ .< hmin,2] .=0
    return dos
end

function computeDensityofStatesB0(A::HofstadterVq)
    h = zeros(Float64,size(A.H0,2),size(A.H0,3))
    for ik in 1:size(A.H0,3)
        h[:,ik] = eigvals(Hermitian(A.H0[:,:,ik]))
    end

    A.dos = zeros(Float64,length(h[:]),1)
    A.dos[:,1] = h[:]

    return nothing
end

function computeOverlapMatrix(A::HofstadterVq)
    ρkp = reshape(A.M,2,A.l1,A.l2,A.q,2,A.l1,A.l2,A.q,size(A.δg,1),size(A.δg,2))
    ρzkp = reshape(A.Mz,2,A.l1,A.l2,A.q,2,A.l1,A.l2,A.q,size(A.δg,1),size(A.δg,2))
    ig0 = (size(A.δg,1) +1 )÷2 # index of δg = (0,0)

    A.O = zeros(ComplexF64,2A.q*length(A.nvec),2A.q*length(A.nvec),A.l1*A.l2)
    A.Oz = zeros(ComplexF64,2A.q*length(A.nvec),2A.q*length(A.nvec),A.l1*A.l2)

    O = reshape(A.O,2,A.q,length(A.nvec),2,A.q,length(A.nvec),A.l1,A.l2)
    Oz = reshape(A.Oz,2,A.q,length(A.nvec),2,A.q,length(A.nvec),A.l1,A.l2)

    for s1 in A.svec
        tmp = eachindex(A.k1) .- s1*A.p*A.nq
        k1c = mod.(tmp.-1,A.l1) .+ 1
        δg1 = (tmp .- k1c) .÷ A.l1 .+ ig0

        tmp = eachindex(A.r) .- s1 *A.p
        r2 = mod.(tmp.-1,A.q) .+ 1 
        δg2 = (tmp .- r2) .÷ A.q .+ ig0
        
        θ0 = @. exp(1im * 2π * A.k1 * s1) * exp(-1im * π * s1*(s1-1) *A.p/(2A.q) )
        for i2 in eachindex(A.k2)
            for ir_r in eachindex(A.r)
                ir_c = r2[ir_r]
                for i1_r in eachindex(A.k1)
                    i1_c = k1c[i1_r]
                    tmp = ρkp[:,i1_r,i2,ir_r,:,i1_c,i2,ir_c,δg1[i1_r],δg2[ir_r]] ./ A.l1
                    tmpz = ρzkp[:,i1_r,i2,ir_r,:,i1_c,i2,ir_c,δg1[i1_r],δg2[ir_r]] ./ A.l1
                    for n2 in eachindex(A.nvec), n1 in eachindex(A.nvec)
                        θ1 = @. exp(1im * 2π * A.k1[i1_r]*A.nvec[n1]) * exp(-1im * 2π * A.k1[i1_c]*(s1+A.nvec[n2])) * θ0
                        O[:,ir_r,n1,:,ir_c,n2,:,i2] += reshape(θ1,1,1,:) .* tmp
                        Oz[:,ir_r,n1,:,ir_c,n2,:,i2] += reshape(θ1,1,1,:) .* tmpz
                    end
                end
            end
        end
    end
    return nothing
end


function computeOrtNormMTG(A::HofstadterVq)
    # |Ψ⟩ = |T⟩ Uort for any k1, k2 in magnetic Brillouin zone
    # ik0 = rand(1:A.l1*A.l2)
    A.Uort = zeros(ComplexF64,2A.q*length(A.nvec),2A.q,A.l1*A.l2)
    A.Σz = zeros(ComplexF64,2A.q,2A.q,A.l1*A.l2)
    for ik in 1:A.l2*A.l1
        O = view(A.O,:,:,ik)
        F = eigen(Hermitian(O))
        A.Uort[:,:,ik] = ( @view F.vectors[:,(end-2A.q+1):end]) * 
                            Diagonal( 1 ./sqrt.( F.values[(end-2A.q+1):end] ) )

        Oz = view(A.Oz,:,:,ik)
        U = view(A.Uort,:,:,ik)
        A.Σz[:,:,ik] = U' * Oz * U

        # if ik == ik0
        #     nrmO = norm(O - O')
        #     if nrmO > 1e-8
        #         println("error with Hermiticity O ",ik, nrmO)
        #     end
        # end
    end
    return nothing
end

function computeSingleParticleSpectrum(A::HofstadterVq)
    # kvec in magnetic strip
    kvec = reshape(A.k1,:,1) * params.g1 .+ reshape(A.k2,1,:) * params.g2
    A.H = zeros(ComplexF64,2A.q,2A.q,A.l1*A.l2,length(A.ν))
    gq = real(A.gq) * A.g1 + imag(A.gq) * A.g2 / A.q
    A.OΨ = zeros(ComplexF64,2A.q,A.l1*A.l2,2A.q,A.l1*A.l2)
    A.Ot = zeros(ComplexF64,2A.q*length(A.nvec),A.l1*A.l2,2A.q*length(A.nvec),A.l1*A.l2)

   
    for ig in CartesianIndices(A.gq)
        # println(real(A.gq[ig])," ",imag(A.gq[ig])÷A.q," ",mod(imag(A.gq[ig]),A.q))
        println(ig[1]," ",ig[2])
        gq = real(A.gq[ig]) * A.g1 + imag(A.gq[ig]) * A.g2/A.q
        Vq = reshape( Coulomb.( reshape(kvec,:,1) .- reshape(kvec,1,:) .- gq), (1,1,A.l1*A.l2,1,A.l1*A.l2) )

        computeCoulombOverlap(A,real(A.gq[ig]),imag(A.gq[ig]))
        OΨ1 = reshape(A.OΨ,2A.q,1,A.l1*A.l2,2A.q,A.l1*A.l2)
        OΨ2 = reshape(A.OΨ,1,2A.q,A.l1*A.l2,2A.q,A.l1*A.l2)
        A.H .+= reshape( sum(Vq .* OΨ1 .* conj.(OΨ2), dims=(4,5)), (2A.q, 2A.q, A.l1*A.l2,1))

        ρkp = reshape(A.OΨ,2A.q*A.l1*A.l2,2A.q*A.l1*A.l2)
        ρkp_gq = tr(ρkp)
        if mod(imag(A.gq[ig]), A.q) ==0 # rhokp_gq should be zero
            println(ρkp_gq)
        end
        
        for ik in 1:(A.l1*A.l2)
            tmp = Vq[1,1,1,1,1] * (ρkp_gq * conj.(OΨ1[:,1,ik,:,ik]) )
            for iν in eachindex(A.ν)
                A.H[:,:,ik,iν] .+= (A.ν[iν]/2) * tmp
            end
        end
        
    end

    Lm = (4π)/(sqrt(3)*abs(A.g1))
    V0 = 1/Lm
    A.H .*= (1/(sqrt(3)*Lm^2*A.l1^2) / V0)
    return nothing
end

function computeCoulombOverlap(A::HofstadterVq,m::Int,n::Int)
    # overlap between trial MTG eigenstates
    ρkp = reshape( A.M, (2,A.l1,A.l2,A.q,2,A.l1,A.l2,A.q,size(A.δg,1),size(A.δg,2)) )
    ig0 = (size(A.δg,1) +1 )÷2 # index of δg = (0,0)

    A.Ot .= 0.0 + 0.0im
    # minimal storage due to δ-constraints
    Ot = reshape(A.Ot,2,A.q,length(A.nvec),A.l1,A.l2,2,A.q,length(A.nvec),A.l1,A.l2)  # 39MB for q=10
    for ip1 in eachindex(A.k1), ik1 in eachindex(A.k1)
        for s1 in A.svec
            tmp = eachindex(A.k1) .- s1*A.p*A.nq .- (ik1-ip1-m*A.l1)
            k1c = mod.(tmp.-1,A.l1) .+ 1
            δg1 = (tmp .- k1c) .÷ A.l1 .+ ig0

            tmp = eachindex(A.r) .- s1 *A.p .+ n
            r2 = mod.(tmp.-1,A.q) .+ 1 
            δg2 = (tmp .- r2) .÷ A.q .+ ig0

            θ0 = exp(1im * 2π * A.k1[ip1] * s1) * exp(-1im * π * s1*(s1-1) *A.p/(2A.q) )
            for ir_r in eachindex(A.r)
                if δg2[ir_r] in 1:size(A.δg,2)
                    ir_c = r2[ir_r]
                    for i1_r in eachindex(A.k1)
                        if δg1[i1_r] in 1:size(A.δg,1)
                            i1_c = k1c[i1_r]
                            tmp = ρkp[:,i1_r,:,ir_r,:,i1_c,:,ir_c,δg1[i1_r],δg2[ir_r]] ./ A.l1 # (2,l2,2,l2)
                            for n2 in eachindex(A.nvec), n1 in eachindex(A.nvec)
                                θ1 = exp(1im * 2π * A.k1[i1_r] * A.nvec[n1]) * exp(-1im * 2π * A.k1[i1_c] *(A.nvec[n2]+s1)) * θ0
                                Ot[:,ir_r,n1,ik1,:,:,ir_c,n2,ip1,:] += tmp .* θ1
                            end
                        end
                    end
                end
            end
        end
    end


    for ip in 1:(A.l1*A.l2), ik in 1:(A.l1*A.l2)
        Uort_k = view(A.Uort,:,:,ik)
        Uort_p = view(A.Uort,:,:,ip)
        A.OΨ[:,ik,:,ip] = Uort_k' * view(A.Ot,:,ik,:,ip) * Uort_p
    end

    # test Hermiticity -- if correct then this correspond to the q=0 overlap matrix
    # if m==0 && n==0 
    #     ik0 = rand(1:A.l1*A.l2)
    #     Ot0 = view(A.Ot,:,ik0,:,ik0)
    #     O0 = view(A.O,:,:,ik0)
    #     nrmO = norm(O0 - Ot0)
    #     if nrmO > 1e-6 
    #         println("error with Hermiticity ",ik0," ",nrmO)
    #     end
    # end
    return nothing
end

# ------------------------------------------------------------------------------------------------------- #
function ConstructHBM_Hoftstadter(hof::HofstadterVq,params::Params;lg::Int=9)
    blk = HBM()
    s0 = Float64[1 0; 0 1]
    s1 = Float64[0 1; 1 0]
    is2 = Float64[0 1; -1 0]

    @assert (lg-1)%2 == 0   # we deal with lg being odd, such that -g and g are related easily
    blk.lg = lg
    blk.listG = zeros(Int,3,lg^2)
    for i2 in 1:lg, i1 in 1:lg
        blk.listG[1,(i2-1)*lg+i1] = i1
        blk.listG[2,(i2-1)*lg+i1] = i2 
        blk.listG[3,(i2-1)*lg+i1] = (i2-1)*lg+i1
    end
    blk.gvec = zeros(ComplexF64,lg^2)
    for ig in 1:lg^2
        blk.gvec[ig] = params.g1 * blk.listG[1,ig] + params.g2 * blk.listG[2,ig]
    end
    G0 = params.g1 * ((lg+1)÷2) + params.g2 * ((lg+1)÷2) # index of 1st Moire BZ is (lg+1)÷2,(lg+1)÷2
    blk.gvec .= blk.gvec .- G0

    # this gives C2T eigenstates
    Ig = Array{Float64}(I,blk.lg^2,blk.lg^2)
    blk.C2T = kron(Ig,kron(s0,s1)) # × conj(...)

    blk.nlocal = 4
    blk.nflat = 2
    
    l1 = size(hof.kvec,1)
    l2 = size(hof.kvec,2)
    blk.Uk = zeros(ComplexF64,blk.nlocal*blk.lg^2,blk.nflat*l1*l2)
    blk.Hk =zeros(Float64,blk.nflat,l1*l2)

    blk.T12 = zeros(ComplexF64,blk.nlocal*blk.lg^2,blk.nlocal*blk.lg^2)
    generate_T12(blk,params)

    # temporary container H for each k
    H = zeros(ComplexF64,blk.nlocal*blk.lg^2,blk.nlocal*blk.lg^2)
    kvec = reshape(hof.kvec,:)
    for ik in eachindex(kvec)
        kval = kvec[ik]
        ComputeH(H,blk,params,kval)
        # Find the smallest eigenvalue and eigenvectors close to zero
        vals, vecs = eigs(Hermitian(H),nev=blk.nflat,which=:SM)
        # C2T
        vecs = vecs + blk.C2T*conj(vecs)
        for i in 1:blk.nflat
            tmp = view(vecs,:,i)
            normalize!(tmp)
        end

        if (norm(imag(vals))<1e-6)
            perm = sortperm(real(vals[:]))
            blk.Uk[:,(blk.nflat*(ik-1)+1):(blk.nflat*ik)] = view(vecs,:,perm)
            blk.Hk[:,ik] = real(vals[perm])
        else
            print("Error with Hermiticity of Hamiltonian!\n")
        end
    end

    return blk
end

function ConstructHybridWannier_Hofstadter(hof::HofstadterVq,blk::HBM)
    """
    Construct hybrid Wannier states from Bloch eigenstates
    """
    l1 = size(hof.kvec,1)
    l2 = size(hof.kvec,2)
    lg = blk.lg
    nlocal = blk.nlocal

    A = HybridWannier()
    A.Wη = zeros(ComplexF64,2,2,l2)  # Wilson loop
    A.αη = zeros(ComplexF64,2,2,l2)
    A.ϵη = zeros(ComplexF64,2,l2)  # the eigenvalue of the Wilson loop associated with the eigenvector [1,i]
    A.Λη = zeros(ComplexF64,2,2,l1,l2) # partial Wilson loop
    A.Wγm = zeros(ComplexF64,2,2,l1,l2)
    A.WγLS = zeros(ComplexF64,4*lg^2,2,l1,l2) # hybrid wannier states in layer-sublattice basis
    A.Hγk = zeros(ComplexF64,2,2,l1,l2)
    Uk = reshape(blk.Uk,:,2,l1,l2)

    for ik in 1:l2 
        A.αη[:,:,ik] = ComplexF64[1 1; 1im -1im] / sqrt(2)
    end
    
    # smooth gauge along g2
    for ik in 1:(l2-1)
        for n in 1:2
            u1 = view(Uk,:,n,1,ik)
            u2 = view(Uk,:,n,1,ik+1)
            Λ = u1'*u2
            if real(Λ[1,1])<0
                u2 .*= -1
            end
        end
    end

    for ik in 1:(l2-1)
        u1 = view(Uk,:,2,1,ik)
        u2 = view(Uk,:,2,1,ik+1)
        Λ = u1'*u2
        if real(Λ[1,1])<0
            u2 .*= -1
        end
    end
    
    # cut along g2
    for ik in 1:l2
        A.Λη[:,:,1,ik] .= Array{ComplexF64}(I,2,2)
        for jk in 1:l1 # along g1 direction
            u1 = view(Uk,:,:,jk,ik)
            if (jk<l1)
                u2 = view(Uk,:,:,jk+1,ik)
                F = svd(u1'*u2)
                A.Λη[:,:,jk+1,ik] .= A.Λη[:,:,jk,ik]*(F.U*F.Vt)
            else 
                #(N-1,1)
                u2 = reshape(view(Uk,:,:,1,ik),nlocal,lg,lg,2)
                u2 = reshape(circshift(u2,(0,-1,0,0)), nlocal*lg^2,2)
                F = svd( u1'*u2 )
                A.Wη[:,:,ik] .= A.Λη[:,:,jk,ik] * (F.U*F.Vt)
            end
        end
        F = eigen(A.Wη[:,:,ik])
        vals = A.αη[:,:,ik]' * A.Wη[:,:,ik] * A.αη[:,:,ik]
        if (abs(vals[1,2]+abs(vals[2,1]))>1e-5)
            print("(1,i) is not the eigenvector\n")
        else
            A.ϵη[1,ik] = vals[1,1]
            A.ϵη[2,ik] = vals[2,2]
        end
    end
    
    # construct wannier states
    for ik in 1:l2
        for j in 1:l1
            A.Wγm[:,:,j,ik] = inv(A.Λη[:,:,j,ik]) * A.αη[:,:,ik] * Diagonal(A.ϵη[:,ik].^((j-1)/l1)) 
            u = view(Uk,:,:,j,ik)
            A.WγLS[:,:,j,ik] = u*A.Wγm[:,:,j,ik]
        end 
        pA = norm(A.WγLS[1:2:(4*lg^2),1,1,ik])
        pB = norm(A.WγLS[2:2:(4*lg^2),1,1,ik])
        # print("\n")
        # print(pA," ",pB,"\n")
        if pA < pB
            A.ϵη[:,ik] = A.ϵη[2:-1:1,ik]
            A.αη[:,:,ik] = A.αη[:,2:-1:1,ik]
            A.Wγm[:,:,:,ik] = A.Wγm[:,2:-1:1,:,ik]
            A.WγLS[:,:,:,ik] = A.WγLS[:,2:-1:1,:,ik]
        end
    end
    
    for i2 in 1:l2
        for i1 in 1:l1
            A.Hγk[:,:,i1,i2] =  A.Wγm[:,:,i1,i2]' * Diagonal(blk.Hk[:,(i2-1)*l1+i1]) * A.Wγm[:,:,i1,i2]
        end
    end
    return A
end

function computeLLOverlap(A::HofstadterVq,basis::HybridWannier,params::Params)
    A.nγ = 2
    A.nLL = 10A.q 
    A.nH = 2A.nLL - 1  # arranged in 0, -1, 1, -2, 2, ...
    lB  = sqrt(sqrt(3)*A.q/(4π)) * abs(params.a1) # magnetic length

    Λt = zeros(ComplexF64,2A.q*length(A.nvec),2A.nH,A.l1*A.l2)
    lg = 9
    gq = (-(lg-1)÷2):((lg-1)÷2)
    Wγ = reshape(basis.WγLS,2,2,lg,lg,2,A.l1,A.l2,A.q)  # sublattice, layer, z1g1, z2g2, γ, \bar{k1}, k2, r

    @inline function indexLL(iα::Int,iLL::Int,ilayer::Int)
        # ilayer = 1,2 ; iLL = 0, 1... nLL-1; iγ = 1,2 
        if iLL == 0 # then iγ is irrelevant
            iH = 1 + A.nH * ( ilayer -1)
        else # iα = 1,2
            iH = iα + (2iLL-1) + A.nH * (ilayer - 1)
        end
        return iH
    end
    indexHw = reshape( collect(1:(2A.q*length(A.nvec))), 2,A.q,length(A.nvec))

    # s1 = @. reshape(A.r,1,:) + reshape(gq,:,1) * A.q  # (z2,r)
    # θ1 = @. exp(- 1im * π * reshape(A.k2,1,1,:) * reshape(s1,lg,A.q) ) # (z2,r,k2)
    # kxbar = @. ( reshape(A.k1,:,1,1,1,1) * reshape(A.nvec,1,:,1,1,1)  - 
    #             (reshape(A.k2,1,1,1,1,:) + reshape(s1,1,1,lg,A.q,1)/A.q)/2  ) *abs(A.g1) #(k1bar,n0,z2,r,k2)
    # x0t = @. ( 2π * (reshape(A.k2,1,1,:) + reshape(s1,lg,A.q,1) /A.q) / abs(params.a1) - imag(params.Kt) ) * lB^2 # (z2,r,k2)
    # x0b = @. ( 2π * (reshape(A.k2,1,1,:) + reshape(s1,lg,A.q,1) /A.q) / abs(params.a1) - imag(params.Kb) ) * lB^2 # (z2,r,k2)
    # θ2 = @. exp(1im * 2π * reshape(A.k1,:,1) * reshape(A.nvec,1,:)) #(k1bar,n0)
    # θ3 = @. exp(-1im * reshape(kxbar,A.l1,length(A.nvec),1,1,1) * reshape(x0l,1,1,lg,A.q,A.l2)) # (k1bar,n0,z2,r,k2)
    
    for ik2 in 1:A.l2
        println(ik2)
        for ir in 1:A.q, z2 in 1:lg 
            s1 = A.r[ir] + gq[z2] * A.q 
            θ1 = exp(1im * 2π * (-A.k2[ik2]/2)*s1) * exp(-1im * π * s1 * (s1-1) /(2A.q))
            for ilayer in 1:2
                Kl = ( ilayer == 1 ? imag(params.Kb) : imag(params.Kt))
                x0l = ( 2π * (A.k2[ik2] + s1/A.q) / abs(params.a1) - Kl ) * lB^2
                for z1 in 1:lg 
                    for ik1bar in eachindex(A.k1)
                        kxbar = ( (A.k1[ik1bar] + gq[z1] )-(A.k2[ik2]+s1/A.q)/2 ) * abs(A.g1)
                        θ3 = exp(-1im * kxbar * x0l)
                        for iLL in 0:(A.nLL-1), iα in 1:2
                            iH = indexLL(iα,iLL,ilayer)
                            if iLL == 0 
                                tmpVec = [ΨLL(iLL,kxbar*lB); 0] .* 0.5  # x0.5 is because iα = 1,2 summs up twice
                            else
                                tmpVec = [ΨLL(iLL,kxbar*lB); -1im * (2iα-3)*ΨLL(iLL-1,kxbar*lB)] ./ sqrt(2) 
                            end
                            for iγ in 1:2
                                tmp = Wγ[:,ilayer,z1,z2,iγ,ik1bar,ik2,ir]' * tmpVec * θ1  * θ3  * ( sqrt(lB/real(params.a1)) / A.l1 )
                                for ik1 in 1:A.l1, n0 in eachindex(A.nvec)
                                    ik = (ik2-1) * A.l1 + ik1
                                    θ2 = exp(1im * 2π * A.k1[ik1bar] * A.nvec[n0]) * exp(1im * 2π * A.k1[ik1]*s1)
                                    Λt[indexHw[iγ,ir,n0],iH,ik] += θ2 * tmp
                                end
                            end
                        end
                    end
                end
            end
        end
    end

    A.Λ = zeros(ComplexF64,2A.q,2A.nH,A.l1*A.l2)
    for ik in 1:(A.l1*A.l2)
        A.Λ[:,:,ik] = A.Uort[:,:,ik]' * Λt[:,:,ik]
    end
    return nothing
end

@inline function ΨLL(n::Int,x::Float64)  #Landau level wavefunction -dimensionless part
    return sqrt(2)*π^(1/4)/(sqrt((2.0)^n*factorial(big(n)))) * (-1im)^n * exp(-x^2/2) * hermiteh(n,x)
end


function computeLLOverlapSingleK(A::HofstadterVq,basis::HybridWannier,params::Params)
    A.nγ = 2
    A.nLL = 10A.q 
    A.nH = 2A.nLL - 1  # arranged in 0, -1, 1, -2, 2, ...
    lB  = sqrt(sqrt(3)*A.q/(4π)) * abs(params.a1) # magnetic length
    ik1, ik2 = 1, 1

    Λt = zeros(ComplexF64,2A.q*length(A.nvec),2A.nH,1)
    lg = 9
    gq = (-(lg-1)÷2):((lg-1)÷2)
    Wγ = reshape(basis.WγLS,2,2,lg,lg,2,A.l1,A.l2,A.q)  # sublattice, layer, z1g1, z2g2, γ, \bar{k1}, k2, r

    @inline function indexLL(iα::Int,iLL::Int,ilayer::Int)
        # ilayer = 1,2 ; iLL = 0, 1... nLL-1; iγ = 1,2 
        if iLL == 0 # then iγ is irrelevant
            iH = 1 + A.nH * ( ilayer -1)
        else # iα = 1,2
            iH = iα + (2iLL-1) + A.nH * (ilayer - 1)
        end
        return iH
    end
    indexHw = reshape( collect(1:(2A.q*length(A.nvec))), 2,A.q,length(A.nvec))
    
    for ir in 1:A.q, z2 in 1:lg 
        s1 = A.r[ir] + gq[z2] * A.q 
        # if abs(s1) < A.l1÷2
            θ1 = exp(1im * 2π * (A.k1[ik1]-A.k2[ik2]/2)*s1) * exp(-1im * π * s1 * (s1-1) /(2A.q))
            for ilayer in 1:2
                Kl = ( ilayer == 1 ? params.Kb : params.Kt)
                x0l = ( 2π * (A.k2[ik2] + s1/A.q) / abs(params.a1) - imag(Kl) ) * lB^2
                for z1 in 1:lg , ik1bar in eachindex(A.k1)
                    kxbar = ( (A.k1[ik1bar] + gq[z1] )-(A.k2[ik2]+s1/A.q)/2 ) * abs(A.g1)
                    θ3 = exp(-1im * kxbar * x0l)
                    for iLL in 0:(A.nLL-1), iα in 1:2
                        xnum = (kxbar - real(Kl)) * lB
                        iH = indexLL(iα,iLL,ilayer)
                        if iLL == 0 
                            tmpVec = [ΨLL(iLL,xnum); 0] .* 0.5  # x0.5 is because iα = 1,2 summs up twice
                        else
                            tmpVec = [ΨLL(iLL,xnum); -1im * (2iα-3)*ΨLL(iLL-1,xnum)] ./ sqrt(2) 
                        end
                        for iγ in 1:2
                            tmp = (Wγ[:,ilayer,z1,z2,iγ,ik1bar,ik2,ir]' * tmpVec) * θ1  * θ3 * ( sqrt(lB/real(params.a1)) / A.l1 )
                            for n0 in eachindex(A.nvec)
                                θ2 = exp(1im * 2π * A.k1[ik1bar] * A.nvec[n0])
                                Λt[indexHw[iγ,ir,n0],iH,1] += θ2 * tmp
                            end
                        end
                    end
                end
            end
        # end
    end

    A.Λ = zeros(ComplexF64,2A.q,2A.nH,1)

    for ik in 1:size(A.Λ,3)
        A.Λ[:,:,ik] = A.Uort[:,:,ik]' * Λt[:,:,ik]
    end
    return nothing
end


function computeLLOverlapSingleKv1(A::HofstadterVq,basis::HybridWannier,params::Params)
    A.nγ = 2
    A.nLL = 10A.q 
    A.nH = 2A.nLL - 1  # arranged in 0, -1, 1, -2, 2, ...
    lB  = sqrt(sqrt(3)*A.q/(4π)) * abs(params.a1) # magnetic length
    ik1, ik2 = 1, 1

    Λt = zeros(ComplexF64,2A.q*length(A.nvec),2A.nH,1)
    lg = 9
    gq = (-(lg-1)÷2):((lg-1)÷2)
    Wγ = reshape(basis.WγLS,2,2,lg,lg,2,A.l1,A.l2,A.q)  # sublattice, layer, z1g1, z2g2, γ, \bar{k1}, k2, r

    @inline function indexLL(iα::Int,iLL::Int,ilayer::Int)
        # ilayer = 1,2 ; iLL = 0, 1... nLL-1; iγ = 1,2 
        if iLL == 0 # then iγ is irrelevant
            iH = 1 + A.nH * ( ilayer -1)
        else # iα = 1,2
            iH = iα + (2iLL-1) + A.nH * (ilayer - 1)
        end
        return iH
    end
    indexHw = reshape( collect(1:(2A.q*length(A.nvec))), 2,A.q,length(A.nvec))
    
    for ir in 1:A.q, z2 in 1:lg 
        s1 = A.r[ir] + gq[z2] * A.q 
        # if abs(s1) < A.l1÷2
            θ1 = exp(1im * 2π * (A.k1[ik1])*s1) * exp(1im * π * s1 * (s1+1) /(2A.q))
            for ilayer in 1:2
                Kl = ( ilayer == 1 ? params.Kb : params.Kt)
                x0l = ( 2π * A.k2[ik2] / abs(params.a1) - imag(Kl) ) * lB^2
                for z1 in 1:lg , ik1bar in eachindex(A.k1)
                    kxbar = ( (A.k1[ik1bar] + gq[z1] )-(A.k2[ik2]+s1/A.q)/2 ) * abs(A.g1)
                    θ3 = exp(-1im * kxbar * x0l)
                    for iLL in 0:(A.nLL-1), iα in 1:2
                        iH = indexLL(iα,iLL,ilayer)
                        if iLL == 0 
                            tmpVec = [ΨLL(iLL,(kxbar-real(Kl))*lB); 0] .* 0.5  # x0.5 is because iα = 1,2 summs up twice
                        else
                            tmpVec = [ΨLL(iLL,(kxbar-real(Kl))*lB); -1im * (2iα-3)*ΨLL(iLL-1,(kxbar-real(Kl))*lB)] ./ sqrt(2) 
                        end
                        for iγ in 1:2
                            tmp = (Wγ[:,ilayer,z1,z2,iγ,ik1bar,ik2,ir]' * tmpVec) * θ1  * θ3 * ( sqrt(lB/real(params.a1)) / A.l1 )
                            for n0 in eachindex(A.nvec)
                                θ2 = exp(1im * 2π * A.k1[ik1bar] * (A.nvec[n0]-s1))
                                Λt[indexHw[iγ,ir,n0],iH,1] += θ2 * tmp
                            end
                        end
                    end
                end
            end
        # end
    end

    A.Λ = zeros(ComplexF64,2A.q,2A.nH,1)

    for ik in 1:size(A.Λ,3)
        A.Λ[:,:,ik] = A.Uort[:,:,ik]' * Λt[:,:,ik]
    end
    return nothing
end