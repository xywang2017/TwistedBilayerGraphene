include("hybridWannier_mod.jl")
using StatsBase
using LinearAlgebra

mutable struct HofstadterVq 
    # This module contains methods for calculating Hofstadter at strong coupling
    # implement calculation of O_{j1k,j2p}(g_q) instead, alternative scheme
    # computes for  single external k1 and k2 value
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
    
    # Strong coupling energetics
    gq::Array{Complex{Int},2}  # m g1 + n g2/q; store m + i n
    OΨ::Array{ComplexF64,4}    # (j1 k1,k2) x (j2 p1 p2) for a given m,n 
    # Ot::Array{ComplexF64,4}    # (n1 r1 γ1 k1,k2) x (n2 r2 γ2 p1 p2) for a given m,n 
    
    #
    Σz::Array{ComplexF64,3} # (γr) x (γr) x k1 k2 # eigenvalues of σz μ_0
    H::Array{ComplexF64,3} # (γr) x (γr) x k1 k2 -- for every k in magnetic strip, γr x γr matrix

    n_ext::Array{Int}  # select momenta 
 
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
    A.svec = collect(-2:2)
    
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
    # A.l2 = A.nq
    A.k2 = collect(0:(A.l2-1)) ./ (A.l2*A.q)
    A.r = collect(0:(A.q-1))
    k2mbz = reshape( reshape(A.k2,:,1) .+ reshape(A.r ./A.q,1,:) , : )
    A.kvec = reshape(A.k1,:,1) * A.g1 .+ reshape(k2mbz,1,:) * A.g2

    #only compute for a few select external momentum states
    A.n_ext = sample(1:(A.l1*A.l2), min(A.l1*A.l2,10), replace=false)

    # --------- Bloch part of the story ------- # 
    blk = ConstructHBM_Hoftstadter(A,params)
    basis = ConstructHybridWannier_Hofstadter(A,blk)
    computeM(A, blk,basis)

    blk = 0 
    basis = 0
    GC.gc()

    # --------- Overlap matrix and orthonormalization ---------- # 
    # O, Oz
    computeOverlapMatrix(A)

    # Uort, Σz 
    computeOrtNormMTG(A)

    # --------- Matrix elements at strong coupling ------- # 
    computeSingleParticleSpectrum(A)

    return nothing
end

@inline function Coulomb(q::ComplexF64)
    return abs(q)>1e-5 ? 4π*tanh(0.5*abs(q)*abs(params.a1))/abs(q) : 0.0
    # return abs(q) > 1e-5 ? 10.0 : 0.0  # flat coulomb reduces finite size effects
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
    
    kext = kvec[A.n_ext]

    A.H = zeros(ComplexF64,2A.q,2A.q,length(A.n_ext))
    gq = real(A.gq) * A.g1 + imag(A.gq) * A.g2 / A.q
    A.OΨ = zeros(ComplexF64,2A.q,1,2A.q,A.l1*A.l2)

    
    for ig2 in 1:size(A.gq,2)
        println(ig2)
        for ig1 in 1:size(A.gq,1) 
            for I_H in eachindex(A.n_ext)
                computeCoulombOverlap(A,ig1,ig2,A.n_ext[I_H])
                OΨ1 = reshape(A.OΨ,2A.q,1,1,2A.q,A.l1*A.l2)
                OΨ2 = reshape(A.OΨ,1,2A.q,1,2A.q,A.l1*A.l2)

                Vq = reshape( Coulomb.( reshape([kext[I_H]],:,1) .- reshape(kvec,1,:) .- gq[ig1,ig2]), 
                            (1,1,1,1,A.l1*A.l2) )

                A.H[:,:,I_H] .+= reshape( sum(Vq .* OΨ1 .* conj.(OΨ2), dims=(4,5)), (2A.q, 2A.q))
            end
        end
    end

    Lm = (4π)/(sqrt(3)*abs(A.g1))
    V0 = 1/Lm
    A.H .*= (0.5/(sqrt(3)*Lm^2*A.l1^2) / V0)
    return nothing
end

function computeCoulombOverlap(A::HofstadterVq,m::Int,n::Int,I_ext::Int)

    Ik1 = mod(I_ext - 1, A.l1) + 1 
    Ik2 = (I_ext -1 )÷ A.l1 +1
    # overlap between trial MTG eigenstates
    ρkp = reshape( A.M, (2,A.l1,A.l2,A.q,2,A.l1,A.l2,A.q,size(A.δg,1),size(A.δg,2)) )
    ig0 = (size(A.δg,1) +1 )÷2 # index of δg = (0,0)

    Ot = zeros(ComplexF64,2,A.q,length(A.nvec),1,1,2,A.q,length(A.nvec),A.l1,A.l2)  # 39MB for q=10
    for ip1 in eachindex(A.k1)
        for s1 in A.svec
            tmp = eachindex(A.k1) .- s1*A.p*A.nq .- (Ik1-ip1-real(A.gq[m,n])*A.l1)
            k1c = mod.(tmp.-1,A.l1) .+ 1
            δg1 = (tmp .- k1c) .÷ A.l1 .+ ig0

            tmp = eachindex(A.r) .- s1 *A.p .+ imag(A.gq[m,n])
            r2 = mod.(tmp.-1,A.q) .+ 1 
            δg2 = (tmp .- r2) .÷ A.q .+ ig0

            θ0 = exp(1im * 2π * A.k1[ip1] * s1) * exp(-1im * π * s1*(s1-1) *A.p/(2A.q) )
            for ir_r in eachindex(A.r)
                if δg2[ir_r] in 1:size(A.δg,2)
                    ir_c = r2[ir_r]
                    for i1_r in eachindex(A.k1)
                        if δg1[i1_r] in 1:size(A.δg,1)
                            i1_c = k1c[i1_r]
                            tmp = ρkp[:,i1_r,Ik2,ir_r,:,i1_c,:,ir_c,δg1[i1_r],δg2[ir_r]] ./ A.l1 # (2,l2,2,l2)
                            for n2 in eachindex(A.nvec), n1 in eachindex(A.nvec)
                                θ1 = exp(1im * 2π * A.k1[i1_r] * A.nvec[n1]) * exp(-1im * 2π * A.k1[i1_c] *(A.nvec[n2]+s1)) * θ0
                                Ot[:,ir_r,n1,1,1,:,ir_c,n2,ip1,:] += tmp .* θ1
                            end
                        end
                    end
                end
            end
        end
    end

    Ot_reshaped = reshape(Ot,2A.q*length(A.nvec),1,2A.q*length(A.nvec),A.l1*A.l2)
    
    Uort_k = view(A.Uort,:,:,I_ext)
    for ip in 1:(A.l1*A.l2)
        Uort_p = view(A.Uort,:,:,ip)
        A.OΨ[:,1,:,ip] .= Uort_k' * view(Ot_reshaped,:,1,:,ip) * Uort_p
    end

    Ot = 0

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
        # F = eigen(Hermitian(H),sortby=abs)
        # vals = F.values[1:2]
        # vecs = F.vectors[:,1:2]
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
    for ik in 1:(l2-1)        for n in 1:2
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
