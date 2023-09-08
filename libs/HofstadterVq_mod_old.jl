include("hybridWannier_mod.jl")
using LinearAlgebra
using TensorOperations

mutable struct HofstadterVq 
    # This module contains methods for calculating Hofstadter at strong coupling
    p::Int
    q::Int 

    lk::Int  # 2nq q
    nq::Int 
    l1::Int
    l2::Int 
    k1::Vector{Float64} # grid in the range of [0,1)
    k2::Vector{Float64} # grid in the range of [0,1/q) 
    k2mbz::Vector{Float64} # grid along g2 in the range of [0,1)
    kvec::Matrix{ComplexF64} # grid of k points in the Moire Brillouin zone
    r::Vector{Int} # 0...q-1
    g1::ComplexF64 
    g2::ComplexF64

    M::Array{ComplexF64,3}  # (γ1, k), (γ2, p), δg 
    Mz::Array{ComplexF64,3} # polarization σz
    δg::Array{Complex{Int}} 

    O::Array{ComplexF64,8}  # (nγr)x(nγr)xk1xk2
    Oz::Array{ComplexF64,8}  # (nγr)x(nγr)xk1xk2
    nvec::Vector{Int} # nvec = [0,1]
    svec::Vector{Int} # svec = -nc:nc

    # coefficients relating trial MTG and orthonormalized MTG eigenstates
    Uort::Array{ComplexF64,4}  #(nγr)x(γr)xk1xk2 
    Σz::Array{ComplexF64,4} # (γr) x (γr) x x k1 x k2 # eigenvalues of σz μ_0

    #
    H::Array{ComplexF64,4} # (γr) x (γr) x k1 x k2 -- for every k in magnetic strip, γr x γr matrix
 
    HofstadterVq() = new()

end

function initHofstadterVq(params::Params;p::Int=1,q::Int=5,lk::Int=5)
    
    A = HofstadterVq()
    ϕ = p//q 

    A.p = numerator(ϕ)
    A.q = denominator(ϕ)
    A.lk = lk 
    A.g1 = params.g1 
    A.g2 = params.g2
    A.nvec = collect(0:1)
    A.svec = collect(-2:2)

    # ------ lattice initialization ------ #
    if (A.q>=lk)
        A.nq = 1
    else
        A.nq = (A.lk-1) ÷ A.q + 1
        A.lk = 2*A.q * A.nq
    end

    A.l1 = 2 * A.q * A.nq 
    A.k1 = collect(0:(A.l1-1)) ./ (A.l1)
    A.l2 = 2*A.nq
    A.k2 = collect(0:(A.l2-1)) ./ (A.l2*A.q)
    A.r = collect(0:(A.q-1))
    A.k2mbz = [ A.k2[j] + A.r[i]/A.q for i in eachindex(A.r) for j in eachindex(A.k2)] 
    A.kvec = reshape(A.k1,:,1) * A.g1 .+ reshape(A.k2mbz,1,:) * A.g2

    # --------- Bloch part of the story ------- # 
    blk = ConstructHBM_Hoftstadter(A,params)
    basis = ConstructHybridWannier_Hofstadter(A,blk)
    computeM(A, blk,basis)
    GC.gc()

    # --------- Overlap matrix and orthonormalization ---------- # 
    A.O, A.Oz, kp_pairs0 = computeOverlapMatrix(A,0+0im,flagz=true)
    computeOrtNormMTG(A)

    # --------- Matrix elements at strong coupling ------- # 
    constructStrongCouplingSP(A)

    return A
end

@inline function Coulomb(q::ComplexF64)
    return abs(q)>1e-5 ? 4π/abs(q) : 0.0
end

function constructStrongCouplingSP(A::HofstadterVq)
    # q grid to consider 
    qc = 1 
    tmpq = (-qc*A.l1÷2):(qc*A.l1÷2)
    qgrid = reshape(tmpq,:,1) * (A.g1/A.l1) .+ reshape(tmpq,1,:) *(A.g2/A.l1)

    A.H = zeros(ComplexF64,2A.q,2A.q,A.l1,A.l2)
    for iq2 in 1:size(qgrid,2)
        for iq1 in 1:size(qgrid,1)
            println(iq1," ",iq2," ")
            qval = qgrid[iq1,iq2]
            Tq, kp_pairs = computeOverlapMatrix(A,tmpq[iq1]+1im*tmpq[iq2])
            Tq = reshape(Tq,2*A.q*length(A.nvec),2*A.q*length(A.nvec),A.l1,A.l2)
            for i2 in 1:A.l2
                for i1 in 1:A.l1
                    Uk = view(A.Uort,:,:,i1,i2)
                    i1p = real(kp_pairs[2,i1,i2])
                    i2p = imag(kp_pairs[2,i1,i2])
                    Up = view(A.Uort,:,:,i1p,i2p)
                    OΨ = Uk' * view(Tq,:,:,i1,i2) * Up 
                    A.H[:,:,i1,i2] .+= Coulomb(qval) *(OΨ * OΨ')
                end
            end
        end
    end
    V0 = 1/abs(params.a1)
    A.H .= A.H .* (0.5/(sqrt(3)*abs(params.a1)^2*A.l1^2) / V0)
    return nothing
end

function computeM(A::HofstadterVq,blk::HBM,basis::HybridWannier)
    # Given hybrid wannier states from blk.Uγk[4lg^2,2,lk1,lk2], compute overlap matrix 
    # M(γ1k,γ2p,q) = U_g(γ1k)^† U_g+δg(γ2p)
    gc = 3
    A.δg = [ig1 + 1im*ig2 for ig2 in -gc:gc for ig1 in -gc:gc]
    Uγk = reshape(basis.WγLS,4blk.lg^2,2A.l1*A.l2*A.q)
    A.M = zeros(ComplexF64,2A.l1*A.l2*A.q,2A.l1*A.l2*A.q,length(A.δg))
    A.Mz = zeros(ComplexF64,2A.l1*A.l2*A.q,2A.l1*A.l2*A.q,length(A.δg))
    Oσ = kron(Array{ComplexF64}(I,blk.lg^2,blk.lg^2),kron(ComplexF64[1 0;0 1],ComplexF64[1 0;0 -1]))
    for ig in eachindex(A.δg)
        Uγp = reshape(Uγk,4,blk.lg,blk.lg,:)
        Uγp = circshift(Uγp,(0,-real(A.δg[ig]),-imag(A.δg[ig]),0))
        Uγp = reshape(Uγp,4blk.lg^2,:)
        A.M[:,:,ig] = Uγk' * Uγp
        A.Mz[:,:,ig] = Uγk' * Oσ * Uγp
    end
    return nothing
end

function computeOverlapMatrix(A::HofstadterVq,iq::Complex{Int};flagz=false)
    # Given A.M, construct the overlap matrix ⟨T|e^{iqr}|T⟩
    # iq = iq1 + 1im * iq2 denote the index shift from the Coulomb term
    ρkp = reshape(A.M,2,A.l1,A.l2*A.q,2,A.l1,A.l2*A.q,length(A.δg))

    if (flagz == true)
        ρzkp = reshape(A.Mz,2,A.l1,A.l2*A.q,2,A.l1,A.l2*A.q,length(A.δg))
        Oz = zeros(ComplexF64,A.l1,A.l1,length(A.svec),2,length(A.r),2,length(A.r),A.l2) 
    end

    @inline function idx_k2(ik::Int)
        return mod(ik-1,A.l2) + 1
    end
    @inline function idx_r(ik::Int)
        return (ik-1)÷A.l2 + 1
    end
    
    # for a given q, there is a unique k-p pair in magnetic Brillouin zone where matrix elements does not vanish
    kp_pairs = zeros(Complex{Int},2,A.l1,A.l2)  
    
    # g1 axis: k1 - q = p1
    tmp = eachindex(A.k1) .- real(iq)
    p1 = mod.(tmp .-1,A.l1) .+ 1
    kp_pairs[1,:,:] .= reshape(eachindex(A.k1),:,1)
    kp_pairs[2,:,:] .= reshape(p1,:,1) 

    # g2 axis: k2 -q = p2 
    tmp = eachindex(A.k2) .- imag(iq)
    p2 = mod.(tmp .-1,A.l2) .+ 1
    kp_pairs[1,:,:] .+= 1im * reshape(eachindex(A.k2),1,:)
    kp_pairs[2,:,:] .+= 1im * reshape(p2,1,:) 

    O0 = zeros(ComplexF64,A.l1,A.l1,length(A.svec),2,length(A.r),2,length(A.r),A.l2)  # each kr uniquely associated with one kc
    
    for s1 in eachindex(A.svec)
        qϕ1 = A.svec[s1] * A.p + real(iq)
        tmp = eachindex(A.k1) .- qϕ1   # split into [k1-q1] + δg
        k1c = mod.(tmp .-1,A.l1) .+ 1
        δg1 = (tmp .- k1c) .÷ A.l1
        
        qϕ2 = A.svec[s1] * A.p * A.l2 + imag(iq)
        tmp = eachindex(A.k2mbz) .- qϕ2   # split into [k2-q2] + δg
        k2c = mod.(tmp .-1,A.l2*A.q) .+ 1
        δg2 = (tmp .- k2c) .÷ (A.l2*A.q)

        δg = reshape(δg1,:,1) .+ 1im * reshape(δg2,1,:) 

        for i2r in eachindex(A.k2mbz)
            i2c = k2c[i2r]
            for i1r in eachindex(A.k1) 
                i1c = k1c[i1r]
                ig = findfirst(x->x==δg[i1r,i2r],A.δg)
                if isnothing(ig)
                    println("Err: no index of ig found in computeOverlapMatrix()")
                end
                # if (idx_k2(i2r)!= idx_k2(i2c))
                #     println("Err: magnetic Brillouin zone indexing incorrectly implemented")
                # end
                O0[i1r,i1c,s1,:,idx_r(i2r),:,idx_r(i2c),idx_k2(i2r)] = @view ρkp[:,i1r,i2r,:,i1c,i2c,ig]
                if (flagz==true)
                    Oz[i1r,i1c,s1,:,idx_r(i2r),:,idx_r(i2c),idx_k2(i2r)] = @view ρzkp[:,i1r,i2r,:,i1c,i2c,ig]
                end
            end
        end
    end

    # 
    tmpO = zeros(ComplexF64,2,A.q,length(A.nvec),2,A.q,length(A.nvec),A.l1,A.l2)
    # expfactors = ones(ComplexF64,A.l1,A.l1,length(A.svec),2,length(A.r),length(A.nvec),2,length(A.r),length(A.nvec),A.l1,A.l2)

    k1 = reshape(A.k1,1,1,1,1,1,1,1,1,1,:,1)
    p1 = reshape(A.k1[real(kp_pairs[2,:,1])],1,1,1,1,1,1,1,1,1,:,1)
    svec = reshape(A.svec,1,1,:,1,1,1,1,1,1,1,1)
    n1vec = reshape(A.nvec,1,1,1,1,1,:,1,1,1,1,1)
    n2vec = reshape(A.nvec,1,1,1,1,1,1,1,1,:,1,1)
    k1bar = reshape(A.k1,:,1,1,1,1,1,1,1,1,1,1)
    p1bar = reshape(A.k1,1,:,1,1,1,1,1,1,1,1,1)
    O0 = reshape(O0,A.l1,A.l1,length(A.svec),2,length(A.r),1,2,length(A.r),1,1,A.l2)
    
    tmpO .= reshape(
                sum( ( @. exp(1im * 2π * p1 * svec) * exp(-1im *π * svec * (svec-1) * A.p/(2A.q)) *
                exp(1im * 2π * k1bar * n1vec) * exp(-1im * 2π * p1bar * (n2vec + svec)) * O0 ) , dims = (1,2,3) ) , 
                (2,A.q,length(A.nvec),2,A.q,length(A.nvec),A.l1,A.l2) )

    if (flagz==true)
        Oz = reshape(Oz,A.l1,A.l1,length(A.svec),2,length(A.r),1,2,length(A.r),1,1,A.l2)
        tmpOz = zeros(ComplexF64,2,A.q,length(A.nvec),2,A.q,length(A.nvec),A.l1,A.l2)
        tmpOz .= reshape(
            sum( ( @. exp(1im * 2π * p1 * svec) * exp(-1im *π * svec * (svec-1) * A.p/(2A.q)) *
            exp(1im * 2π * k1bar * n1vec) * exp(-1im * 2π * p1bar * (n2vec + svec)) * Oz ) , dims = (1,2,3) ) , 
            (2,A.q,length(A.nvec),2,A.q,length(A.nvec),A.l1,A.l2) )
    end

    if (flagz == true)
        return tmpO, tmpOz, kp_pairs
    else 
        return tmpO, kp_pairs
    end
end

function computeOrtNormMTG(A::HofstadterVq)
    # |Ψ⟩ = |T⟩ Uort for any k1, k2 in magnetic Brillouin zone
    A.Uort = zeros(ComplexF64,2A.q*length(A.nvec),2A.q,A.l1,A.l2)
    A.Σz = zeros(ComplexF64,2A.q,2A.q,A.l1,A.l2)
    for i2 in 1:A.l2
        for i1 in 1:A.l1
            O = reshape(view(A.O,:,:,:,:,:,:,i1,i2),2A.q*length(A.nvec),2A.q*length(A.nvec))
            Oz = reshape(view(A.Oz,:,:,:,:,:,:,i1,i2),2A.q*length(A.nvec),2A.q*length(A.nvec))
            F = eigen(Hermitian(O))

            A.Uort[:,:,i1,i2] = ( @view F.vectors[:,(end-2A.q+1):end]) * 
                                Diagonal( 1 ./sqrt.( F.values[(end-2A.q+1):end] ) )

            Uort = view(A.Uort,:,:,i1,i2)
            
            A.Σz[:,:,i1,i2] = Uort' * Oz * Uort
        end
    end
    return nothing
end

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

    for ik in eachindex(view(hof.kvec,:))
        kval = view(hof.kvec,:)[ik]
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

