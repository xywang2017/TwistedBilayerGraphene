include("hybridWannier_mod.jl")
using SparseArrays

mutable struct HofstadterHoppingElements
    # This data structure stores Hn for a few nearest neighbors 
    ndim::Int
    nvec::Vector{Int} # <0| ... | n > 

    q::Int
    p::Int 
    l1::Int    # grid size along g1 
    l2::Int    # grid size along g2 
    δq1::Int   # steps for p/(2q) along g1 
    δq2::Int   # steps for p/q along g2

    k1::Vector{Float64}  # (0...l1-1)/l1
    k2::Vector{Float64}  # (0...l2-1)/l2
    kvec::Matrix{ComplexF64}

    O0::Array{ComplexF64,6}
    O1::Array{ComplexF64,6}
    O2::Array{ComplexF64,6}
    O3::Array{ComplexF64,6}
    O4::Array{ComplexF64,6}
    Osum::Array{ComplexF64,6}
    # term 1 and 2 of the Hamiltonian
    A1::Array{ComplexF64,6}   
    A2::Array{ComplexF64,6}
    Asum::Array{ComplexF64,6}

    H::Array{ComplexF64,6}
    Hsq::Array{ComplexF64,6}
    M::Array{ComplexF64,6}  #overlap matrix

    spectrum::Vector{Float64}

    HofstadterHoppingElements() = new()
end

function ComputeOverlapMatrix(hof::HofstadterHoppingElements,basis::HybridWannier,blk::HBM,params::Params)
    # compute and save the matrices O0 through O4, defined as ⟨γ,k1,k2|O(n)|γ',k1',k2'⟩
    # O0 is the Hamiltonian matrix H0, 2 × 2 × l1 × l2 × 1 matrix elements 
    # O1 is exp(i n qϕ r), 2 × 2 × l1 × l2 × n matrix elements 
    # O2 is x exp(i n qϕ r), 2 × l1 × 2 × l1 × l2 × n matrix elements 
    # O3 is x^2 exp(i n qϕ r), 2 × l1 × 2 × l1 × l2 × n matrix elements 
    # O4 is x τyμ0 exp(i n qϕ r), 2 × l1 × 2 × l1 × l2 × n matrix elements 
    # O1...O4 are vectors of sparsematrices!
    # energies are vf l1x/lB^2,vf^2l1x^2/lB^4
    hof.O0 = zeros(ComplexF64,2,hof.l1,2,hof.l1,hof.l2,hof.ndim)
    hof.O1 = zeros(ComplexF64,2,hof.l1,2,hof.l1,hof.l2,hof.ndim)
    hof.O2 = zeros(ComplexF64,2,hof.l1,2,hof.l1,hof.l2,hof.ndim)
    hof.O3 = zeros(ComplexF64,2,hof.l1,2,hof.l1,hof.l2,hof.ndim)
    hof.O4 = zeros(ComplexF64,2,hof.l1,2,hof.l1,hof.l2,hof.ndim)

    hof.A1 = zeros(ComplexF64,2,hof.l1,2,hof.l1,hof.l2,hof.ndim)
    hof.A2 = zeros(ComplexF64,2,hof.l1,2,hof.l1,hof.l2,hof.ndim)

    ϵ0 = 2π * params.vf / abs(params.a1)    # this is 1.5 vF kθ
    ϵ0sq = ϵ0^2

    
    

    # -------------------------------- O0 -------------------------------- #
    lg = blk.lg
    Ig = Array{ComplexF64}(I,lg^2,lg^2)

    ## under strain 
    e2 = [real(params.a2)/abs(params.a2); imag(params.a2)/abs(params.a2)]
    tmp = kron(ComplexF64[1 0;0 1],-ComplexF64[0 -1im;1im 0]*e2[2]+ComplexF64[0 1;1 0]*e2[1])
    # e2S = params.S/2 * e2
    # tmp = tmp - kron(ComplexF64[1 0;0 -1],-ComplexF64[0 -1im;1im 0]*e2S[2]+ComplexF64[0 1;1 0]*e2S[1])
    O_LS = kron(Ig,tmp)

    k1rc = reshape(hof.k1,:,1) .- reshape(hof.k1,1,:)
    gq = (-(lg-1)÷2):((lg-1)÷2)

    # Fourier transform of x/L1x
    @inline function V1(q1::Float64,N1::Int)
        return abs(q1)<1e-5 ? 0.0im : 1im*cos(π*q1*N1)/(2π*q1)
    end

    # Fourier transform of (x/L1x)^2
    @inline function V2(q1::Float64,N1::Int)
        return abs(q1)<1e-5 ? N1^2/12 : cos(π*q1*N1)/(2*π^2*q1^2)
    end

    for n in eachindex(hof.nvec)
        qϕ1 = hof.nvec[n] * hof.δq1
        tmp = eachindex(hof.k1) .- qϕ1   # split into [k1-q1] + δg
        k1c = mod.(tmp .-1,hof.l1) .+ 1
        δg1 = (tmp .- k1c) .÷ hof.l1
        
        qϕ2 = hof.nvec[n] * hof.δq2
        tmp = eachindex(hof.k2) .- qϕ2   # split into [k1-q1] + δg
        k2c = mod.(tmp .-1,hof.l2) .+ 1
        δg2 = (tmp .- k2c) .÷ hof.l2
        
        for i2r in eachindex(hof.k2)
            i2c = k2c[i2r]
            for i1r in eachindex(hof.k1) 
                i1c = k1c[i1r]
                ur = view(basis.WγLS,:,:,i1r,i2r) 
                uc = view(basis.WγLS,:,:,i1c,i2c)
                uc = reshape(uc,4,blk.lg,blk.lg,2)
                uc = reshape(circshift(uc,(0,-δg1[i1r],-δg2[i2r],0)),4*blk.lg^2,2)
                λrc = ur' * uc
                hof.O0[:,i1r,:,i1c,i2r,n] = λrc 
                ϵ2x2 = basis.Hγk[:,:,i1r,i2r] ./ϵ0
                hof.O1[:,i1r,:,i1c,i2r,n] = ϵ2x2*ϵ2x2*λrc
                hof.A1[:,i1r,:,i1c,i2r,n] = ϵ2x2*λrc
            end

            for iq in eachindex(gq)
                 q1rc = k1rc .- gq[iq] .- hof.nvec[n]*hof.p/(2hof.q)
                 V1q = reshape(V1.(q1rc,hof.l1),1,hof.l1,1,hof.l1)
                 V2q = reshape(V2.(q1rc,hof.l1),1,hof.l1,1,hof.l1)
                 
                 
                 ur = reshape(view(basis.WγLS,:,:,:,i2r),4*blk.lg^2,2*hof.l1)
                 uc = view(basis.WγLS,:,:,:,i2c)
                 uc = reshape(uc,4,blk.lg,blk.lg,2*hof.l1)
                 uc = reshape(circshift(uc,(0,-gq[iq],-δg2[i2r],0)),4*blk.lg^2,2*hof.l1)

                 λrc = ur' * uc
                 λrc = reshape(λrc,2,hof.l1,2,hof.l1)
                 hof.O2[:,:,:,:,i2r,n] .+= λrc.*V2q
                 hof.O3[:,:,:,:,i2r,n] .+= λrc.*V1q .* hof.nvec[n]

                 λrc = ur' * O_LS * uc
                 λrc = reshape(λrc,2,hof.l1,2,hof.l1)
                 hof.O4[:,:,:,:,i2r,n] .+= λrc.*V1q
                 hof.A2[:,:,:,:,i2r,n] .+= λrc.*V1q
            end 
            for i1c in eachindex(hof.k1)
                for i1r in eachindex(hof.k1)
                    ϵr = basis.Hγk[:,:,i1r,i2r] ./ϵ0
                    ϵc = basis.Hγk[:,:,i1c,i2c] ./ϵ0
                    hof.O4[:,i1r,:,i1c,i2r,n] = ϵr * hof.O4[:,i1r,:,i1c,i2r,n] +
                                                    hof.O4[:,i1r,:,i1c,i2r,n] * ϵc
                end
            end
        end
    end

    # -------------------------------- Add up all terms H^2-------------------------------- #
    hof.Osum = hof.O1 + hof.O2*(hof.p/hof.q)^2- hof.O3*(hof.p/hof.q)^2 - hof.O4*(hof.p/hof.q)
    hof.Asum = hof.A1 - hof.A2*(hof.p/hof.q)
    n0 = collect(-4:4)
    n0dim = length(n0)
    hof.Hsq = zeros(ComplexF64,2,hof.l2,n0dim,2,hof.l2,n0dim)
    hof.H = zeros(ComplexF64,2,hof.l2,n0dim,2,hof.l2,n0dim)
    hof.M = zeros(ComplexF64,2,hof.l2,n0dim,2,hof.l2,n0dim)

    for nr in eachindex(n0)
        for nc in eachindex(n0)
            for n in eachindex(hof.nvec)
                factor = 1.0 
                if (n==1)
                    factor = 0.5
                end
                θr = reshape(exp.(1im * 2π * hof.k1 * n0[nr]),1,hof.l1,1,1)
                θc = reshape(exp.(-1im * 2π * hof.k1 * (n0[nc]+hof.nvec[n])),1,1,1,hof.l1)
                θϕ = exp(-1im * π * hof.nvec[n]*(hof.nvec[n]-1) * hof.p/(2hof.q) )

                qϕ2 = hof.nvec[n] * hof.δq2
                tmp = eachindex(hof.k2) .- qϕ2   # split into [k1-q1] + δg
                k2c = mod.(tmp .-1,hof.l2) .+ 1
                for i2r in eachindex(hof.k2)
                    i2c = k2c[i2r]
                    hof.Hsq[:,i2r,nr,:,i2c,nc] .+= factor * reshape(sum(θr.*view(hof.Osum,:,:,:,:,i2r,n).*θc,dims=(2,4)),2,2) * θϕ / hof.l1
                    hof.H[:,i2r,nr,:,i2c,nc] .+= factor * reshape(sum(θr.*view(hof.Asum,:,:,:,:,i2r,n).*θc,dims=(2,4)),2,2) * θϕ / hof.l1
                    hof.M[:,i2r,nr,:,i2c,nc] .+= factor * reshape(sum(θr.*view(hof.O0,:,:,:,:,i2r,n).*θc,dims=(2,4)),2,2) * θϕ /hof.l1
                end
            end
        end
    end

    return nothing
end

function orthonormalizationProcedure(hof::HofstadterHoppingElements,blk::HBM,params::Params)
    n0dim = size(hof.M,3)
    M = reshape(hof.M,n0dim*2hof.l2,n0dim*2hof.l2)
    H2 = reshape(hof.Hsq,n0dim*2hof.l2,n0dim*2hof.l2)
    H = reshape(hof.H,n0dim*2hof.l2,n0dim*2hof.l2)
    M = M + M' 
    H2 = H2 + H2'
    H = H + H'
    
    ϵ0 = maximum(blk.Hk) / (2π * params.vf/abs(params.a1))

    # how many eigenvectors to keep
    F = eigen(Hermitian(M))
    nstates = 2hof.l2

    val1 = F.values[end-nstates+1]
    val2 = F.values[end-nstates]

    if val2 / val1 > 0.8  # if no significant jump in overlap matrix
        threshold = 0.2
        nstates = length(F.values[F.values .> threshold])
    end
    println("Total number of states kept stage 1 is $nstates")

    # stage 1 using H^2 to filter out states
    vec = F.vectors'
    vals = 1 ./ sqrt.(F.values[(end-nstates+1):end]) 
    U = Diagonal(vals) * vec[(end-nstates+1):end,:]
    H2new = U * H2 * U'
    F = eigen(Hermitian(H2new))

    nstates = length(F.values[F.values .< ϵ0^2])
    if nstates < 2hof.l2  # incomplete basis at this flux
        println("Incomplete basis generated by MTG")
        println("Total number of sub-gap states kept stage 2 is $nstates")
    end

    # keep 82 states nontheless
    if (nstates<2hof.l2)
        nstates=2hof.l2
        vals0 = F.values[nstates-1]
        vals1 = F.values[nstates]
        if length(F.values)>=(nstates+1)
            vals2 = F.values[nstates + 1]
            if (vals2-vals1)/(vals1-vals0) < 5 
                println("There is no gap separating 2q states from higher manifolds")
            end
        end
    end
    # further downprojection for Hamiltonian
    Unew = F.vectors[:,1:nstates]
    Hnew = Unew' * (U * H * U') * Unew
    ϵ = eigvals(Hermitian(Hnew))
    return ϵ
end

function initHofstadterHoppingElements(q::Int,p::Int,params::Params;ndim::Int=11,lkmin::Int=32)

    hof = HofstadterHoppingElements()
    hof.q = q 
    hof.p = p

    n1q = lkmin ÷ (2q)
    if n1q > 0  # large-field regime
        hof.l1 = 2q * n1q 
        hof.δq1 = n1q * hof.p
        hof.l2 = q * n1q
        hof.δq2 =  n1q * hof.p 
    else    # low field regime 
        n1q = 1
        hof.l1 = 2q *n1q 
        hof.δq1 = n1q * hof.p
        hof.l2 = q * n1q # consider 1 of k2 values within a strip  (extend one Moire Brillouin zone away from first)
        hof.δq2 = n1q * hof.p
    end
    hof.k1 = collect(0:(hof.l1-1)) ./ hof.l1 
    hof.k2 = collect(0:(hof.l2-1)) ./ hof.l2 
    hof.kvec = reshape(hof.k1,:,1).*params.g1 .+ reshape(hof.k2,1,:).*params.g2
    hof.ndim = ndim
    # hof.nvec = collect( (-(ndim-1)÷2):((ndim-1)÷2) )
    hof.nvec = collect(0:(ndim-1))

   
    # BM Hamiltonian basics
    blk = ConstructHBM_Hoftstadter(hof,params)
    # Hybrid Wannier basics
    basis = ConstructHybridWannier_Hofstadter(hof,blk)
    
    # 
    ComputeOverlapMatrix(hof,basis,blk,params)

    # 
    hof.spectrum = orthonormalizationProcedure(hof,blk,params)
    # hof.spectrum = orthonormalizationProcedurev2(hof,blk,params)

    return hof,blk,basis
end

function ConstructHBM_Hoftstadter(hof::HofstadterHoppingElements,params::Params;lg::Int=9)
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

function ConstructHybridWannier_Hofstadter(hof::HofstadterHoppingElements,blk::HBM)
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


function orthonormalizationProcedurev2(hof::HofstadterHoppingElements,blk::HBM,params::Params)
    n0dim = size(hof.M,3)
    M = reshape(hof.M,n0dim*2hof.l2,n0dim*2hof.l2)
    H2 = reshape(hof.Hsq,n0dim*2hof.l2,n0dim*2hof.l2)
    H = reshape(hof.H,n0dim*2hof.l2,n0dim*2hof.l2)
    M = M + M' 
    H2 = H2 + H2'
    H = H + H'
    
    ϵ0 = maximum(blk.Hk) / (2π * params.vf/abs(params.a1))

    # how many eigenvectors to keep
    F = eigen(Hermitian(M))
    nstates = 2hof.l2

    # stage 1 using H^2 to filter out states
    vec = F.vectors'
    vals = 1 ./ sqrt.(F.values[(end-nstates+1):end]) 
    U = Diagonal(vals) * vec[(end-nstates+1):end,:]
    Hnew = U * H * U'
    ϵ = eigvals(Hermitian(Hnew))
    return ϵ
end