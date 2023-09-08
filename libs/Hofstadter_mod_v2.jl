include("hybridWannier_mod.jl")

mutable struct HofstadterHoppingElements
    # This data structure stores Hn for a few nearest neighbors 
    ndim::Int
    nvec::Vector{Int} # <0| ... | n > 

    q::Int
    p::Vector{Int}
    l1::Int    # grid along g1 (0...l1-1)/l1
    l2::Int    # grid along g2 (0...l2-1)/l2
    δq1::Vector{Int}   # steps for p/(2q) along g1 
    δq2::Vector{Int}   # steps for p/q along g2

    k1::Vector{Float64}
    k2::Vector{Float64}
    kvec::Matrix{ComplexF64}

    O1::Array{ComplexF64,6}  # exp(iqϕ⋅r)*H0
    O2::Array{ComplexF64,6} # wavefunction overlap 
    O3::Array{ComplexF64,6} # exp(iqϕ⋅r)(x-nL1x/2)

    Term1::Array{ComplexF64,6} 
    Term2::Array{ComplexF64,6} 
    M::Array{ComplexF64,6}  # overlap matrix for ⟨γ,k1,k2,n0|γ',p1,p2,n1⟩

    Spectrum::Array{Float64,3}
    Ham::Array{ComplexF64,4}  # Hamiltonian in orthonormalized basis
    Vec::Array{ComplexF64,4}  # Eigenvector in orthonormalized basis
    D::Array{Float64,3}

    HofstadterHoppingElements() = new()
end

function initHofstadterHoppingElements(q::Int,params::Params;ndim::Int=7,lkmin::Int=32)

    hof = HofstadterHoppingElements()
    hof.q = q 
    hof.p = collect(0:q)

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
    hof.nvec = collect( (-(ndim-1)÷2):((ndim-1)÷2) )

    # Nk = 1 # only calculate a few select k1s
    # k1s = rand(1:hof.l1,Nk)
    # k1s = [14]
    k1s = eachindex(hof.k1)


    ## basis guiding center 
    guideR = collect(0:1)
    dimR = length(guideR)
    # BM Hamiltonian basics
    blk = ConstructHBM_Hoftstadter(hof,params)
    # Hybrid Wannier basics
    basis = ConstructHybridWannier_Hofstadter(hof,blk)
    
    lg = blk.lg
    Ig = Array{ComplexF64}(I,lg^2,lg^2)
    O_LS = kron(Ig,kron(ComplexF64[1 0;0 1],ComplexF64[0 -1im;1im 0]) )
    ϵ0 = 2π * params.vf / abs(params.a1) * (1/q)   # energy scale, need to x p[ip] later

    # -----------------------   Term 1 and (-x0) part of Term 2 --------------------------- #
    # find connected pairs, k' - k + qϕ = s g
    # g1: k1' = k1 - qϕ1 + s
    # g2: k2' = k2 - qϕ2 + s

    hof.O1 = zeros(ComplexF64,2*hof.l2,dimR,2*hof.l2,dimR,hof.ndim,length(hof.p)) # term 1 at n
    hof.O2 = zeros(ComplexF64,2*hof.l2,dimR,2*hof.l2,dimR,hof.ndim,length(hof.p)) # overlap matrix at n
    hof.O3 = zeros(ComplexF64,2*hof.l2,dimR,2*hof.l2,dimR,hof.ndim,length(hof.p)) # term 2 at n


    for ip in eachindex(hof.p)
        for n in eachindex(hof.nvec)
            x0 = hof.nvec[n]
            qϕ1 = hof.nvec[n] * hof.δq1[ip]
            tmp = eachindex(hof.k1) .- qϕ1   # split into [k1-q1] + δg
            k1c = mod.(tmp .-1,hof.l1) .+ 1
            δg1 = (tmp .- k1c) .÷ hof.l1
            
            qϕ2 = hof.nvec[n] * hof.δq2[ip]
            tmp = eachindex(hof.k2) .- qϕ2   # split into [k1-q1] + δg
            k2c = mod.(tmp .-1,hof.l2) .+ 1
            δg2 = (tmp .- k2c) .÷ hof.l2
            
            # sum over |γ;k1⟩ projector 
            for i2r in eachindex(hof.k2)
                i2c = k2c[i2r]
                for i1r in eachindex(hof.k1) 
                    i1c = k1c[i1r]
                    ur = view(basis.WγLS,:,:,i1r,i2r) 
                    uc = view(basis.WγLS,:,:,i1c,i2c)
                    uc = reshape(uc,4,blk.lg,blk.lg,2)
                    uc = reshape(circshift(uc,(0,-δg1[i1r],-δg2[i2r],0)),4*blk.lg^2,2)
                    H2x2 = basis.Hγk[:,:,i1c,i2c]
                    λrc = ur' * uc
                    λrc1 = ur' * O_LS * uc

                    for nr in eachindex(guideR)
                        for nc in eachindex(guideR)
                            # Hamiltonian overlap first term
                            hof.O1[(2i2r-1):(2i2r),nr,(2i2c-1):(2i2c),nc,n,ip] = hof.O1[(2i2r-1):(2i2r),nr,(2i2c-1):(2i2c),nc,n,ip] + 
                                        λrc *H2x2/ hof.l1 *
                                        exp(-1im * 2π * hof.k1[i1c] * (hof.nvec[n] + guideR[nc]) )  * 
                                        exp(1im * 2π * hof.k1[i1r] * guideR[nr])
                            
                            # wavefunction overlap
                            hof.O2[(2i2r-1):(2i2r),nr,(2i2c-1):(2i2c),nc,n,ip] = hof.O2[(2i2r-1):(2i2r),nr,(2i2c-1):(2i2c),nc,n,ip] + 
                                        λrc / hof.l1 *
                                        exp(-1im * 2π * hof.k1[i1c] * (hof.nvec[n]+ guideR[nc]) ) * 
                                        exp(1im * 2π * hof.k1[i1r] * guideR[nr] )
                            
                            # Term 2 first contribution
                            hof.O3[(2i2r-1):(2i2r),nr,(2i2c-1):(2i2c),nc,n,ip] = hof.O3[(2i2r-1):(2i2r),nr,(2i2c-1):(2i2c),nc,n,ip] + 
                                        (-ϵ0 * x0 * hof.p[ip] ) * λrc1 / hof.l1 *
                                        exp(-1im * 2π * hof.k1[i1c] * (hof.nvec[n]+ guideR[nc]) ) * 
                                        exp(1im * 2π * hof.k1[i1r] * guideR[nr] )
                        end
                    end
                end
            end
        end
    end

    hof.Term1 = zeros(ComplexF64,2*hof.l2,dimR,2*hof.l2,dimR,length(k1s),length(hof.p)) # 10 is a few select indices 
    hof.M = zeros(ComplexF64,2*hof.l2,dimR,2*hof.l2,dimR,length(k1s),length(hof.p)) # last is k1
    for ip in eachindex(hof.p)
        for m in eachindex(k1s)
            for n in eachindex(hof.nvec)
                hof.Term1[:,:,:,:,m,ip] = hof.Term1[:,:,:,:,m,ip] + view(hof.O1,:,:,:,:,n,ip)*
                                        exp(-1im *π * hof.nvec[n]*(hof.nvec[n]-1) * hof.p[ip]/(2q)) * 
                                        exp( 1im * 2π * hof.nvec[n] * hof.k1[k1s[m]])
                hof.M[:,:,:,:,m,ip] = hof.M[:,:,:,:,m,ip] + view(hof.O2,:,:,:,:,n,ip)*
                                        exp(-1im *π * hof.nvec[n]*(hof.nvec[n]-1) * hof.p[ip]/(2q)) * 
                                        exp( 1im * 2π * hof.nvec[n] * hof.k1[k1s[m]])
            end
        end
    end

    println("Stage 1 Complete")
    # -------------------------- Dense matrix part of Term 2 ------------------------ #
    gq = collect((-(lg-1)÷2):((lg-1)÷2))
    # this is kr - kc
    q1vals = reshape(hof.k1,:,1) .- reshape(hof.k1,1,:)

    for ip in eachindex(hof.p)
        for n in eachindex(hof.nvec)
            qϕ2 = hof.nvec[n] * hof.δq2[ip]
            tmp = eachindex(hof.k2) .- qϕ2   # split into [k1-q1] + δg
            k2c = mod.(tmp .-1,hof.l2) .+ 1
            δg2 = (tmp .- k2c) .÷ hof.l2

            for i2r in eachindex(hof.k2)
                i2c = k2c[i2r]
                tmp = zeros(ComplexF64,2,hof.l1,2,hof.l1)
                for iq in eachindex(gq)
                    qn1 = q1vals .- gq[iq] .- hof.nvec[n]* hof.p[ip]/(2q)
                    Vqvals = MagneticGuidingCenter.(qn1,hof.l1) .* (ϵ0 * hof.p[ip])
                    Vqvals = reshape(Vqvals,1,hof.l1,1,hof.l1)
                    
                    ur = reshape(view(basis.WγLS,:,:,:,i2r),4*blk.lg^2,2*hof.l1)
                    uc = view(basis.WγLS,:,:,:,i2c)
                    uc = reshape(uc,4,blk.lg,blk.lg,2*hof.l1)
                    uc = reshape(circshift(uc,(0,-gq[iq],-δg2[i2r],0)),4*blk.lg^2,2*hof.l1)
                    λrc = ur' * O_LS * uc
                    λrc = reshape(λrc,2,hof.l1,2,hof.l1)

                    tmp .= tmp + λrc.*Vqvals
                end

                for nr in eachindex(guideR)
                    for nc in eachindex(guideR)
                        θr = reshape(exp.(1im * 2π * hof.k1 * guideR[nr]) ,1,hof.l1,1,1)
                        θc = reshape(exp.(-1im * 2π * hof.k1 * (guideR[nc]+hof.nvec[n])) ,1,1,1,hof.l1)
                        λrc = sum( θr.*tmp.*θc ,dims=(2,4)) / hof.l1

                        hof.O3[(2i2r-1):(2i2r),nr,(2i2c-1):(2i2c),nc,n,ip] =  
                                    hof.O3[(2i2r-1):(2i2r),nr,(2i2c-1):(2i2c),nc,n,ip] + reshape(λrc,2,2)  
                    end
                end
            end
        end
    end
    
    hof.Term2 = zeros(ComplexF64,2*hof.l2,dimR,2*hof.l2,dimR,length(k1s),length(hof.p))   # this is 8GB !..
    for ip in eachindex(hof.p)
        for m in eachindex(k1s)
            for n in eachindex(hof.nvec)
                hof.Term2[:,:,:,:,m,ip] = hof.Term2[:,:,:,:,m,ip] + view(hof.O3,:,:,:,:,n,ip)*
                                        exp(-1im *π * hof.nvec[n]*(hof.nvec[n]-1) * hof.p[ip]/(2q)) * 
                                        exp( 1im * 2π * hof.nvec[n] * hof.k1[k1s[m]])
            end
        end
    end

    println("Stage 2 Complete")

    # ---------------------- Basis orthonormalization and Full Hamiltonian -------------- # 
    
    # hof.Ham = zeros(ComplexF64,2*hof.l2,2*hof.l2,length(k1s),length(hof.p))
    # hof.Vec = zeros(ComplexF64,2*hof.l2,4*hof.l2,length(k1s),length(hof.p))  # orthonormalized eigenvectors
    # hof.D = zeros(Float64,2*hof.l2,length(k1s),length(hof.p))
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
            # if (ip<8)
            #     F.values[F.values .< 0.2] .= Inf
            # else
            #     F.values[F.values .< 0.05] .= Inf
            # end
            F.values[1:(end-2*hof.l2)] .= Inf
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

    # ------------------------  End ---------------------------- #

    return hof,blk,basis
end

@inline function MagneticGuidingCenter(q1::Float64,N1::Int)
    # q1, x0, a are all in absolute physical units 
    # remember to multiply the overall normalization factor of   a^2/(N L1x)
    # N is 2q in this case
    if abs(q1) < 1e-5 
        return 0.0im
    else
        return  1im/(2π*q1) * cos(π*q1*N1)
    end
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
        vals, vecs = eigs(H,nev=2,which=:SM)
        # C2T
        vecs = vecs + blk.C2T*conj(vecs)
        for i in 1:2
            tmp = view(vecs,:,i)
            normalize!(tmp)
        end

        if (norm(imag(vals))<1e-6)
            perm = sortperm(real(vals[:]))
            blk.Uk[:,(2ik-1):(2ik)] = view(vecs,:,perm)
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