include("hybridWannier_mod.jl")
using LinearAlgebra
using TensorOperations

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
    
    # Strong coupling energetics
    gq::Array{Complex{Int},2}  # m g1 + n g2/q; store m + i n
    OΨ::Array{ComplexF64,6}            # (j1 k1,k2) x (j2 p1 p2) x gq1 x gq2
    
    #
    Σz::Array{ComplexF64,3} # (γr) x (γr) x k1 k2 # eigenvalues of σz μ_0
    H::Array{ComplexF64,3} # (γr) x (γr) x k1 k2 -- for every k in magnetic strip, γr x γr matrix
 
    HofstadterVq() = new()

end

function initHofstadterVq(A::HofstadterVq, params::Params;p::Int=1,q::Int=5,lk::Int=5)
    
    # A = HofstadterVq()
    ϕ = p//q 

    A.p = numerator(ϕ)
    A.q = denominator(ϕ)
    A.lk = lk 
    A.g1 = params.g1 
    A.g2 = params.g2
    A.nvec = collect(0:1)
    A.svec = collect(-2:2)
    
    gc = 1
    A.δg = reshape((-gc):(gc),:,1) .+ 1im * reshape((-gc):(gc),1,:)
    A.gq = reshape(-gc:gc,:,1) .+ 1im * reshape((-gc*A.q):(gc*A.q),1,:)

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
    k2mbz = reshape( reshape(A.k2,:,1) .+ reshape(A.r ./A.q,1,:) , : )
    A.kvec = reshape(A.k1,:,1) * A.g1 .+ reshape(k2mbz,1,:) * A.g2

    # --------- Bloch part of the story ------- # 
    blk = ConstructHBM_Hoftstadter(A,params)
    basis = ConstructHybridWannier_Hofstadter(A,blk)
    computeM(A, blk,basis)
    GC.gc()

    # --------- Overlap matrix and orthonormalization ---------- # 
    computeOverlapMatrix(A)
    computeOrtNormMTG(A)
    # A.O, A.Oz, A.Uort

    # --------- Matrix elements at strong coupling ------- # 
    computeCoulombOverlap(A)  # Otrial, and Uort'*Otrial*Uort
    computeSingleParticleSpectrum(A) # A.H

    return nothing
end

@inline function Coulomb(q::ComplexF64)
    return abs(q)>1e-5 ? 4π/abs(q) : 0.0
end

function computeM(A::HofstadterVq,blk::HBM,basis::HybridWannier)
    # Given hybrid wannier states from blk.Uγk[4lg^2,2,lk1,lk2], compute overlap matrix 
    # M(γ1k,γ2p,q) = U_g(γ1k)^† U_g+δg(γ2p)

    Uγk = reshape(basis.WγLS,4blk.lg^2,2A.l1*A.l2*A.q)
    A.M = zeros(ComplexF64,2A.l1*A.l2*A.q,2A.l1*A.l2*A.q,size(A.δg,1),size(A.δg,2))
    A.Mz = zeros(ComplexF64,2A.l1*A.l2*A.q,2A.l1*A.l2*A.q,size(A.δg,1),size(A.δg,2))
    Oσ = kron(Array{ComplexF64}(I,blk.lg^2,blk.lg^2),kron(ComplexF64[1 0;0 1],ComplexF64[1 0;0 -1]))
    for ig in CartesianIndices(A.δg)
        Uγp = reshape(Uγk,4,blk.lg,blk.lg,:)
        Uγp = circshift(Uγp,(0,-real(A.δg[ig]),-imag(A.δg[ig]),0))
        Uγp = reshape(Uγp,4blk.lg^2,:)
        A.M[:,:,ig[1],ig[2]] = Uγk' * Uγp
        A.Mz[:,:,ig[1],ig[2]] = Uγk' * Oσ * Uγp
    end
    return nothing
end

function computeOverlapMatrix(A::HofstadterVq)
    ρkp = reshape(A.M,2,A.l1,A.l2,A.q,2,A.l1,A.l2,A.q,size(A.δg,1),size(A.δg,2))
    ρzkp = reshape(A.Mz,2,A.l1,A.l2,A.q,2,A.l1,A.l2,A.q,size(A.δg,1),size(A.δg,2))

    A.O = zeros(ComplexF64,2A.q*length(A.nvec),2A.q*length(A.nvec),A.l1*A.l2)
    A.Oz = zeros(ComplexF64,2A.q*length(A.nvec),2A.q*length(A.nvec),A.l1*A.l2)
    
    @inline function idx_k2(ik::Int)
        return mod(ik-1,A.l2) + 1
    end
    @inline function idx_r(ik::Int)
        return (ik-1)÷A.l2 + 1
    end
    
    O0 = zeros(ComplexF64,A.l1,A.l1,length(A.svec),2,length(A.r),2,length(A.r),A.l2)  # each kr uniquely associated with one kc
    Oz = zeros(ComplexF64,A.l1,A.l1,length(A.svec),2,length(A.r),2,length(A.r),A.l2) 

    for s1 in eachindex(A.svec)
        qϕ1 = A.svec[s1] * A.p *A.nq
        tmp = eachindex(A.k1) .- qϕ1   # split into [k1-q1] + δg
        k1c = mod.(tmp .-1,A.l1) .+ 1
        δg1 = (tmp .- k1c) .÷ A.l1
        
        qϕ2 = A.svec[s1] * A.p
        tmp = eachindex(A.r) .- qϕ2   # split into [k2-q2] + δg
        k2c = mod.(tmp .-1,A.q) .+ 1
        δg2 = (tmp .- k2c) .÷ (A.q)

        δg = reshape(δg1,:,1) .+ 1im * reshape(δg2,1,:) 

        for i2r in eachindex(A.r)
            i2c = k2c[i2r]
            for i1r in eachindex(A.k1) 
                i1c = k1c[i1r]
                ig = findfirst(x->x==δg[i1r,i2r],A.δg)
                if isnothing(ig)
                    println("Err: no index of ig found in computeOverlapMatrix()")
                end
                for ik2 in eachindex(A.k2)
                    O0[i1r,i1c,s1,:,i2r,:,i2c,ik2] = ρkp[:,i1r,ik2,i2r,:,i1c,ik2,i2c,ig[1],ig[2]]./A.l1
                    Oz[i1r,i1c,s1,:,i2r,:,i2c,ik2] = ρzkp[:,i1r,ik2,i2r,:,i1c,ik2,i2c,ig[1],ig[2]]./A.l1
                end
            end
        end
    end

    # 
    k1 = reshape(A.k1,1,1,1,1,1,1,1,1,1,:,1)
    svec = reshape(A.svec,1,1,:,1,1,1,1,1,1,1,1)
    n1vec = reshape(A.nvec,1,1,1,1,1,:,1,1,1,1,1)
    n2vec = reshape(A.nvec,1,1,1,1,1,1,1,1,:,1,1)
    k1bar = reshape(A.k1,:,1,1,1,1,1,1,1,1,1,1)
    p1bar = reshape(A.k1,1,:,1,1,1,1,1,1,1,1,1)
    O0 = reshape(O0,A.l1,A.l1,length(A.svec),2,length(A.r),1,2,length(A.r),1,1,A.l2) 
    Oz = reshape(Oz,A.l1,A.l1,length(A.svec),2,length(A.r),1,2,length(A.r),1,1,A.l2) 

    A.O .= reshape(
                sum( ( @. exp(1im * 2π * k1 * svec) * exp(-1im *π * svec * (svec-1) * A.p/(2A.q)) *
                exp(1im * 2π * k1bar * n1vec) * exp(-1im * 2π * p1bar * (n2vec + svec)) * O0 ) , dims = (1,2,3) ) , 
                (2A.q*length(A.nvec),2A.q*length(A.nvec),A.l1*A.l2) )

    A.Oz .= reshape(
        sum( ( @. exp(1im * 2π * k1 * svec) * exp(-1im *π * svec * (svec-1) * A.p/(2A.q)) *
        exp(1im * 2π * k1bar * n1vec) * exp(-1im * 2π * p1bar * (n2vec + svec)) * Oz ) , dims = (1,2,3) ) , 
        (2A.q*length(A.nvec),2A.q*length(A.nvec),A.l1*A.l2) )

    return nothing
end

function computeOrtNormMTG(A::HofstadterVq)
    # |Ψ⟩ = |T⟩ Uort for any k1, k2 in magnetic Brillouin zone
    A.Uort = zeros(ComplexF64,2A.q*length(A.nvec),2A.q,A.l1*A.l2)
    A.Σz = zeros(ComplexF64,2A.q,2A.q,A.l1*A.l2)
    for ik in 1:A.l2*A.l1
        O = view(A.O,:,:,ik)
        F = eigen(Hermitian(O))
        A.Uort[:,:,ik] = ( @view F.vectors[:,(end-2A.q+1):end]) * 
                            Diagonal( 1 ./sqrt.( F.values[(end-2A.q+1):end] ) )

        Oz = view(A.Oz,:,:,ik)
        Uort = view(A.Uort,:,:,ik)
        A.Σz[:,:,ik] = Uort' * Oz * Uort
    end
    return nothing
end

function computeCoulombOverlap(A::HofstadterVq)
    # m,n are indices of gq
    # Given A.M, construct the overlap matrix ⟨T|e^{iqr}|T⟩
    ρkp = reshape(A.M,2,A.l1,A.l2,A.q,2,A.l1,A.l2,A.q,size(A.δg,1),size(A.δg,2))
    ρkp = permutedims(ρkp,[2,6,9,10,1,4,3,5,8,7])  # k1bar, p1bar, δg1, δg2, γ1, r1, k2, γ2, r2, p2
    A.OΨ = zeros(ComplexF64,2A.q,A.l1*A.l2,2A.q,A.l1*A.l2,size(A.gq,1),size(A.gq,2))

    # ---- main routine ---- #
    Ot = zeros(ComplexF64,2,A.q,length(A.nvec),A.l1,A.l2,2,A.q,length(A.nvec),A.l1,A.l2)
    for ig in  CartesianIndices(A.gq)
        Ot .= 0.0im + 0.0
        m,n = ig[1],ig[2]
        println(m," ",n)
        computeOt(Ot,m,n,ρkp,A) # compute Ot for any given gq
        Ot_reshaped = reshape(Ot,(2A.q*length(A.nvec),A.l1*A.l2,2A.q*length(A.nvec),A.l1*A.l2))
        for ip in 1:A.l1*A.l2, ik in 1:A.l1*A.l2
            Uort_k = view(A.Uort,:,:,ik)
            Uort_p = view(A.Uort,:,:,ip)
            A.OΨ[:,ik,:,ip,m,n] = Uort_k' * view(Ot_reshaped,:,ik,:,ip) * Uort_p
        end
    end
    return nothing
end

function computeOt(Ot::Array{ComplexF64,10},m::Int,n::Int,ρkp::Array{ComplexF64,10},A::HofstadterVq)

    δconstraint = zeros(Int,A.l1,A.l1,size(A.δg,1),size(A.δg,2),A.q,A.q,length(A.svec),A.l1,A.l1)
    get_δconstraint(δconstraint,m,n,A)

    # this is summed over k1bar, p1bar, a, and b; 
    tmpOt = zeros(ComplexF64,2,A.q,A.l2,2,A.q,A.l2,length(A.nvec),length(A.nvec),A.l1,A.l1)

    for ip1 in 1:A.l1, ik1 in 1:A.l1,s1 in eachindex(A.svec)
        δconst = reshape( view(δconstraint,:,:,:,:,:,:,s1,ik1,ip1), (A.l1,A.l1,size(A.δg,1),size(A.δg,2),1,A.q,1,1,A.q,1) ) 
        θ1 = exp(1im * 2π * A.k1[ip1] * A.svec[s1]) * exp(-1im *π * A.svec[s1]*(A.svec[s1]-1) * A.p/(2A.q)) / A.l1
        for n2 in eachindex(A.nvec), n1 in eachindex(A.nvec)
            θtot = reshape( exp.(1im * 2π * A.k1 * A.nvec[n1]) .* exp.(-1im * 2π * A.k1' * (A.nvec[n2]+A.svec[s1])) * θ1, 
                                 (A.l1,A.l1,1,1,1,1,1,1,1,1) )
            tmpOt[:,:,:,:,:,:,n1,n2,ik1,ip1] += reshape( sum(θtot .* δconst .* ρkp,dims=(1,2,3,4)), 
                                                    (2,A.q,A.l2,2,A.q,A.l2) )
        end
    end

    # permutedims 
    Ot .= permutedims(tmpOt,[1,2,7,9,3,4,5,8,10,6])
    GC.gc()
    return nothing
end

@inline function get_δconstraint1(m::Int,n::Int,ik1::Int,ip1::Int,s1::Int,A::HofstadterVq)
    gc = (size(A.δg,1)-1) ÷2 
    ik1bar = collect(1:A.l1)
    tmp = @. ik1bar - (ik1 - ip1 - real(A.gq[m,n])*A.l1) - s1*A.p *A.nq
    ip1bar = @. mod(tmp - 1,A.l1) + 1 
    a = @. (tmp - ip1bar) ÷ A.l1 + gc+1 # +gc+1 is because δg runs from -gc:gc 
    return ik1bar,ip1bar, a
end

@inline function get_δconstraint2(m::Int,n::Int,ik1::Int,ip1::Int,s1::Int,A::HofstadterVq)
    gc = (size(A.δg,1)-1) ÷2 
    ir1 = collect(1:A.q)
    tmp = @. ir1 + imag(A.gq[m,n])- s1*A.p 
    ir2 = @. mod(tmp -1, A.q) + 1
    b = @. (tmp - ir2) ÷ A.q + gc + 1 # +gc+1 is because δg runs from -gc:gc 
    return ir1, ir2, b
end

@inline function get_δconstraint(δconstraint::Array{Int,9},m::Int,n::Int,A::HofstadterVq)
    δconstraint .= 0
    for s1 in eachindex(A.svec)
        for ip1 in 1:A.l1, ik1 in 1:A.l1
            ik1bar, ip1bar, a = get_δconstraint1(m,n,ik1,ip1,A.svec[s1],A)
            ir1, ir2, b = get_δconstraint2(m,n,ik1,ip1,A.svec[s1],A)
            for j1 in eachindex(ir1), j2 in eachindex(ik1bar) 
                if (a[j2] in 1:size(A.δg,1)) && (b[j1] in 1:size(A.δg,2))
                        δconstraint[ik1bar[j2],ip1bar[j2],a[j2],b[j1],ir1[j1],ir2[j1],s1,ik1,ip1] = 1
                end
            end
        end
    end
    return nothing
end

function computeCoulombOverlapv0(A::HofstadterVq)
    # m,n are indices of gq
    # Given A.M, construct the overlap matrix ⟨T|e^{iqr}|T⟩
    gc = (size(A.δg,1) -1 )÷2
    ρkp = reshape(A.M,2,A.l1,A.l2,A.q,2,A.l1,A.l2,A.q,size(A.δg,1),size(A.δg,2))
    A.OΨ = zeros(ComplexF64,2A.q,A.l1*A.l2,2A.q,A.l1*A.l2,size(A.gq,1),size(A.gq,2))
    # Too big of a matrix!
    # TempMat = zeros(ComplexF64,A.l1,A.l1,size(A.δg,1),size(A.δg,2),
    #                     2,A.q,length(A.nvec),A.l1,A.l2,2,A.q,length(A.nvec),A.l1,A.l2,size(A.gq,1),size(A.gq2))
    
    @inline function get_δconstraints1(m::Int,n::Int,ik1::Int,ip1::Int,s1::Int)
        ik1bar = collect(1:A.l1)
        tmp = @. ik1bar - (ik1 - ip1 - real(A.gq[m,n])*A.l1) - s1*A.p *A.nq
        ip1bar = @. mod(tmp - 1,A.l1) + 1 
        a = @. (tmp - ip1bar) ÷ A.l1 + gc+1 # +gc+1 is because δg runs from -gc:gc 
        return ik1bar,ip1bar, a
    end

    @inline function get_δconstraints2(m::Int,n::Int,ik1::Int,ip1::Int,s1::Int)
        ir1 = collect(1:A.q)
        tmp = @. ir1 + imag(A.gq[m,n])- s1*A.p 
        ir2 = @. mod(tmp -1, A.q) + 1
        b = @. (tmp - ir2) ÷ A.q + gc + 1 # +gc+1 is because δg runs from -gc:gc 
        return ir1, ir2, b
    end

    @inline function get_δconstraint(δconstraint::Array{Float64,9},m::Int,n::Int)
        δconstraint .= 0.0 
        for s1 in eachindex(A.svec)
            for ip1 in 1:A.l1, ik1 in 1:A.l1
                ik1bar, ip1bar, a = get_δconstraints1(m,n,ik1,ip1,A.svec[s1])
                ir1, ir2, b = get_δconstraints2(m,n,ik1,ip1,A.svec[s1])
                for j1 in eachindex(ir1), j2 in eachindex(ik1bar) 
                    δconstraint[ir1[j1],ik1bar[j2],ir2[j1],ip1bar[j2],a[j2],b[j1],ik1,ip1,s1] = 1.0 
                end
            end
        end
        return nothing
    end
    
    # ---- main routine ---- #
    Ot = zeros(ComplexF64,2,A.q,length(A.nvec),A.l1,A.l2,2,A.q,length(A.nvec),A.l1,A.l2)
    δconstraint = zeros(Float64,A.q,A.l1,A.q,A.l1,size(A.δg,1),size(A.δg,2),A.l1,A.l1,length(A.svec))  # constraints on r1, k1bar, r2, k2bar, a, b 
    svec = reshape(A.svec,1,:)
    k1vec = reshape(A.k1,:,1)
    θk1 = reshape( exp.(1im *2π * k1vec .* svec) .* exp.(-1im * (π * A.p/(2A.q)) .* svec .*(svec .-1)), 
                    (1,1,1,1,1,1,1,1,1,1,A.l1,1,length(A.svec)) ) 
    kernel = reshape(ρkp,2,A.l1,A.l2,A.q,2,A.l1,A.l2,A.q,size(A.δg,1),size(A.δg,2),1,1,1)
    for ig in  CartesianIndices(A.gq)
        Ot .= 0.0im + 0.0
        m,n = ig[1],ig[2]
        println(m," ",n)
        get_δconstraint(δconstraint,m,n)
        for n2 in eachindex(A.nvec), n1 in eachindex(A.nvec)
            k1bar = reshape(A.k1,:,1,1)
            p1bar = reshape(A.k1,1,:,1)
            svec = reshape(A.svec,1,1,:)
            θkpbar = reshape(exp.(1im * 2π  * A.nvec[n1] * k1bar .- 1im * 2π * p1bar .* (svec .+ A.nvec[n2])),
                                (1,A.l1,1,1,1,A.l1,1,1,1,1,1,1,length(A.svec)) )
                
            tmp = reshape( sum( (θk1 .* θkpbar) .* kernel .* 
                    reshape(δconstraint,1,A.l1,1,A.q,1,A.l1,1,A.q,size(A.δg,1),size(A.δg,2),A.l1,A.l1,length(A.svec)), dim=(2,6,9,10) ),
                    (2,A.l2,A.q,2,A.l2,A.q,A.l1,A.l1) )
            Ot[:,:,n1,:,:,:,:,n2,:,:] .= permutedims(tmp,[1,3,7,2,4,6,8,5])
        end
        Ot_reshaped = reshape(Ot,(2A.q*length(A.nvec),A.l1*A.l2,2A.q*length(A.nvec),A.l1*A.l2))
        for ip in 1:A.l1*A.l2, ik in 1:A.l1*A.l2
            Uort_k = view(A.Uort,:,:,ik)
            Uort_p = view(A.Uort,:,:,ip)
            A.OΨ[:,ik,:,ip,m,n] = Uort_k' * view(Ot_reshaped,:,ik,:,ip) * Uort_p
        end
    end

    # for ig in CartesianIndices(A.gq)
    #     Ot .= 0.0im + 0.0
    #     m,n = ig[1],ig[2]
    #     println(m," ",n)
    #     for s1 in eachindex(A.svec)
    #         for ip1 in 1:A.l1, ik1 in 1:A.l1
    #             ik1bar, ip1bar, a = get_δconstraints1(m,n,ik1,ip1,A.svec[s1])
    #             ir1, ir2, b = get_δconstraints2(m,n,ik1,ip1,A.svec[s1])
    #             for jr1 in ir1, jk1bar in ik1bar 
    #                 tmp = view(ρkp,:,jk1bar,:,jr1,:,ip1bar[jk1bar],:,ir2[jr1],a[jk1bar],b[jr1])
    #                 for n2 in eachindex(A.nvec), n1 in eachindex(A.nvec)
    #                     expθ = exp(1im * 2π * A.k1[jk1bar] * A.nvec[n1]) * 
    #                             exp( - 1im * 2π * A.k1[ip1bar[jk1bar]] * (A.nvec[n2]+A.svec[s1]))  * 
    #                             exp(1im * A.k1[ik1] * A.svec[s1]) / A.l1
    #                     Ot[:,jr1,n1,ik1,:,:,ir2[jr1],n2,ip1,:] .+= tmp .* expθ 
    #                 end 
    #             end
    #         end
    #     end
    #     Ot_reshaped = reshape(Ot,(2A.q*length(A.nvec),A.l1*A.l2,2A.q*length(A.nvec),A.l1*A.l2))
        
    #     for ip in 1:A.l1*A.l2, ik in 1:A.l1*A.l2
    #         Uort_k = view(A.Uort,:,:,ik)
    #         Uort_p = view(A.Uort,:,:,ip)
    #         A.OΨ[:,ik,:,ip,m,n] = Uort_k' * view(Ot_reshaped,:,ik,:,ip) * Uort_p
    #     end
    # end

    return nothing
end


function computeSingleParticleSpectrum(A::HofstadterVq)
    # kvec in magnetic strip
    kvec = reshape(A.k1,:,1) * params.g1 .+ reshape(A.k2,1,:) * params.g2
    A.H = zeros(ComplexF64,2A.q,2A.q,A.l1*A.l2)
    gq = real(A.gq) * A.g1 + imag(A.gq) * A.g2 / A.q
    Vq = Coulomb.( reshape(kvec,1,1,:,1,1,1,1) .- reshape(kvec,1,1,1,1,:,1,1) .- 
                    reshape(gq,1,1,1,1,1,size(gq,1),size(gq,2)) )
    OΨ1 = reshape(A.OΨ,2A.q,1,A.l1*A.l2,2A.q,A.l1*A.l2,size(gq,1),size(gq,2))
    OΨ2 = reshape(A.OΨ,1,2A.q,A.l1*A.l2,2A.q,A.l1*A.l2,size(gq,1),size(gq,2))
    
    A.H .= reshape( sum(Vq .* OΨ1 .* conj.(OΨ2), dims=(4,5,6,7)), (2A.q, 2A.q, A.l1*A.l2))

    Lm = (4π)/(sqrt(3)*abs(A.g1))
    V0 = 1/Lm
    A.H .*= (0.5/(sqrt(3)*Lm^2*A.l1^2) / V0)
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