include("Parameters_mod.jl")
using ClassicalOrthogonalPolynomials
using SpecialFunctions
using LinearAlgebra

mutable struct HofstadterVqLL
    # this implements 1/q sequence LL with strain
    p::Int 
    q::Int 
    nLL::Int   # 0,1,2... 
    nγ::Int    # 2, particle-hole symmetry 
    nH::Int    # nLL*nγ - 1
    lk::Int
    k1::Vector{Float64} 
    k2::Vector{Float64}
    kvec::Matrix{ComplexF64}
    l1::Int
    l2::Int 
    nq::Int

    svec::Vector{Int}
    nvec::Vector{Int}   # integers along g1 direction 

    lB::Float64  # magnetic length in absolute units 
    Λw::Array{ComplexF64,4}  # 2nH x 2nH x p for fixed s1
    ΛΨ::Array{ComplexF64,4} # 2q x 2q x p for fixed s1

    ρq::Array{ComplexF64,2}
    ν::Vector{Int}

    H::Array{ComplexF64,3}  # 2q x 2q 

    HofstadterVqLL() = new()
end

function constructHofstadterVqLL(params::Params;p::Int=1,q::Int=16,nLL::Int=10,nγ::Int=2,lk::Int=10)
    A = HofstadterVqLL()
    A.p = p 
    A.q = q 
    A.nLL = nLL 
    A.nγ = nγ
    A.nH = nLL * nγ - 1
    A.lB = sqrt(sqrt(3)*q/(4π * p)) * abs(params.a1)
    gc = 3
    A.svec = collect((-gc*q):(gc*q))
    A.nvec = collect(-gc:gc)
    A.ν = [-2;0;2]

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
    A.kvec = reshape(A.k1,:,1)*params.g1 .+ reshape(A.k2,1,:)*params.g2

    # only consider one external momentum point at (k1,k2)=(0,0)
    A.Λw = zeros(ComplexF64,A.nH*2,A.nH*2,A.l1,A.l2)
    A.ΛΨ = zeros(ComplexF64,2q,2q,A.l1,A.l2)
    calculateρq(A,params)
    computeSingleParticleSpectrum(A,params)

    return A
end

@inline function Coulomb(q::ComplexF64)
    return abs(q)>1e-5 ? 2π*tanh(0.5*abs(q)*abs(params.a1))/abs(q) : 0.0
end

@inline function Knm(n::Int,m::Int,qlB::ComplexF64)
    cplus = 1im /sqrt(2) * (real(qlB) + 1im * imag(qlB))
    cminus = 1im /sqrt(2) * (real(qlB) - 1im * imag(qlB))
    x = - 0.5 * abs2(qlB)
    if n>= m 
        return exp(x/2)*sqrt(Float64(factorial(big(m))/factorial(big(n)))) * laguerrel(m,n-m,-x) * cplus^(n-m)
    else 
        return exp(x/2)*sqrt(Float64(factorial(big(n))/factorial(big(m)))) * laguerrel(n,m-n,-x) * cminus^(m-n)
    end
end

@inline function expfactor(k2l::Float64,qval::ComplexF64,lB::Float64) 
    return exp(1im * real(qval) * (k2l - 0.5*imag(qval)) * lB^2)
end

@inline function indexLL(iH::Int)
    if iH == 1 
        iLL, iγ = 0,0
    else
        iLL, iγ = (iH -1 -1 )÷2 + 1, mod((iH -1 -1 ),2) + 1
    end
    return iLL,iγ
end

function constructΛs(A::HofstadterVqLL,params::Params,s1::Int,m1::Int)
    Λw = reshape(A.Λw,A.nH,2,A.nH,2,A.l1,A.l2)
    ik1,ik2 = 1,1 # external momenta at (0,0)
    for ip2 in 1:A.l2, ip1 in 1:A.l1 
        qval = A.kvec[ik1,ik2] - A.kvec[ip1,ip2] - s1/A.q * params.g2 - m1 * params.g1
        for iH1 in 1:A.nH 
            n1,iγ1 = indexLL(iH1)
            for iH2 in 1:A.nH
                n2,iγ2 = indexLL(iH2)
                Knm1 = Knm(n1,n2,qval*A.lB)
                Knm2 = (n1==0 || n2==0) ? 0 : (2iγ1-3)*(2iγ2-3) *  Knm(n1-1,n2-1,qval*A.lB)
                for ilayer in 1:2
                    Kl = (ilayer == 1) ? params.Kb : params.Kt
                    k2l = - imag(Kl)
                    Λw[iH1,ilayer,iH2,ilayer,ip1,ip2]  = 
                        exp(1im * 2π * (A.k1[ip1]-A.k2[ip2]/2) * s1) * 
                        exp(-1im * π * s1 * (s1-1) /(2A.q) ) *
                        expfactor(k2l,qval,A.lB) * exp(-1im *s1 * real(Kl)*real(params.a1)) *
                        (Knm1+Knm2) * ((n1==0) ? 1 : 1/sqrt(2)) * ((n2==0) ? 1 : 1/sqrt(2) )
                end
            end
        end
    end

    w0 = params.w0/params.w1
    q = A.q
    nLL = A.nLL ÷q
    Λ0 = load("VqLL/NarrowBandEigenstates_w0$(w0)_q$(q)_nLL$(nLL).jld","vec")  # (2nH,2q,l1,l2)
    Uort_k = Λ0[:,:,ik1,ik2]
    for ip2 in 1:A.l2, ip1 in 1:A.l1 
        Uort_p = Λ0[:,:,ip1,ip2]
        A.ΛΨ[:,:,ip1,ip2] = Uort_k' * A.Λw[:,:,ip1,ip2] * Uort_p
    end

    return nothing
end


function constructΛs_diagonal(A::HofstadterVqLL,params::Params,s1::Int,m1::Int)
    Λw = reshape(A.Λw,A.nH,2,A.nH,2,A.l1,A.l2)
    # external and internal momenta are equal
    qval =  - s1/A.q * params.g2 - m1 * params.g1
    for iH1 in 1:A.nH 
        n1,iγ1 = indexLL(iH1)
        for iH2 in 1:A.nH
            n2,iγ2 = indexLL(iH2)
            Knm1 = Knm(n1,n2,qval*A.lB)
            Knm2 = (n1==0 || n2==0) ? 0 : (2iγ1-3)*(2iγ2-3) *  Knm(n1-1,n2-1,qval*A.lB)
            for ip2 in 1:A.l2, ip1 in 1:A.l1 
                for ilayer in 1:2
                    Kl = (ilayer == 1) ? params.Kb : params.Kt
                    k2l = A.k2[ip2]*imag(params.g2) - imag(Kl)
                    Λw[iH1,ilayer,iH2,ilayer,ip1,ip2]  = 
                        exp(1im * 2π * (A.k1[ip1]-A.k2[ip2]/2) * s1) * 
                        exp(-1im * π * s1 * (s1-1) /(2A.q) ) *
                        expfactor(k2l,qval,A.lB) * exp(-1im *s1 * real(Kl)*real(params.a1)) *
                        (Knm1+Knm2) * ((n1==0) ? 1 : 1/sqrt(2)) * ((n2==0) ? 1 : 1/sqrt(2) )
                end
            end
        end
    end

    w0 = params.w0/params.w1
    q = A.q
    nLL = A.nLL ÷q
    Λ0 = load("VqLL/NarrowBandEigenstates_w0$(w0)_q$(q)_nLL$(nLL).jld","vec")  # (2nH,2q,l1,l2)
    is1 = s1 - A.svec[1] + 1
    im1 = m1 - A.nvec[1] + 1
    for ip2 in 1:A.l2, ip1 in 1:A.l1 
        Uort_p = Λ0[:,:,ip1,ip2]
        A.ρq[im1,is1] += tr(Uort_p' * A.Λw[:,:,ip1,ip2] * Uort_p)
    end
    # display("text/plain",real(A.ρq[4,:])')
    return nothing
end

function computeSingleParticleSpectrum(A::HofstadterVqLL,params::Params)
    A.H = zeros(ComplexF64,2A.q,2A.q,length(A.ν))

    for s1 in A.svec
        println(s1)
        for m1 in A.nvec
            Vq = reshape( Coulomb.( A.kvec[1,1] .- A.kvec .- s1/A.q * params.g2 .- m1*params.g1  ), 1,1,1,:)
            constructΛs(A,params,s1,m1)
            Λ1 = reshape(A.ΛΨ,2A.q,1,2A.q,A.l1*A.l2)
            Λ2 = reshape(A.ΛΨ,1,2A.q,2A.q,A.l1*A.l2)
            A.H .+= reshape( sum( Vq .* Λ1 .* conj.(Λ2), dims=(3,4)), 2A.q,2A.q,1)

            is1 = s1 - A.svec[1] + 1
            im1 = m1 - A.nvec[1] + 1
            tmp = Coulomb( -s1/A.q*params.g2 - m1*params.g1 ) * (A.ρq[im1,is1] * A.ΛΨ[:,:,1,1]' )
            for iν in eachindex(A.ν)
                A.H[:,:,iν] .+= A.ν[iν] * tmp
            end
        end
    end

    Lm = (4π)/(sqrt(3)*abs(params.g1))
    V0 = 1/Lm
    A.H .*= (1/(sqrt(3)*Lm^2*A.l1^2) / V0)
    return nothing
end

function calculateρq(A::HofstadterVqLL,params::Params)
    A.ρq = zeros(ComplexF64,length(A.nvec),length(A.svec))

    for m1 in A.nvec, s1 in A.svec 
        constructΛs_diagonal(A,params,s1,m1)
    end
    return nothing
end