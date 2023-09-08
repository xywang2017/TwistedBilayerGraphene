include("Parameters_mod.jl")
using ClassicalOrthogonalPolynomials
using LinearAlgebra
using Arpack

mutable struct HofstadterLL
    # this implements a p \neq 1 sequence
    p::Int 
    q::Int 
    nLL::Int   # 0,1,2... 
    nγ::Int    # 2, particle-hole symmetry 
    nH::Int    # nLL*nγ - 1
    lk::Int
    k1::Vector{Float64} 
    k2::Vector{Float64}

    lB::Float64  # magnetic length in absolute units 
    kθ::Float64  
    ϵ0::Float64  # vfkθ

    qjs::Vector{Int}
    Knm::Array{ComplexF64,3}   # matrix with Laguerre Polynomials
    Is::Array{ComplexF64,3}  # third axis correspond to the 3 qjs
    expfactors::Vector{ComplexF64}  # exp(1im*...) factors for 3 qjs
    
    info::Array{Int,3} # info[1,i,iqj] and info[2,i,iqj] are k2c and s1 values
    O1::Array{ComplexF64,7} # q1 q2 q3 contributions
    O0::Array{ComplexF64,4} # diagonal term, k-independent

    # Hamiltonian can be constructed entirely based on info, O; 
    # here we only save eigenvalues of a few low-lying energy states using eigs
    nlow::Int  # number of low energy states to keep
    spectrum::Array{Float64,2}

    HofstadterLL() = new()
end

function constructHofstadterLL(params::Params;p::Int=1,q::Int=16,nLL::Int=10,nγ::Int=2,lk::Int=20)
    A = HofstadterLL()
    A.p = p 
    A.q = q 
    A.nLL = nLL 
    A.nγ = nγ
    A.nH = nLL * nγ - 1
    A.qjs = collect(0:2)
    A.lB = sqrt(sqrt(3)*q/(4π * p)) * abs(params.a1)
    A.kθ = params.kb
    A.ϵ0 = params.vf*A.kθ

    A.lk = lk
    A.k1 = collect(0:(lk-1)) ./ lk 
    A.k2 = collect(0:(lk*A.p-1)) ./ (lk*A.q)

    ## here we first work out the indices
    A.info = zeros(Int,2,length(A.k2),length(A.qjs))  
    for jq in eachindex(A.qjs)
        qj2 = 0 
        if (jq!=1)
            qj2 = 1
        end
        tmp = eachindex(A.k2) .- qj2 * A.lk * A.q 
        A.info[1,:,jq] = mod.(tmp .- 1,length(A.k2)) .+ 1
        A.info[2,:,jq] = (tmp .- A.info[1,:,jq]) .÷ length(A.k2)
    end
        


    A.Knm = zeros(ComplexF64,nLL,nLL,length(A.qjs))
    constructKnm(A)

    A.Is = zeros(ComplexF64,A.nH,A.nH,length(A.qjs))
    constructIs(A,params)

    # 2 comes from layer degree of freedom, q1, q2, and q3
    A.O1 = zeros(ComplexF64,A.nH,2,A.nH,2,length(A.k2),length(A.k1),length(A.qjs))

    for i1 in eachindex(A.k1)
        k1 = A.k1[i1]
        for i2 in eachindex(A.k2)
            k2 = A.k2[i2]
            # qjxj = [0; -π*A.q/A.p *(k2-1/6); π*A.q/A.p *(k2-1/6)]
            for jq in eachindex(A.qjs)
                s1 = A.info[2,i2,jq]
                # expfactor = exp(1im*2π*(k1-k2/2)*s1 - 1im*π*s1*(s1-1)/2*A.p/A.q + 1im * qjxj[jq])
                q1 = real(A.qjs[jq])
                q2 = imag(A.qjs[jq])
                expiθ = phasefactor(k1,k2,1.0q1,1.0q2,s1,params,A)
                A.O1[:,1,:,2,i2,i1,jq] = view(A.Is,:,:,jq) * expiθ #expfactor
            end
        end
    end

    # diagonal part
    A.O0 = zeros(ComplexF64,A.nH,2,A.nH,2)
    ϵB = sqrt(2) * params.vf / A.lB
    for n in 1:(A.nLL-1)
        for iγ in 1:2
            iH = (n-1)*2 + iγ + 1 
            γ = 2iγ - 3
            A.O0[iH,1,iH,1] = ϵn(n,γ) * ϵB / A.ϵ0
            A.O0[iH,2,iH,2] = ϵn(n,γ) * ϵB / A.ϵ0
        end
    end

    # spectrum of Hamiltonian
    A.nlow = 2A.nH * length(A.k2) # this should cover only the flat band physics
    A.spectrum = zeros(Float64,A.nlow,length(A.k1))
    computeSpectrum(A)

    return A
end

function computeSpectrum(A::HofstadterLL)
    H = zeros(ComplexF64,2A.nH,length(A.k2),2A.nH,length(A.k2))
    for i1 in eachindex(A.k1)
        println(i1)
        H .= 0.0im
        for ir in eachindex(A.k2)
            H[:,ir,:,ir] .+= 0.5* reshape(A.O0,2A.nH,2A.nH)
            for jq in eachindex(A.qjs)
                ic = A.info[1,ir,jq]
                H[:,ir,:,ic] .+= reshape(view(A.O1,:,:,:,:,ir,i1,jq),2A.nH,2A.nH)
            end
        end
        Hnum = reshape(H,2A.nH*length(A.k2),:)+reshape(H,2A.nH*length(A.k2),:)'
        if (mod(A.q,A.p)==0)
            # splits into diagonals in k2 
            for ir in eachindex(A.k2)
                id = (2A.nH*(ir-1) + 1) : (2A.nH*ir)
                A.spectrum[id,i1] = eigvals(Hermitian(view(Hnum,id,id)))
            end
        else
            # cannot split into diagonals in k2
            Hnum = reshape(H,2A.nH*length(A.k2),:)+reshape(H,2A.nH*length(A.k2),:)'
            A.spectrum[:,i1] = eigvals( Hermitian(Hnum) )
        end
    end
end


@inline function ϵn(n::Int,γ::Int)
    return γ * sqrt(n) 
end


@inline function qjx(q1::Float64,q2::Float64,params::Params)
    e1xe2 = abs(sin(params.θ12))
    e1e2 = cos(params.θ12)
    return 2π/(abs(params.a1)*e1xe2) * (q1 - q2 * abs(params.a1)/abs(params.a2)*e1e2)
end

@inline function xj(q1::Float64,q2::Float64,params::Params)
    Kty = abs(params.Kt) * cos(angle(params.Kt)-angle(params.a2))
    Kby = abs(params.Kb) * cos(angle(params.Kb)-angle(params.a2))
    return (Kty - Kby - 2π/abs(params.a2) * q2)  # this is xj in units of lB^2
end

@inline function phasefactor(k1::Float64,k2::Float64,q1::Float64,q2::Float64,s1::Int,params::Params,A::HofstadterLL)
    θ = 2π * (k1)* s1
    θ = θ - 2π * k2 * s1 * abs(params.a1)/abs(params.a2) * abs( cos(params.θ12) )
    θ = θ - s1*(s1-1)/2 * 2π * A.p/A.q * abs(params.a1)/abs(params.a2) * abs( cos(params.θ12) )
    θ = θ - s1 * real(params.a1) * real(params.Kb)  # newly added line! 1/6/2022 4:07pm
    θ = θ + A.lB^2 * qjx(q1,q2,params) * (2π/abs(params.a2)*k2 - imag(params.Kt) )
    θ = θ + 0.5 * A.lB^2 * xj(q1,q2,params) * qjx(q1,q2,params)
    
    return exp(1im * θ)
end


function constructKnm(A::HofstadterLL)
    x = A.kθ^2 * A.lB^2 /2
    cplus = sqrt(x) * [1; exp(-1im * 2π/3) ; exp(1im * 2π/3)] 
    cminus = sqrt(x) * [-1; exp(-1im * π/3) ; exp(1im * π/3)] 
    for jq in eachindex(A.qjs)
        for n in 0:(size(A.Knm,1)-1)
            for m in 0:(size(A.Knm,2)-1)
                if (n>=m)
                    A.Knm[n+1,m+1,jq] = exp(-x/2)*sqrt(factorial(big(m))/factorial(big(n))) * laguerrel(m,n-m,x) * cplus[jq]^(n-m)
                else
                    A.Knm[n+1,m+1,jq] = exp(-x/2)*sqrt(factorial(big(n))/factorial(big(m))) * laguerrel(n,m-n,x) * cminus[jq]^(m-n)
                end
            end
        end
    end
    return nothing
end

function constructIs(A::HofstadterLL,params::Params)
    # first construct tmp assuming all states are paired, then delete the first row and column when computing Is
    tmp = zeros(ComplexF64,A.nLL*A.nγ,A.nLL*A.nγ)
    
    for jq in eachindex(A.qjs)
        if jq == 1
            Tj = params.T0 ./A.ϵ0
        elseif jq == 2
            Tj = params.T2 ./ A.ϵ0
        else
            Tj = params.T1 ./ A.ϵ0
        end
        for ir in 1:size(tmp,1)
            n = (ir-1)÷A.nγ
            γ1 = 2*mod(ir-1,A.nγ)-1
            for ic in 1:size(tmp,2)
                m = (ic-1)÷A.nγ
                γ2 = 2*mod(ic-1,A.nγ)-1
                tmp[ir,ic] = ( Tj[1,1]*(A.Knm[n+1,m+1,jq]+ ( (n==0 || m==0) ? 0 : γ1*γ2*A.Knm[n,m,jq] ) ) + 
                            (-1im) * γ2 * Tj[1,2] * ( (m==0) ? 0 : A.Knm[n+1,m,jq] ) + 
                            (1im) * γ1 * Tj[2,1] * ( (n==0) ? 0 : A.Knm[n,m+1,jq] ) ) * 0.5
                
                if (n==0)
                    tmp[ir,ic] *= sqrt(2)
                end
                if (m==0)
                    tmp[ir,ic] *= sqrt(2)
                end
            end 
        end
        A.Is[:,:,jq] = view(tmp,2:A.nLL*A.nγ,2:A.nLL*A.nγ)
    end

    return nothing
end