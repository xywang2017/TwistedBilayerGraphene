include("Parameters_mod.jl")
using ClassicalOrthogonalPolynomials
using LinearAlgebra

## 
# We note that this is a wrong implementation of the Hilbert space!
# due to the fact that the MTG generators are defined at 1/q rather than actual flux
mutable struct HofstadterLL
    # Here we try to implement the calculation for a given 1/q, and add the energetics from x/lb1^2 
    # the test is to see whether the truncated set of LL wavefunctions can capture the spectrum away from 1/q
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
    Σz::Array{ComplexF64,4}  #  σz operator 
    X::Array{ComplexF64,6}   # this is  x/lB0^2 * σy μ0
    factor::Float64   # this is lB0^2/lB1^2
 
    H::Array{ComplexF64,6}  # diagonal + tunneling part 
    spectrum::Array{Float64,3}
    PΣz::Array{ComplexF64,4}  #  σz operator projected onto 2q states
    σz::Array{Float64,2} # trace of PΣz

    HofstadterLL() = new()
end

function constructHofstadterLL(params::Params;p::Int=1,q::Int=16,nLL::Int=10,nγ::Int=2,lk::Int=20,factor::Float64=0.0)
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
    A.k2 = collect(0:(lk-1)) ./ lk  .* A.p/A.q

    A.Knm = zeros(ComplexF64,nLL,nLL,length(A.qjs))
    constructKnm(A)

    A.Is = zeros(ComplexF64,A.nH,A.nH,length(A.qjs))
    constructIs(A,params)

    # X operator 
    A.factor = factor
    println("factor= ",factor)
    A.X = zeros(ComplexF64,A.nH,2,A.nH,2,lk,lk)
    constructX(A,params)

    # 2 comes from layer degree of freedom
    A.H = zeros(ComplexF64,A.nH,2,A.nH,2,lk,lk)

    for i2 in eachindex(A.k2)
        for i1 in eachindex(A.k1)
            k1 = A.k1[i1]
            k2 = A.k2[i2]
            A.expfactors = ComplexF64[1;exp(-1im*2π*k1*A.q - 1im*π/2 - 1im*π/3*A.q); exp(-1im*2π*(k1-k2)*A.q - 1im*π/2 - 1im*2π/3*A.q) ]
            for jq in eachindex(A.qjs)
                A.H[:,1,:,2,i1,i2] += view(A.Is,:,:,jq) * A.expfactors[jq]
            end
        end
    end

    # diagonal part
    ϵB = sqrt(2) * params.vf / A.lB
    for n in 1:(A.nLL-1)
        for iγ in 1:2
            iH = (n-1)*2 + iγ + 1 
            γ = 2iγ - 3
            A.H[iH,1,iH,1,:,:] .= ϵn(n,γ) * ϵB
            A.H[iH,2,iH,2,:,:] .= ϵn(n,γ) * ϵB
        end
    end

    #
    A.H .-= A.X
    A.H ./= A.ϵ0

    # Σz operator 
    A.Σz = zeros(ComplexF64,A.nH,2,A.nH,2)
    A.Σz[1,1,1,1] = 1
    A.Σz[1,2,1,2] = 1
    for n in 1:(A.nLL-1)
        for iγ1 in 1:2
            γ1 = 2iγ1 - 3
            iH1 = (n-1)*2 + iγ1 + 1 
            for iγ2 in 1:2
                γ2 = 2iγ2 - 3
                iH2 = (n-1)*2 + iγ2 + 1 
                A.Σz[iH1,1,iH2,1] = (1-γ1*γ2)/2
                A.Σz[iH1,2,iH2,2] = (1-γ1*γ2)/2
            end
        end
    end

    # spectrum 
    A.spectrum = zeros(Float64,2A.nH,lk,lk)
    A.PΣz = zeros(ComplexF64,2A.q,2A.q,lk,lk)
    A.σz = zeros(Float64,lk,lk)
    Σz = reshape(A.Σz,2A.nH,2A.nH)
    H = reshape(A.H,2A.nH,2A.nH,lk,lk)
    # X = reshape(A.X,2A.nH,2A.nH,lk,lk)
    for i2 in eachindex(A.k2)
        for i1 in eachindex(A.k1)
            Hk = view(H,:,:,i1,i2)
            # Xk = view(X,:,:,i1,i2)
            # if norm(Xk-Xk') > 1e-6
            #     println("Non Hermitian Hamiltonian from X")
            #     println(norm(Xk-Xk')/norm(Xk))
            # end
            F = eigen(Hermitian(Hk))
            A.spectrum[:,i1,i2] = F.values 
            # pick out 2q states in the middle 
            idx_flat = (A.nH+1-q):(A.nH+q)
            vec = F.vectors[:,idx_flat]
            A.PΣz[:,:,i1,i2] .= vec' * Σz * vec
            # if imag(tr(A.PΣz[:,:,i1,i2]))>1e-6
            #     println("Error with realness of tr(PΣz)")
            # end
            A.σz[i1,i2] = real(tr(A.PΣz[:,:,i1,i2]))
        end
    end
    return A
end


@inline function ϵn(n::Int,γ::Int)
    return γ * sqrt(n) 
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
            Tj = params.T0 
        elseif jq == 2
            Tj = params.T2
        else
            Tj = params.T1 
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

function constructX(A::HofstadterLL,params::Params)
    @inline function iH(n::Int,iγ::Int) ::Int
        return n==0 ? 1 : 2(n-1) + iγ + 1
    end
    @inline function γ(iγ::Int) ::Int 
        return 2iγ - 3
    end
    # construct X in the Landau level basis 
    # A.X = zeros(ComplexF64,A.nH,2,A.nH,2,A.lk,A.lk)
    # n,m \neq 0 cases
    for n in 0:(A.nLL-1)
        m = n
        for iγ1 in 1:2 
            iγ2 = iγ1
            A.X[iH(n,iγ1),1,iH(m,iγ2),1,:,:] .= - 1/(2*sqrt(2)*A.lB)*(γ(iγ1)+γ(iγ2))*sqrt(m) 
            A.X[iH(n,iγ1),2,iH(m,iγ2),2,:,:] .= - 1/(2*sqrt(2)*A.lB)*(γ(iγ1)+γ(iγ2))*sqrt(m)
        end

        m = n - 2
        if (m>=0)
            for iγ1 in 1:2 
                for iγ2 in 1:2 
                    A.X[iH(n,iγ1),1,iH(m,iγ2),1,:,:] .= - 1/(2*sqrt(2)*A.lB)*γ(iγ1)*sqrt(m+1) * ( m==0 ? sqrt(2) : 1 ) 
                    A.X[iH(n,iγ1),2,iH(m,iγ2),2,:,:] .= - 1/(2*sqrt(2)*A.lB)*γ(iγ1)*sqrt(m+1) * ( m==0 ? sqrt(2) : 1 ) 
                end
            end
        end

        m = n + 2
        if (m<=A.nLL-1)
            for iγ1 in 1:2 
                for iγ2 in 1:2 
                    A.X[iH(n,iγ1),1,iH(m,iγ2),1,:,:] .= - 1/(2*sqrt(2)*A.lB)*γ(iγ2)*sqrt(m-1) * ( n==0 ? sqrt(2) : 1 ) 
                    A.X[iH(n,iγ1),2,iH(m,iγ2),2,:,:] .= - 1/(2*sqrt(2)*A.lB)*γ(iγ2)*sqrt(m-1) * ( n==0 ? sqrt(2) : 1 ) 
                end
            end
        end

        m = n - 1
        if (m>=0)
            for iγ1 in 1:2 
                for iγ2 in 1:2 
                    for ik2 in eachindex(A.k2)
                        k2 = 2π/abs(params.a1) * A.k2[ik2]
                        A.X[iH(n,iγ1),1,iH(m,iγ2),1,:,ik2] .= - 0.5*(k2-imag(params.Kb)) * γ(iγ1) * ( m==0 ? sqrt(2) : 1 ) 
                        A.X[iH(n,iγ1),2,iH(m,iγ2),2,:,ik2] .= - 0.5*(k2-imag(params.Kt)) * γ(iγ1) * ( m==0 ? sqrt(2) : 1 ) 
                    end
                end
            end
        end

        m = n + 1
        if (m<=A.nLL-1)
            for iγ1 in 1:2 
                for iγ2 in 1:2 
                    for ik2 in eachindex(A.k2)
                        k2 = 2π/abs(params.a1) * A.k2[ik2]
                        A.X[iH(n,iγ1),1,iH(m,iγ2),1,:,ik2] .= - 0.5*(k2-imag(params.Kb)) * γ(iγ2) * ( n==0 ? sqrt(2) : 1 ) 
                        A.X[iH(n,iγ1),2,iH(m,iγ2),2,:,ik2] .= - 0.5*(k2-imag(params.Kt)) * γ(iγ2) * ( n==0 ? sqrt(2) : 1 ) 
                    end
                end
            end
        end
    end

    A.X .*= (params.vf * A.factor)
    return nothing
end