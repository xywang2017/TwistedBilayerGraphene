include("Parameters_mod.jl")
using ClassicalOrthogonalPolynomials
using LinearAlgebra

mutable struct HofstadterLL
    # this implements 1/q sequence
    p::Int 
    q::Int 
    nq::Int 
    nLL::Int   # 0,1,2... 
    nγ::Int    # 2, particle-hole symmetry 
    nH::Int    # nLL*nγ - 1
    lk::Int
    k1::Vector{Float64} 
    k2::Vector{Float64}
    l1::Int
    l2::Int

    lB::Float64  # magnetic length in absolute units 
    kθ::Float64  
    ϵ0::Float64  # vfkθ

    qjs::Vector{Int}
    Knm::Array{ComplexF64,3}   # matrix with Laguerre Polynomials
    Is::Array{ComplexF64,3}  # third axis correspond to the 3 qjs
    expfactors::Vector{ComplexF64}  # exp(1im*...) factors for 3 qjs
    Σz::Array{ComplexF64,4}  #  σz operator 
 
    H::Array{ComplexF64,6}  # diagonal + tunneling part 
    spectrum::Array{Float64,3}
    PΣz::Array{ComplexF64,4}  #  σz operator projected onto 2q states
    σz::Array{Float64,2} # trace of PΣz

    HofstadterLL() = new()
end

function constructHofstadterLL(params::Params;p::Int=1,q::Int=16,nLL::Int=10,nγ::Int=2,lk::Int=10)
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
    

    A.Knm = zeros(ComplexF64,nLL,nLL,length(A.qjs))
    constructKnm(A)

    A.Is = zeros(ComplexF64,A.nH,A.nH,length(A.qjs))
    constructIs(A,params)

    # 2 comes from layer degree of freedom
    A.H = zeros(ComplexF64,A.nH,2,A.nH,2,A.l1,A.l2)

    for i2 in eachindex(A.k2)
        for i1 in eachindex(A.k1)
            k1 = A.k1[i1]
            k2 = A.k2[i2]
            qjxj = [0; -π*A.q/A.p *(k2-1/6); π*A.q/A.p *(k2-1/6)]
            s1 = [0; -A.q; - A.q]
            A.expfactors = @. exp(1im*2π*(k1-k2/2)*s1 - 1im*π*s1*(s1-1)/2*A.p/A.q + 1im * qjxj)
            # A.expfactors = ComplexF64[1;exp(-1im*2π*k1*A.q - 1im*π/2 - 1im*π/3*A.q); exp(-1im*2π*(k1-k2)*A.q - 1im*π/2 - 1im*2π/3*A.q) ]
            for jq in eachindex(A.qjs)
                A.H[:,2,:,1,i1,i2] += view(A.Is,:,:,jq) *  A.expfactors[jq]
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
    A.spectrum = zeros(Float64,2A.nH,A.l1,A.l2)
    A.PΣz = zeros(ComplexF64,2A.q,2A.q,A.l1,A.l2)
    A.σz = zeros(Float64,A.l1,A.l2)
    Σz = reshape(A.Σz,2A.nH,2A.nH)
    H = reshape(A.H,2A.nH,2A.nH,A.l1,A.l2)
    for i2 in eachindex(A.k2)
        for i1 in eachindex(A.k1)
            F = eigen( Hermitian( H[:,:,i1,i2], :L ) ,sortby=abs)
            A.spectrum[:,i1,i2] = F.values 
            # pick out 2q states in the middle 
            idx_flat = (A.nH+1-q):(A.nH+q)
            vec = F.vectors[:,idx_flat]
            A.PΣz[:,:,i1,i2] .= vec' * Σz * vec
            # if imag(tr(A.PΣz[:,:,i1,i2]))>1e-6
            #     println("Error with realness of tr(PΣz)")
            # end
            A.σz[i1,i2] = real(tr(A.PΣz[:,:,i1,i2]))
            
            # writeout 
            if (i1==1 && i2==1) 
                dθ = params.dθ * 180 / π
                w0 = params.w0 
                w1 = params.w1
                fname = "StrongCoupling/dtheta$(dθ)_w0$(w0)_w1$(w1)/LL_eigenstates_q$(q)_w0$(w0).jld"
                # save(fname,"LLvec",vec)
                save(fname,"LLvec",F.vectors)
            end
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
    γ = [-1;1]
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