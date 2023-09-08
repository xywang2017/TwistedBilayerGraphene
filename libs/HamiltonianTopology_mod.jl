mutable struct Hamiltonian 
    nsites::Int 
    norbs::Int 
    θ::Vector{Float64}
    t::Float64 
    Δ::Float64   # onsite term
    s0::Matrix{ComplexF64}
    s1::Matrix{ComplexF64}
    s2::Matrix{ComplexF64}
    s3::Matrix{ComplexF64}

    H::Matrix{ComplexF64}
    Hamiltonian() = new()
end

function initHamiltonian(;nsites::Int=100,norbs::Int=2,t::Float64=1.0,Δ::Float64=1.0)
    A = Hamiltonian()
    A.nsites = nsites 
    A.norbs = norbs 
    A.t = t
    A.Δ = Δ
    A.s0 = ComplexF64[1 0;0 1]
    A.s1 = ComplexF64[0 1;1 0]
    A.s2 = ComplexF64[0 -1im;1im 0]
    A.s3 = ComplexF64[1 0;0 -1]
    A.H = zeros(ComplexF64,norbs *nsites, norbs*nsites) 

    A.θ = collect(0:(nsites-1))./nsites * π   # π phase winding instead of 2π

    # onsite term
    for i in 1:nsites
        A.H[(2i-1):(2i),(2i-1):(2i)] = Δ*A.s1 * tanh((A.θ[i]-π/2)/0.1)

    end

    for i in 1:nsites 
        inn = i%nsites + 1
        A.H[(2i-1):(2i),(2inn-1):(2inn)] = t * (A.s1- 1im * exp(1im * A.θ[i])*A.s2)
    end
    A.H = A.H + A.H' 
    return A
end