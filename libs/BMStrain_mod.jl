using Arpack
include("Parameters_mod.jl")
include("Lattice_mod.jl")
# --------------------------------------------------------------------------------------------------------------- #
mutable struct HBM
    

    nlocal::Int  # 4 is the local Hilbert space (layerxsublattice)
    lg::Int 
    listG::Matrix{Int}  # indices of the reciprocal lattice vectors g specifying the Hamiltonian 
    gvec::Vector{ComplexF64} # values of gvectors corresponding to the list
    nflat::Int # number of flat bands per valley is 2

    T12::Matrix{ComplexF64} # k-independent part of the Hamiltonian
    Hk::Matrix{Float64} # flat band energies 2xlk^2
    Uk::Matrix{ComplexF64} # gauge-fixed eigenvectors 4lg^2 x nflat lk^2
    
    C2T::Matrix{Float64} # Unitary part of the C2T symmetry 
    Ph::Matrix{Float64} # Particle-hole symmetry 

    HkKbar::Matrix{Float64} # flat band energies in the opposite valley, constructed by symmetry

    HBM() = new()
end

@inline function dirac(k::ComplexF64) ::Matrix{ComplexF64}
    return  abs(k)*[0 exp(-1im*angle(k));exp(1im*angle(k)) 0]
end

@inline function V(q::ComplexF64) ::Float64
    res = 1e-6
    if abs(q) < res
        Vq = 0
    else
        Vq = 2π/abs(q)
    end
    return Vq
end

function generate_T12(blk::HBM,params::Params)
    # p.b.c. is used 
    idg = reshape(collect(1:blk.lg^2),blk.lg,blk.lg)

    # idg_nn1 = circshift(idg,(-1,0))  # T1 * (|t><b|)
    # idg_nn2 = circshift(idg,(0,-1))  # T2 * (|t><b|)
    # idg_nn12 = circshift(idg,(-1,-1))  # T0 * (|t><b|)

    # per Oskar & Jian choice of g1 and g2
    # idg_nn1 = circshift(idg,(-1,-1))  # T1 * (|t><b|)
    # idg_nn2 = circshift(idg,(0,-1))  # T2 * (|t><b|)
    # idg_nn12 = circshift(idg,(-1,-2))  # T0 * (|t><b|)

    # per Oskar & Jian choice of g1 and g2
    idg_nn1 = circshift(idg,(0,1))  # T1 * (|t><b|)
    idg_nn2 = circshift(idg,(1,1))  # T2 * (|t><b|)
    idg_nn12 = circshift(idg,(0,0))  # T0 * (|t><b|)

    tmp = zeros(ComplexF64,4,blk.lg^2,4,blk.lg^2)

    for ig in eachindex(idg)
        tmp[3:4,idg[ig],1:2,idg_nn1[ig]] = params.T2
        tmp[1:2,idg_nn1[ig],3:4,idg[ig]] = params.T2

        tmp[3:4,idg[ig],1:2,idg_nn2[ig]] = params.T1
        tmp[1:2,idg_nn2[ig],3:4,idg[ig]] = params.T1

        tmp[3:4,idg[ig],1:2,idg_nn12[ig]] = params.T0
        tmp[1:2,idg_nn12[ig],3:4,idg[ig]] = params.T0
    end

    blk.T12 .= reshape(tmp,blk.nlocal*blk.lg^2,blk.nlocal*blk.lg^2)

    return nothing
end

function ComputeH(H::Matrix{ComplexF64},blk::HBM,params::Params,k::ComplexF64;ig::Vector{Int}=[0,0])
    """
        Dirac Hamiltonian in the Bloch band basis
    """
    # Note here k only takes values within first mBZ 
    # if k is outside of first BZ, it is labeled by k + ig[1]*blk.g1 + ig[2]*blk.g2
    H .= 0.0 + 0.0im

    itr = reshape(collect(1:blk.lg^2),blk.lg,blk.lg)
    idg = view(circshift(itr,(ig[1],ig[2])),:)
    R = Float64[cos(params.dθ/2)-1 sin(params.dθ/2);-sin(params.dθ/2) cos(params.dθ/2)-1]

    # dispersive part
    for ig in 1:blk.lg^2
        qc = blk.gvec[ig]
        kb = k - params.Kb + qc
        kt = k - params.Kt + qc
        # k1 = (I + R' + params.S/2)*[real(kb);imag(kb)]
        # k2 = (I + R -params.S/2)*[real(kt);imag(kt)]
        k1 = [real(kb);imag(kb)]
        k2 = [real(kt);imag(kt)]
        H[(4idg[ig]-3):(4idg[ig]-2),(4idg[ig]-3):(4idg[ig]-2)] = params.vf*dirac(k1[1]+1im*k1[2])
        H[(4idg[ig]-1):(4idg[ig]),(4idg[ig]-1):(4idg[ig])] = params.vf*dirac(k2[1]+1im*k2[2])
    end
    
    H .= H + blk.T12 - params.μ*I

    return nothing
end

function initHBM(blk::HBM,Latt::Lattice,params::Params;lg::Int=9)
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

    # this gives i mu_y I operation in the Bloch basis
    Ig = reverse(Array{Float64}(I,blk.lg^2,blk.lg^2),dims=1)
    blk.Ph = -kron(Ig,kron(is2,s0))

    # this gives C2T eigenstates
    Ig = Array{Float64}(I,blk.lg^2,blk.lg^2)
    blk.C2T = kron(Ig,kron(s0,s1)) # × conj(...)

    blk.nlocal = 4
    blk.nflat = 2
    
    blk.Uk = zeros(ComplexF64,blk.nlocal*blk.lg^2,blk.nflat*Latt.lk^2)
    blk.Hk =zeros(Float64,blk.nflat,Latt.lk^2)

    blk.T12 = zeros(ComplexF64,blk.nlocal*blk.lg^2,blk.nlocal*blk.lg^2)
    generate_T12(blk,params)

    # temporary container H for each k
    H = zeros(ComplexF64,blk.nlocal*blk.lg^2,blk.nlocal*blk.lg^2)

    for ik in eachindex(Latt.kvec)
        kval = Latt.kvec[ik] #+ 0.5*(1+1im)/Latt.lk
        ComputeH(H,blk,params,kval)
        # Find the smallest eigenvalue and eigenvectors close to zero
        vals, vecs = eigs(Hermitian(H),nev=blk.nflat,which=:SM)
        # F = eigen(Hermitian(H),sortby=abs)
        # vals = F.values[1:2]
        # vecs = F.vectors[:,1:2]

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
    
    blk.HkKbar = zeros(Float64,blk.nflat,Latt.lk^2)
    blk.HkKbar .= - circshift(blk.Hk,(1,0))
    return nothing
end