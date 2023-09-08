using LinearAlgebra

mutable struct Lattice 
    lk::Int # only lk=even is implemented
    listk::Matrix{Int} # indices of the k-grid in the first Moire Brillouin zone, i1,i2, (i2-1)lk+i1
    indΓ::Vector{Int}  # index of the Γ point in the mBZ, iΓ1,iΓ2,(iΓ2-1)lk+iΓ1
    kvec::Vector{ComplexF64} # values of kvectors in the first Moire Brilloun zone
    indkinv::Vector{Int} # indices of the k-grid corresponding to the inverse in the first MBZ, i.e. listk[:,indkinv]
    flagG::Matrix{Int}  # this takes care of the inverse where -k => k + n G; n is a 2-element vector 

    Lattice() = new()
end

function initLattice(Latt::Lattice,params::Params;lk::Int=12)
    @assert lk%2 ==0
    Latt.lk = lk

    itr = 1:Latt.lk
    Latt.listk = zeros(Int,3,lk^2)
    for i2 in itr, i1 in itr
        Latt.listk[3,(i2-1)*lk+i1] = (i2-1)*lk+i1
        Latt.listk[1,(i2-1)*lk+i1] = i1 
        Latt.listk[2,(i2-1)*lk+i1] = i2 
    end

    # Latt.indΓ=Latt.listk[:,lk^2÷2+lk÷2+1]
    Latt.indΓ = Latt.listk[:,1]

    Latt.kvec = zeros(ComplexF64,lk^2)
    for ik in 1:lk^2
        Latt.kvec[ik] = (Latt.listk[1,ik] * params.g1 + Latt.listk[2,ik] * params.g2)/lk
    end
    Latt.kvec .= Latt.kvec .- Latt.kvec[Latt.indΓ[3]] 

    # inverse 
    Latt.indkinv = zeros(Int,lk^2)
    Latt.flagG = zeros(Int,2,lk^2)
    for ik in 1:lk^2 
        i1inv = Latt.indΓ[1]*2 - Latt.listk[1,ik]
        i2inv = Latt.indΓ[1]*2 - Latt.listk[2,ik]
        if (i1inv-1)÷lk !=0
            Latt.flagG[1,ik] = (i1inv-1)÷lk
        end
        if (i2inv-1)÷lk !=0
            Latt.flagG[2,ik] = (i2inv-1)÷lk
        end
        Latt.indkinv[ik] = ((i2inv-1)%lk)*lk + (i1inv-1)%lk+1
    end

end