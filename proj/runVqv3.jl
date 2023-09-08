using PyPlot
using JLD
fpath = pwd()
include(joinpath(fpath,"libs/HofstadterVq_mod.jl"))

##
ϕ = [3//10;1//4;1//5;3//16;1//6;1//7;1//8;1//9;1//10;1//16];
ϕ = [3//10;1//4;1//5;1//6;1//7;1//8;1//16];
# ϕ = [1//8;1//9;1//10;1//16];
params = Params(dθ=1.05π/180,ϵ=0.0,w0=0.96056,w1=96.056);
for iϕ in eachindex(ϕ)
    p = numerator(ϕ[iϕ])
    q = denominator(ϕ[iϕ])
    hof = HofstadterVq()
    initHofstadterVq(hof,params,p=p,q=q,lk=10);
    save("StrongCoupling/strongcoupling_q$(q)_p$(p)_chiralv2.jld","spectrum",hof.H,"sigmaz",hof.Σz)
end
##

fig = figure() 
for iϕ in eachindex(ϕ)
    p = numerator(ϕ[iϕ])
    q = denominator(ϕ[iϕ])
    # println(p//q)
    data = load("StrongCoupling/strongcoupling_q$(q)_p$(p)_chiralv2.jld")
    H = data["spectrum"]
    Pz = data["sigmaz"]
    energies = zeros(Float64,size(H,2),size(H,3))
    for i1 in 1:size(H,3)
        F = eigen(Hermitian(Pz[:,:,i1]))
        Hnew = F.vectors' * H[:,:,i1] * F.vectors
        energies[1:(q+p),i1] = eigvals(Hermitian(Hnew[1:(q+p),1:(q+p)]))
        energies[(q+p+1):(2q),i1] = eigvals(Hermitian(Hnew[(q+p+1):(2q),(q+p+1):(2q)]))
    end
    ϵ1 = energies[1:(q+p),:]
    ϵ2 = energies[(q+p+1):(2q),:]

    ϵmin = maximum(energies) * 0
    # if ϕ[iϕ] in [1//4,3//10,3//16]
    #     ϵmin = 0.122
    # end
    plot(ones(length(ϵ1))/q*p,ϵ1[:].-ϵmin ,"bo",ms=1,markerfacecolor="none")
    plot(ones(length(ϵ2))/q*p,ϵ2[:].-ϵmin,"ro",ms=1,markerfacecolor="none")
end

xlabel("p/q")
ylabel("Spectrum")
tight_layout()
savefig("StrongCoupling/hofstadter_chiralv2.pdf")
display(fig)
close(fig)


######


fig = figure() 
for iϕ in eachindex(ϕ)
    p = numerator(ϕ[iϕ])
    q = denominator(ϕ[iϕ])
    println(p//q)
    data = load("StrongCoupling/strongcoupling_q$(q)_p$(p)_chiralv2.jld")
    H = data["spectrum"]
    Pz = data["sigmaz"]
    energies = zeros(Float64,size(H,2),size(H,3))
    for i1 in 1:size(H,3)
        F = eigen(Hermitian(Pz[:,:,i1]))
        Hnew = F.vectors' * H[:,:,i1] * F.vectors
        energies[:,i1] = eigvals(Hermitian(Hnew))
        # energies[1:(q+1),i1] = eigvals(Hermitian(Hnew[1:(q+1),1:(q+1)]))
        # energies[(q+2):(2q),i1] = eigvals(Hermitian(Hnew[(q+2):(2q),(q+2):(2q)]))
    end
    ϵ1 = energies
    plot(ones(length(ϵ1))/q*p,ϵ1[:],"bo",ms=1,markerfacecolor="none")
end

xlabel("p/q")
ylabel("Spectrum")
tight_layout()
# savefig("StrongCoupling/hofstadter_chiralv2.pdf")
display(fig)
close(fig)