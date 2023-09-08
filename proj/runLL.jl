using PyPlot
using Printf
using DelimitedFiles
using JLD
fpath = pwd()
include(joinpath(fpath,"libs/HofstadterLL_modv3.jl"))
params = Params()
##
# w1=96.056
# w0=0.0
# params = Params(dθ=1.05π/180,w0=w0,w1=w1,ϵ=0.0)  #chiral limit
# initParamsWithStrain(params)
function calculate_LL_Weizmann()
    params = Params()
    qs = collect(2:18)
    p = 1
    data = Dict()
    σz = Dict()
    for iq in eachindex(qs)
        println("q=$(qs[iq])")
        q = qs[iq]
        lk = (q>=10) ? 10 : 20
        hof = constructHofstadterLL(params,q=q,p=p,nLL=25q÷abs(p),lk=lk)
        data["$iq"] =  hof.spectrum
        σz["$iq"] = hof.σz
    end
    save("Weizmann/erez_result.jld","data",data,"σz",σz)
end
calculate_LL_Weizmann()
##
function plot_LL()
    fig = figure(figsize=(3,3))
    qs = collect(2:18)
    d=load("Weizmann/erez_result.jld")
    data = d["data"]
    σz = d["σz"]
    for iq in eachindex(qs)
        p=1
        q = qs[iq]
        lk = (q>=10) ? 4 : 10
        if (q>=lk)
            nq = 1
        else
            nq = (lk-1) ÷ q + 1
        end
        l1, l2 = 2nq, 2nq
        # E = α 0.668 μB B, α is defined as E (meV) = α ϕ/ϕ0 
        # if slope in E (meV) vs ϕ/ϕ0 is 1.5 then it is 1μB
    
        ϵ = data["$iq"]
        σ = σz["$iq"]
        ϕ = ones(Float64,2q) * p/q
        # scatter(ones(length(ϵ))*p/q,ϵ[:],c=σ[:],s=3,cmap="bwr",vmin=-1,vmax=1)

        # Eν = zeros(Float64,2q)
        # for iq in eachindex(Eν)
        #     Eν[iq] = sum(ϵ[1:iq,:,:]) /(l1*l2) #*q)
        # end
        # plot(ϕ,Eν,".",c="k",ms=2,markeredgecolor="none")

        νs = [0.2,0.5,0.8,1,1.4,2]
        colors = ["r","b","g","m","c","y"]
        for iν in eachindex(νs)
            ν = νs[iν]
            idxmax = Int(round(ν/2 * (l1*l2*2q)))
            Eν1 = sum(sort(ϵ[:])[1:idxmax])/(l1*l2*q)
            # Eν1 = sum(ϵ[1:(q-1),:,:])/(l1*l2*q*q)
            # Eν2 = sum(ϵ[q:(2q),:,:])/(l1*l2*q*q)
            plot(p/q,Eν1,".",c=colors[iν],ms=4,markeredgecolor="none")
            # plot(p/q,Eν2,".",c="r",ms=4,markeredgecolor="none")
        end
       
    end
    # colorbar() 
    plot(1 ./ qs, -1.5 ./qs )   # conclusion -> always less than 1 μB per Moire unit cell
    xlabel(L"ϕ/ϕ_0")
    # xticks([0,0.1,0.2,0.3,0.4,0.5])
    # ylim([-6,6])
    xlim([0,0.52])
    # ylabel("E")
    tight_layout()
    display(fig)
    # savefig("Weizmann/LL_spectrum_erez.png",dpi=500,transparent=false)
    close(fig)
end

plot_LL()


##
##
function plot_LL_μ()
    fig, ax = subplots(1,2,figsize=(5,3))
    qs = collect(2:18)
    d=load("Weizmann/erez_result.jld")
    data = d["data"]
    σz = d["σz"]
    for iq in eachindex(qs)
        p=1
        q = qs[iq]
        lk = (q>=10) ? 10 : 20
        if (q>=lk)
            nq = 1
        else
            nq = (lk-1) ÷ q + 1
        end
        nq = 6
        l1, l2 = 2nq, 2nq
        # E = α 0.668 μB B, α is defined as E (meV) = α ϕ/ϕ0 
        # if slope in E (meV) vs ϕ/ϕ0 is 1.5 then it is 1μB
    
        ϵ = data["$iq"]
        σ = σz["$iq"]
        ϕ = ones(Float64,2q) * p/q
        
        μs = collect(-5.9:0.5:4)
        νs = zeros(Float64,length(μs))
        Eνs = zeros(Float64,length(μs))
        colors = [((maximum(μs) .- μs) ./ (maximum(μs)-minimum(μs)))';
                 0 .*cos.((μs.-minimum(μs))./maximum(abs.(μs))*π/4)' ;
                 ((μs.-minimum(μs)) ./ (maximum(μs)-minimum(μs)))']

        colorsB = [( (iq-1)/(length(qs)-1));
                 0.3  ;
                 ( (length(qs)-iq)/(length(qs)-1) )]
        for iμ in eachindex(μs)
            μ = μs[iμ]
            νs[iμ] = sum((sign.(μ .- ϵ).+1)./2)  /(l1*l2*q) # per Moiré unit cell 
            Eνs[iμ] = sum( ϵ.*((sign.(μ .- ϵ).+1)./2) ) /(l1*l2*q) - μ * νs[iμ]
            ax[1].plot(p/q,Eνs[iμ],".",c=colors[:,iμ],ms=3)
        end
        ax[2].plot(μs,νs,".",c=colorsB)
    end
    # ax[1].plot(1 ./ qs, -4 .- 4 ./qs, "--")
    ax[1].set_xlabel(L"ϕ/ϕ_0")
    ax[1].set_ylabel(L"Φ")
    ax[1].set_xlim([0,0.52])
    ax[2].set_xlabel(L"μ")
    ax[2].set_ylabel(L"ν")

    tight_layout()
    display(fig)
    close(fig)
end

plot_LL_μ()

##
# here I show the sublattice polarization as a function of ϕ/ϕ0
fig = figure(figsize=(3,3))

d=load("BMresults/theta1.05_LL_w00.0.jld")
data = d["data"]
σz = d["σz"]
ϕ = p./qs
nσz = [sum(σz[i])/length(σz[i]) for i in eachindex(σz)]
plot(ϕ,nσz,"ro",markerfacecolor="none",ms=3,label=L"w_0/w_1=0.0")

d=load("BMresults/theta1.05_LL_w00.7.jld")
data = d["data"]
σz = d["σz"]
ϕ = p./qs
nσz = [sum(σz[i])/length(σz[i]) for i in eachindex(σz)]
plot(ϕ,nσz,"bo",markerfacecolor="none",ms=3,label=L"w_0/w_1=0.7")

xlabel(L"ϕ/ϕ_0")
ylabel(L"Tr⟨σ_z⟩")
legend(loc="right")
ylim([0,2.2])
tight_layout()
savefig("projected_sigmaz.pdf",transparent=true)
display(fig)
close(fig)

##
# params = Params(dθ=1.38π/180);
# hof = constructHofstadterLL(params,q=3,p=1,nLL=40,lk=128);

# kmesh = ( reshape(hof.k1,:,1)*params.g1 .+ reshape(hof.k2,1,:)*params.g2 ) / abs(params.g2);
# kx = real(kmesh);
# ky = imag(kmesh);

# ##
# fig,ax = subplots(2,2,figsize=(8,4))
# cnt = -1
# for r in 1:2 
#     for c in 1:2
#         pl = ax[r,c].contourf(kx,ky,hof.spectrum[hof.nH+cnt,:,:],cmap="Blues_r")
#         colorbar(pl,ax=ax[r,c])
#         ax[r,c].axis("equal")
#         cnt = cnt + 1
#     end
# end
# tight_layout()
# savefig("Angle1.38/lowest_LLsq3.pdf")
# display(fig)
# close(fig)

# ## 
# # cut 
# fig = figure(figsize=(4,3)) 
# maxLL = 2
# ϵcut = hof.spectrum[(hof.nH-maxLL):(hof.nH+maxLL+1),1:(hof.lk÷2+1),24]
# ϵcut = [ϵcut  hof.spectrum[(hof.nH-maxLL):(hof.nH+maxLL+1),hof.lk÷2+1,25:end]]

# for n in 1:size(ϵcut,1)
#     plot(ϵcut[n,:],"-",lw=0.5)
# end
# tight_layout()
# savefig("Angle1.38/lowest_LLs_cutq3.pdf")
# display(fig)
# close(fig)
