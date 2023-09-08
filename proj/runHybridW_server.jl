using DelimitedFiles
# using PyPlot
fpath = pwd()
include(joinpath(fpath,"libs/Hofstadter_mod.jl"))
# ps = collect(1:2)
params = Params()
##
# for p in ps
    @time begin
        # w1=96.056
        # w0=0.7w1
        # params = Params(dθ=1.38π/180,w0=w0,w1=w1,ϵ=0.0) 
        params = Params()
        q = 149
        p = parse(Int,ARGS[1])
        ϕ = p//q
        println("ϕ/ϕ0 = ",ϕ)
        hof = initHofstadterHoppingElements(params,q=q,p=p)
        # ϵ = 1.5 .* hof.spectrum  # hof spectrum in vf/Lm not vf ktheta 
        ϵ = hof.spectrum
        σz = hof.σz

        # fname = "BMresults/theta1.38_q$(q)p$(p)_w00.7.txt"
        # fname = "Weizmann/q$(q)p$(p)_valleyKprime.txt"
        fname = "Weizmann/q$(q)p$(p).txt"
        writedlm(fname,[ϵ σz])
    end
# end


# ##
# function plot_hWS(flag=false)
#     fig = figure(figsize=(4,3))
#     ps = collect(1:2)
#     ps = [1;2;3;4;5;6]
#     q = 101
#     for p in ps
#         d=readdlm("Weizmann/q$(q)p$(p)_valleyKprime.txt")
#         data = d[:,1] * params.vf*params.kb*1.5
#         σz = d[:,2]
#         ϵ = data
#         ϕ = ones(2q) * p/q
#         plot(ϕ,ϵ,"g.",ms=2)
#     end

#     # xlim([0,0.52])
#     xlabel(L"ϕ/ϕ_0")
#     tight_layout()
#     display(fig)
#     if flag == true
#         savefig("figures/bmLL_spectrum.pdf",dpi=500,transparent=false)
#     end
#     close(fig)
# end

# plot_hWS(false)



# plot orbital magnetization 
function plot_orbital_magnetization_from_firsttwopoints(flag=false)
    # E = α 0.668 μB B, α is defined as E (meV) = α ϕ/ϕ0 
    # if slope in E (meV) vs ϕ/ϕ0 is 1.5 then it is 1μB
    
    # ϕs = 1 .// collect(17:18)
    ϕs = [1//149; 2//149]
    
    μs = collect(-6:0.15:6)
    moment = zeros(Float64,length(μs))
    νavg = zeros(Float64,length(μs))
    for iμ in eachindex(μs)
        Eν = zeros(Float64,2)
        νs = zeros(Float64,2)
        for iϕ in eachindex(ϕs)
            p = numerator(ϕs[iϕ])
            q = denominator(ϕs[iϕ])
            fname = joinpath(fpath,"Weizmann/q$(q)p$(p).txt")
            data = readdlm(fname)
            ϵ = data[:,1]* params.vf*params.kb*1.5

            μ = μs[iμ]
            νs[iϕ] = sum((sign.(μ .- ϵ).+1)./2)  /(length(ϵ)÷2) # per Moiré unit cell 
            Eν[iϕ] = sum( ϵ.*((sign.(μ .- ϵ).+1)./2) ) /(length(ϵ)÷2) - μ * νs[iϕ]
        end
        moment[iμ] = (Eν[2] - Eν[1])/(ϕs[1]-ϕs[2]) *0.668 # in units of Bohr magneton
        νavg[iμ] = (νs[1]+νs[2])/2
    end
    
    fig, ax = subplots(figsize=(4,3))
    ax.plot(μs,moment,"b-")
    ax.set_xlabel(L"μ")
    ax.set_ylabel(L"M_z (μ_B)")
    ax.set_title(L"ϕ/ϕ_0=1.5/149")
    ax1 = ax.twinx()
    ax1.plot(μs,νavg,"m-")
    ax1.set_ylabel(L"ν")
    tight_layout()
    display(fig)
    if (flag ==true)
        fname = joinpath(fpath,"figures/orbital_magnetization_hBN_vs_mu.pdf")
        savefig(fname,dpi=500)
    end
    close(fig)
end

plot_orbital_magnetization_from_firsttwopoints(false)

# ###

# plot orbital magnetization for fixed particle number
function plot_orbital_magnetization_from_firsttwopoints_fixedN(flag=false)
    # E = α 0.668 μB B, α is defined as E (meV) = α ϕ/ϕ0 
    # if slope in E (meV) vs ϕ/ϕ0 is 1.5 then it is 1μB
    
    # ϕs = 1 .// collect(17:18)
    ϕs = [1//149; 2//149]
    q = 149
    νs = collect(0:(2q)) ./(q)
    moment = zeros(Float64,length(νs))
    for iν in eachindex(νs)
        Eν = zeros(Float64,2)
        if (iν>1)
            for iϕ in eachindex(ϕs)
                p = numerator(ϕs[iϕ])
                q = denominator(ϕs[iϕ])
                fname = joinpath(fpath,"Weizmann/q$(q)p$(p).txt")
                data = readdlm(fname)
                # fname = joinpath(fpath,"Weizmann/q$(q)p$(p)_valleyKprime.txt")
                # data = [data; readdlm(fname)]
                # data = readdlm(fname)
                ϵ = sort(data[:,1])* params.vf*params.kb*1.5
                Eν[iϕ] = sum(ϵ[1:(iν-1)]) /(length(ϵ)÷2)
            end
            moment[iν] = (Eν[2] - Eν[1])/(ϕs[1]-ϕs[2]) *0.668 # in units of Bohr magneton
        end
    end
    
    fig, ax = subplots(figsize=(4,3))
    ax.plot(νs,moment,"b-")
    ax.set_xlabel(L"ν")
    ax.set_ylabel(L"M_z (μ_B)")
    ax.set_title(L"ϕ/ϕ_0=1.5/149")
    tight_layout()
    display(fig)
    if (flag ==true)
        fname = joinpath(fpath,"figures/orbital_magnetization_hBN_vs_mu_fixedN.pdf")
        savefig(fname,dpi=500)
    end
    close(fig)
end

plot_orbital_magnetization_from_firsttwopoints_fixedN(true)


# plot orbital magnetization 
function plot_orbital_magnetization_vs_filling(flag=false)
    # E = α 0.668 μB B, α is defined as E (meV) = α ϕ/ϕ0 
    # if slope in E (meV) vs ϕ/ϕ0 is 1.5 then it is 1μB
    
    ϕs = [1//101; 2//101]
    μs = collect(-6:0.1:6)
    moment = zeros(Float64,length(μs))
    νavg = zeros(Float64,length(μs))
    for iμ in eachindex(μs)
        Eν = zeros(Float64,2)
        νs = zeros(Float64,2)
        for iϕ in eachindex(ϕs)
            p = numerator(ϕs[iϕ])
            q = denominator(ϕs[iϕ])

            fname = joinpath(fpath,"Weizmann/q$(q)p$(p).txt")
            data = readdlm(fname)
            ϵ = data[:,1]* params.vf*params.kb*1.5

            μ = μs[iμ]
            νs[iϕ] = sum((sign.(μ .- ϵ).+1)./2)  /(length(ϵ)÷2) # per Moiré unit cell 
            Eν[iϕ] = sum( ϵ.*((sign.(μ .- ϵ).+1)./2) ) /(length(ϵ)÷2) - μ * νs[iϕ]
        end
        moment[iμ] = (Eν[2] - Eν[1])/(ϕs[1]-ϕs[2]) *0.668 # in units of Bohr magneton
        νavg[iμ] = (νs[1]+νs[2])/2
    end
    fig, ax = subplots(figsize=(4,3))
    ax.plot(νavg,moment,"b.")
    ax.set_xlabel(L"ν")
    ax.set_ylabel(L"M_z (μ_B)")
    ax.set_title(L"ϕ/ϕ_0=1/100")
    tight_layout()
    display(fig)
    if (flag ==true)
        fname = joinpath(fpath,"figures/orbital_magnetization_hBN_vs_filling.pdf")
        savefig(fname,dpi=500)
    end
    close(fig)
end

# plot_orbital_magnetization_vs_filling(true)




# plot orbital magnetization for fixed particle number
function plot_orbital_magnetization_fixedN(flag=false)
    # E = α 0.668 μB B, α is defined as E (meV) = α ϕ/ϕ0 
    # if slope in E (meV) vs ϕ/ϕ0 is 1.5 then it is 1μB
    
    # ϕs = 1 .// collect(17:18)
    ϕs = collect(1:6) .// 149
    q = denominator(ϕs[1])
    νs = collect(0:(2q)) ./(q)
    Eν = zeros(Float64,length(νs),length(ϕs))

    for iϕ in eachindex(ϕs)
        p = numerator(ϕs[iϕ])
        q = denominator(ϕs[iϕ])
        fname = joinpath(fpath,"Weizmann/q$(q)p$(p).txt")
        data = readdlm(fname)
        fname = joinpath(fpath,"Weizmann/q$(q)p$(p)_valleyKprime.txt")
        data = [data; readdlm(fname)]
        # data = readdlm(fname)
        ϵ = sort(data[:,1])* params.vf*params.kb*1.5
        for iν in eachindex(νs)
            if (iν>1)
                Eν[iν,iϕ] = sum(ϵ[1:(iν-1)]) /(length(ϵ)÷4)
            end
        end
    end
    
    fig, ax = subplots(figsize=(4,3))
    ax.plot(ϕs,Eν[100:5:120,:]',"x-")
    ax.set_xlabel(L"ϕ/ϕ_0")
    ax.set_ylabel(L"E")
    ax.set_xlim([0,0.08])
    tight_layout()
    display(fig)
    if (flag ==true)
        fname = joinpath(fpath,"figures/orbital_magnetization_hBN_vs_mu_fixedN.pdf")
        savefig(fname,dpi=500)
    end
    close(fig)
end

plot_orbital_magnetization_fixedN(false)