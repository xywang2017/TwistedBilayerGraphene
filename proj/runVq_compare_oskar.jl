using PyPlot
using DelimitedFiles
fpath = pwd()
include(joinpath(fpath,"libs/HofstadterVq_modv3.jl")) # single k calculation
# include(joinpath(fpath,"libs/HofstadterVq_modv3_kselect.jl")) # multi k calculation

# ϕ = [1 // parse(Int,ARGS[1])]
ϕ = [1//15]
w1 = 96.056
w0=0.7w1
dθ = 1.05
params = Params(dθ=dθ*π/180,ϵ=0.0,w0=w0,w1=w1);
##
for iϕ in eachindex(ϕ)
    @time begin
        p = numerator(ϕ[iϕ])
        q = denominator(ϕ[iϕ])
        println(ϕ[iϕ]," w0=",w0," w1=",w1)
        hof = HofstadterVq()
        lk = 8
        initHofstadterVq(hof,params,p=p,q=q,lk=lk);
        ϵ = zeros(Float64,size(hof.H,2),size(hof.H,3))
        σz = zeros(Float64,size(hof.H,2),size(hof.H,3))
        for iν in eachindex(hof.ν)
            for ik in 1:size(hof.H,3)
                F = eigen(Hermitian(hof.H[:,:,ik,iν]))
                σz[:,ik] = diag(real(F.vectors'*hof.Σz[:,:,ik]*F.vectors))
                ϵ[:,ik] = F.values
            end
            νval = hof.ν[iν]
            fname = "q$(q)_w0$(w0)_nu$(νval)_xiaoyu.txt"
            writedlm(fname,[ϵ[:] σz[:]])      
        end
    end
end

# 
fig = figure(figsize=(3,3))
fname = "q15_w0$(w0)_nu2_xiaoyu.txt"
xw = readdlm(fname)[:,1]

# --- q = 15
# hole
oskar_hl = [-0.421479, -0.418999, -0.41181, -0.407661, -0.398588, -0.394299,
-0.38755, -0.383402, -0.372421, -0.369975, -0.364579, -0.362531,
-0.346327, -0.248894, -0.240045, -0.238102, -0.231632, -0.225181,
-0.212513, -0.199092, -0.178956, -0.153695, -0.124897, -0.0829862,
-0.0477005, -0.0261877, 0.0576306, 0.167572, 0.233823, 0.699027]
# electron
oskar_el=[0.916717, 2.02699, 2.24662, 2.4852, 2.73799, 2.89867, 2.975, 3.04854,
3.14407, 3.22572, 3.29115, 3.3488, 3.37982, 3.41502, 3.43601,
3.45934, 3.49364, 3.50424, 3.63348, 3.65639, 3.65803, 3.69005,
3.71883, 3.73162, 3.75566, 3.75912, 3.78368, 3.79676, 3.80695, 3.81144]
# cnp 
oskar_cnp=[0.807773, 1.09254, 1.23838, 1.25357, 1.33989, 1.39707, 1.44134,
1.46431, 1.47411, 1.4999, 1.51874, 1.53613, 1.54455, 1.55769,
1.56099, 1.56636, 1.57077, 1.5975, 1.66685, 1.6921, 1.69464, 1.69545,
1.69684, 1.70028, 1.70359, 1.70554, 1.71089, 1.71124, 1.71621, 1.7168]

plot(eachindex(xw),2xw,"r+",label="XW")
plot(eachindex(oskar_hl),oskar_el,"bx",label="Oskar")
title("hole")
legend()
tight_layout()
display(fig)
close(fig)