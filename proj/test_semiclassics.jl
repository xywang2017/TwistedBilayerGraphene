using PyPlot
using FFTW
using Printf
using DelimitedFiles
fpath = pwd()
include(joinpath(fpath,"libs/Hoftstadter_mod.jl"))

## 
# Testing term 2, in particular on the energetics and hermiticity
params = Params(dθ = 1.83π/180)
q=33

# ϵ0 = params.vf * params.kb
@printf("q=%d\n",q)
hof,blk,basis= initHofstadterHoppingElements(q,params);


##

##
ham = hof.Term1 - hof.Term2 
vals = eigvals((ham+ham')/2)
fig = figure()
plot(vals,"g.",ms=2)
display(fig)
close(fig)


##
fig = figure()
tmp = abs.(hof.Term2-Term3)
pl=imshow(tmp[1:2:end,1:2:end],origin="lower")
colorbar(pl)
display(fig)
close(fig)