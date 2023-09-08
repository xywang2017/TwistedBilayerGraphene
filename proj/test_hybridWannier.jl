using PyPlot
using Printf
fpath = pwd()
include(joinpath(fpath,"libs/hybridWannier_mod.jl"))

##
w1=96.056
w0=0.8w1
dθ = 1.05
params = Params(dθ=dθ*π/180,ϵ=0.0,w0=w0,w1=w1);
Latt = Lattice()
initLattice(Latt,params;lk=64)
blk = HBM()
initHBM(blk,Latt,params;lg=9)

##
HW = initHybridWannier(blk,Latt,params);
HW_R = initHybridWannierRealSpace(blk,Latt,params,HW;rdim=61);
for nk in 1:64
    if nk < 10 
        str = "00$(nk)"
    else
        str = "0$(nk)"
    end
    plot_HybridWannier_RealSpaceCombined(HW_R,HW,params,nk=nk,savename="tmp/$(str).png");
end
