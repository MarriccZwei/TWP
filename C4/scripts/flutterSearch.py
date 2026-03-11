from ..Optimiser import Optimiser
from ..ConfigFiles import mainConfig as mc
from ..ConfigFiles import userConfig as uc

Ns = [5, 6, 7, 8, 9]
Nfoils = [8, 9, 10, 11, 12]

for N, Nfoil in zip(Ns, Nfoils):
    optimiser = Optimiser(mc.DESVARS_INITIAL, mc.LC_INFO, mc.GEOM_SOURCE, mc.HYPERPARAMS, mc.MASSES, mc.N, mc.MATERIALS, mc.RES, mc.G0, mc.MTOM, mc.NAIRFS, mc.LBUCKLSF,
                        mc.BOUNDS)
    #TODO: finish
        