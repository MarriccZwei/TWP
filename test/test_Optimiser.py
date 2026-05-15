from ..C4.Optimiser import Optimiser
from ..C4.ConfigFiles import mainConfig as mc
from ..C4.ConfigFiles import userConfig as uc

import aerosandbox.numpy as np

#lc = LoadCase(1., 76000, model.N, 9.81, 112800, asb.OperatingPoint(asb.Atmosphere(0.), alpha=10., velocity=90.), les, tes, airfs, 
#                  nneighs=5, cres=8, bres=20, nlg=1.5, bank=6

lcinf = [mc.LC_INFO[3]]
lcinf[0]['Ttot'] = mc.LC_INFO[0]['Ttot']

optimiser = Optimiser(mc.DESVARS_INITIAL, lcinf, mc.GEOM_SOURCE, mc.HYPERPARAMS, mc.MASSES, mc.N, mc.MATERIALS, mc.RES, mc.G0, mc.MTOM, mc.NAIRFS, mc.LBUCKLSF,
                      mc.BOUNDS)

assert np.allclose(optimiser.desvarvec(optimiser.desvars_from_vec(optimiser.desvarvec())), optimiser.desvarvec())

print(optimiser.simulate_constraints(optimiser.desvarvec(), True, uc.FW_SAVE_PATH))
print(optimiser.objective(optimiser.desvarvec()))