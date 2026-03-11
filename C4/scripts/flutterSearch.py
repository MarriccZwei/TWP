from ..Optimiser import Optimiser
from ..ConfigFiles import mainConfig as mc
from ..ConfigFiles import userConfig as uc

N = [5, 6, 7, 8, 9]
NAIRFS = [7, 8, 9, 10, 11]

optimiser = Optimiser(mc.DESVARS_INITIAL, mc.LC_INFO, mc.GEOM_SOURCE, mc.HYPERPARAMS, mc.MASSES, mc.N, mc.MATERIALS, mc.RES, mc.G0, mc.MTOM, mc.NAIRFS, mc.LBUCKLSF,
                      mc.BOUNDS)
print(optimiser.simulate_constraints(optimiser.desvarvec(), True, uc.FW_SAVE_PATH))
print(optimiser.objective(optimiser.desvarvec()))
        