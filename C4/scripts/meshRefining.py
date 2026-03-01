from ..Optimiser import Optimiser
from ..ConfigFiles import mainConfig as mc
from ..ConfigFiles import userConfig as uc

for n, in zip([8, 9, 10]):
    optimiser = Optimiser(mc.DESVARS_INITIAL, mc.LC_INFO, mc.CADSTRS, mc.GEOM_SOURCE, mc.HYPERPARAMS, mc.MASSES, n, mc.MATERIALS, mc.RES, mc.G0, mc.MTOM, mc.AIRFS, mc.LBUCKLSF,
                        mc.BOUNDS)
    print(optimiser.simulate_constraints(optimiser.desvarvec(), True, uc.REFINE_SAVE_PATH+f"{n}\\"))
    print(optimiser.objective(optimiser.desvarvec()))