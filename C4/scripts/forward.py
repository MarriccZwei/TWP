from ..Optimiser import Optimiser
from ..ConfigFiles import mainConfig as mc
from ..ConfigFiles import userConfig as uc

optimiser = Optimiser(mc.DESVARS_INITIAL, mc.LC_INFO, mc.CAD_STRS, mc.MATERIALS, mc.RES, mc.G0, mc.MTOM, mc.AIRFS, mc.LBUCKLSF,
                      mc.BOUNDS)
print(optimiser.simulate_constraints(optimiser.desvarvec(), True, uc.FW_SAVE_PATH))
print(optimiser.objective(optimiser.desvarvec()))