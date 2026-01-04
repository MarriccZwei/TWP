from ..Optimiser import Optimiser
from ..ConfigFiles import mainConfig as mc
from ..ConfigFiles import userConfig as uc

import scipy.optimize as opt

optimiser = Optimiser(mc.DESVARS_INITIAL, mc.LC_INFO, mc.CAD_STRS, mc.MATERIALS, mc.RES, mc.G0, mc.MTOM, mc.AIRFS, mc.LBUCKLSF,
                      mc.WMAX, logEveryNIters=1)
result = opt.minimize(optimiser.objective, optimiser.desvarvec(), method='COBYLA', constraints=optimiser.constraint())
desvarsResult = Optimiser.desvars_from_vec(result.x)
print(f"Converged to: {desvarsResult},\nwith success: {result.success}\nand message: {result.message}")

#checking constraints for the converged design
verifier = Optimiser(desvarsResult, mc.LC_INFO, mc.CAD_STRS, mc.MATERIALS, mc.RES, mc.G0, mc.MTOM, mc.AIRFS, mc.LBUCKLSF,
                     mc.WMAX)
print(verifier.simulate_constraints(verifier.desvarvec(), True, uc.FW_SAVE_PATH))