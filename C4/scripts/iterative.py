from ..Optimiser import Optimiser
from ..ConfigFiles import mainConfig as mc
from ..ConfigFiles import userConfig as uc

import scipy.optimize as opt

desvarsInit = dict()
for key in mc.BOUNDS[0].keys(): #centered initial conditions
    desvarsInit[key]=(mc.BOUNDS[0][key]+mc.BOUNDS[1][key])/2

resNoBuckl = mc.RES.copy()
resNoBuckl["klb"] = 0 #exclude linear buckling from iterative analysis as it should be not constraining

optimiser = Optimiser(desvarsInit, [mc.LC_INFO[0]], mc.GEOM_SOURCE, mc.HYPERPARAMS, mc.MASSES, mc.N, mc.MATERIALS, resNoBuckl, mc.G0, mc.MTOM, mc.NAIRFS, mc.LBUCKLSF,
                      mc.BOUNDS, logEveryNIters=1)
result = opt.minimize(optimiser.objective, optimiser.desvarvec(), method='COBYLA', constraints=optimiser.constraint(),
                      options={'rhobeg':.2})
desvarsResult = optimiser.desvars_from_vec(result.x)
print(f"Converged to: {desvarsResult},\nwith success: {result.success}\nand message: {result.message}")

#checking constraints for the converged design
verifier = Optimiser(desvarsResult, mc.LC_INFO, mc.MATERIALS, mc.RES, mc.G0, mc.MTOM, mc.NAIRFS, mc.LBUCKLSF,
                     mc.BOUNDS)
print(verifier.simulate_constraints(verifier.desvarvec(), True, uc.FW_SAVE_PATH))