from ..Optimiser import Optimiser
from ..ConfigFiles import mainConfig as mc
from ..ConfigFiles import userConfig as uc

import scipy.optimize as opt
import gc

desvarsInit = dict()
for key in mc.BOUNDS[0].keys(): #centered initial conditions
    desvarsInit[key]=(mc.BOUNDS[0][key]+mc.BOUNDS[1][key])/2

resNoBuckl = mc.RES.copy()
resNoBuckl["klb"] = 0 #exclude linear buckling from iterative analysis as it should be not constraining

rail_ds = [.01,.02,.03, .04, .05]
hypers = mc.HYPERPARAMS.copy()
ojectives_report = "=====FINAL REPORT=====\n"

for rail_d in rail_ds:
    hypers["d"] = rail_d
    optimiser = Optimiser(desvarsInit, [mc.LC_INFO[0]], mc.GEOM_SOURCE, hypers, mc.MASSES, mc.N, mc.MATERIALS, resNoBuckl, mc.G0, mc.MTOM, mc.NAIRFS, mc.LBUCKLSF,
                        mc.BOUNDS, logEveryNIters=1)
    result = opt.minimize(optimiser.objective, optimiser.desvarvec(), method='COBYLA', constraints=optimiser.constraint(),
                        options={'rhobeg':.2})
    desvarsResult = optimiser.desvars_from_vec(result.x)
    print(f"Converged to: {desvarsResult},\nwith success: {result.success}\nand message: {result.message}")
    gc.collect()
    objectives_report += f"For rail_d: {rail_d}, objective: {optimiser.objective(optimiser.desvarvec(desvarsResult))}\nOptimisantion result: {desvarsResult}\n"

print(objectives_report)