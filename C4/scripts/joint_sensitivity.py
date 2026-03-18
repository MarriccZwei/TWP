from ..Optimiser import Optimiser
from ..ConfigFiles import mainConfig as mc
from ..ConfigFiles import userConfig as uc

import scipy.optimize as opt
import aerosandbox.numpy as np
import gc

desvarsInit = dict()
for key in mc.BOUNDS[0].keys(): #centered initial conditions
    desvarsInit[key]=(mc.BOUNDS[0][key]+mc.BOUNDS[1][key])/2

resNoBuckl = mc.RES.copy()
resNoBuckl["klb"] = 0 #exclude linear buckling from iterative analysis as it should be not constraining

rjcs = np.linspace(.03/5, .1/5, 8) #from: Sizing and Layout Design of an Aeroelastic Wingbox through Nested Optimization - Bret K. Stanford, Christine V. Jutte, Christian A. Coker
hypers = mc.HYPERPARAMS.copy()
ojectives_report = "=====FINAL REPORT=====\n"

for rjc in rjcs:
    hypers["rj/c"] = rjc
    optimiser = Optimiser(desvarsInit, [mc.LC_INFO[0]], mc.GEOM_SOURCE, hypers, mc.MASSES, mc.N, mc.MATERIALS, resNoBuckl, mc.G0, mc.MTOM, mc.NAIRFS, mc.LBUCKLSF,
                        mc.BOUNDS, logEveryNIters=1)
    result = opt.minimize(optimiser.objective, optimiser.desvarvec(), method='COBYLA', constraints=optimiser.constraint(),
                        options={'rhobeg':.2})
    desvarsResult = optimiser.desvars_from_vec(result.x)
    print(f"Converged to: {desvarsResult},\nwith success: {result.success}\nand message: {result.message}")
    gc.collect()
    objectives_report += f"For rjc: {rjc}, objective: {optimiser.objective(optimiser.desvarvec(desvarsResult))}\nOptimisantion result: {desvarsResult}\n"

print(objectives_report)