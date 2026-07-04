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

rho_fs = [100, 175, 250.2, 325, 400]
materials = mc.MATERIALS.copy()
objectives_report = "=====FINAL REPORT=====\n"

for rho_f in rho_fs:
    materials['RHO_FOAM'] = rho_f
    materials['E_FOAM'] = materials["E_ALU"]*(rho_f/materials["RHO_ALU"])**2
    materials['SF_FOAM'] = 450e6 * .3*(rho_f/materials["RHO_ALU"])**1.5
    optimiser = Optimiser(desvarsInit, [mc.LC_INFO[0]], mc.GEOM_SOURCE, mc.HYPERPARAMS, mc.MASSES, mc.N, materials, resNoBuckl, mc.G0, mc.MTOM, mc.NAIRFS, mc.LBUCKLSF,
                        mc.BOUNDS, logEveryNIters=1, loadCasesJoint=mc.LC_INFO)
    result = opt.minimize(optimiser.objective, optimiser.desvarvec(), method='COBYLA', constraints=optimiser.constraint(),
                        options={'rhobeg':.2})
    desvarsResult = optimiser.desvars_from_vec(result.x)
    print(f"Converged to: {desvarsResult},\nwith success: {result.success}\nand message: {result.message}")
    
    objectives_report += f"For rho foam: {rho_f}, objective: {optimiser.objective(optimiser.desvarvec(desvarsResult))}\nOptimisantion result: {desvarsResult}\n"
    gc.collect()

    verifier = Optimiser(desvarsResult, mc.LC_INFO, mc.GEOM_SOURCE, mc.HYPERPARAMS, mc.MASSES, mc.N, materials, mc.RES, mc.G0, mc.MTOM, mc.NAIRFS, mc.LBUCKLSF,
                      mc.BOUNDS)
    objectives_report += f"failure margins: {verifier.forward(verifier.desvarvec())}\n"
    gc.collect()

print(objectives_report)