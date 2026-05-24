from ..Optimiser import Optimiser
from ..ConfigFiles import mainConfig as mc
from ..ConfigFiles import userConfig as uc
import matplotlib.pyplot as plt

ns = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
beam_stresses = list()
quad_stresses = list()

lcinfo = [mc.LC_INFO[3]]
lcinfo[0]["Ttot"] = 112800. # [N]

res_no_buckl = mc.RES
res_no_buckl["klb"] = 0

for n in ns:
    optimiser = Optimiser(mc.DESVARS_INITIAL, lcinfo, mc.GEOM_SOURCE, mc.HYPERPARAMS, mc.MASSES, n, mc.MATERIALS, res_no_buckl, mc.G0, mc.MTOM, mc.NAIRFS, mc.LBUCKLSF,
                        mc.BOUNDS)
    dvv = optimiser.simulate_constraints(optimiser.desvarvec(), True, uc.REFINE_SAVE_PATH+f"{n}\\")
    quad_stresses.append(dvv[0])
    beam_stresses.append(dvv[1])
    print(dvv)
    print(optimiser.objective(optimiser.desvarvec()))

plt.plot(ns, quad_stresses, label="quad elements")
plt.plot(ns, beam_stresses, label="beam elements")
plt.ylabel("Max. stress / failure stress")
plt.xlabel("Nodes per sheet width")
plt.legend()
plt.savefig(uc.FW_SAVE_PATH+"StressConvergence.pdf")
plt.show()