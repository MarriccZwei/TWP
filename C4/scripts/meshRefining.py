from ..Optimiser import Optimiser
from ..ConfigFiles import mainConfig as mc
from ..ConfigFiles import userConfig as uc
import matplotlib.pyplot as plt

ns = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
beam_stresses = list()
quad_stresses = list()
for n in ns:
    optimiser = Optimiser(mc.DESVARS_INITIAL, [mc.LC_INFO[0]], mc.CAD_STRS, mc.GEOM_SOURCE, mc.HYPERPARAMS, mc.MASSES, n, mc.MATERIALS, mc.RES, mc.G0, mc.MTOM, mc.AIRFS, mc.LBUCKLSF,
                        mc.BOUNDS)
    dvv = optimiser.simulate_constraints(optimiser.desvarvec(), True, uc.REFINE_SAVE_PATH+f"{n}\\")
    quad_stresses.append(dvv[0])
    beam_stresses.append(dvv[1])
    print(dvv)
    print(optimiser.objective(optimiser.desvarvec()))

plt.plot(ns, quad_stresses, label="quad elements")
plt.plot(ns, beam_stresses, label="beam elements")
plt.ylabel("max. stress / failure stress")
plt.xlabel("nodes per sheet width")
plt.legend()
plt.savefig(uc.FW_SAVE_PATH+"StressConvergence.pdf")
plt.show()