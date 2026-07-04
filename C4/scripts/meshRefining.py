from ..Optimiser import Optimiser
from ..ConfigFiles import mainConfig as mc
from ..ConfigFiles import userConfig as uc
import matplotlib.pyplot as plt

ns = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
# beam_stresses = list()
# quad_stresses = list()
lms = list()

lcinfo = [mc.LC_INFO[3]]
lcinfo[0]["Ttot"] = 112800. # [N]

for n in ns:
    optimiser = Optimiser(mc.DESVARS_INITIAL, lcinfo, mc.GEOM_SOURCE, mc.HYPERPARAMS, mc.MASSES, n, mc.MATERIALS, mc.RES, mc.G0, mc.MTOM, mc.NAIRFS, mc.LBUCKLSF,
                        mc.BOUNDS, loadCasesJoint=mc.LC_INFO)
    dvv = optimiser.forward(optimiser.desvarvec(), True, uc.REFINE_SAVE_PATH+f"{n}\\")
    # quad_stresses.append(dvv[0])
    # beam_stresses.append(dvv[1])
    lms.append(dvv[2])
    print(dvv)
    print(optimiser.objective(optimiser.desvarvec()))

# plt.plot(ns, quad_stresses, label="quad elements")
# plt.plot(ns, beam_stresses, label="beam elements")
plt.plot(ns, lms)
plt.ylabel("Buckling load multiplier")
#plt.ylabel("max. stress / failure stress")
plt.xlabel("Nodes per sheet width")
#plt.legend()
plt.savefig(uc.FW_SAVE_PATH+"BucklingConvergence.pdf")
plt.show()