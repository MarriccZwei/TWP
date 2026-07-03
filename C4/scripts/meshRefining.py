from ..Optimiser import Optimiser
from ..ConfigFiles import mainConfig as mc
from ..ConfigFiles import userConfig as uc
import matplotlib.pyplot as plt
import numpy as np

ns = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]#, 13, 14, 15, 16]
# beam_stresses = list()
# quad_stresses = list()
lms = list()

desvars = {'(2t/H)_Sq': np.float64(0.14779078527131706), '(2t/H)_Pq': np.float64(0.1941012111518347), '(2t/H)_Aq': np.float64(0.17841345464904979), 'W_bb': np.float64(0.01578562454259829), 'W_mb': np.float64(0.018572223880988704), 'W_lb': np.float64(0.01574421166844625), 'ds': np.float64(0.015763858666424515), 'de': np.float64(0.01780049033655861), '(2t/H)_sq': np.float64(0.16046288064007755), '(2t/H)_pq': np.float64(0.10110644736027476), '(2t/H)_aq': np.float64(0.12659052092093015)}


lcinfo = [mc.LC_INFO[0]]

for n in ns:
    optimiser = Optimiser(desvars, lcinfo, mc.GEOM_SOURCE, mc.HYPERPARAMS, mc.MASSES, n, mc.MATERIALS, mc.RES, mc.G0, mc.MTOM, mc.NAIRFS, mc.LBUCKLSF,
                        mc.BOUNDS)
    assert np.isclose(optimiser.lcs[0].nlg, 0)
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