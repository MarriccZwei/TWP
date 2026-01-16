from ..Optimiser import Optimiser
from ..ConfigFiles import mainConfig as mc
from ..ConfigFiles import userConfig as uc

import aerosandbox.numpy as np
import matplotlib.pyplot as plt

min_pow = 7
max_pow = 14
num_ks = max_pow-min_pow+1
margins = np.zeros((4, num_ks))
ks = np.logspace(min_pow, max_pow, num_ks)

for i, k in enumerate(ks):
    desvars = mc.DESVARS_INITIAL
    desvars['sks']=(k, 0, 0., k, 0., 0., 0., 1., 0.)
    print(f"analysis for k={k} ...")

    optimiser = Optimiser(desvars, mc.LC_INFO, mc.CAD_STRS, mc.MATERIALS, mc.RES, mc.G0, mc.MTOM, mc.AIRFS, mc.LBUCKLSF,
                      mc.BOUNDS)
    margins[:, i] = optimiser.simulate_constraints(optimiser.desvarvec())

logks = np.log10(ks)
plt.figure(figsize=(10, 10))
plt.plot(logks, margins[0, :], label="Quad stress ratio")
plt.plot(logks, margins[1, :], label="Beam stress ratio")
plt.plot(logks, margins[2, :], label="Load multiplier")
plt.plot(logks, margins[3, :], label="N complex freq.")
plt.legend()
plt.xlabel(r"$\log_{10}(k/k_0)\; k_0=1 N/m$")
plt.savefig(uc.FW_SAVE_PATH+"k_sensitivity.pdf")
plt.show()