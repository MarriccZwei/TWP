from ..Optimiser import Optimiser
from ..ConfigFiles import mainConfig as mc
from ..ConfigFiles import userConfig as uc
from ..Solution.processLoadCase import process_aeroelastic_load_case

import aerosandbox.numpy as np
import matplotlib.pyplot as plt
import time

Ns = [8, 9, 10, 11, 12]
Nfoils = [10]*5
nfreq = 5
omegan = np.zeros((nfreq, len(Ns)), dtype=np.complex64)

for i, N, Nfoil in zip(range(len(Ns)), Ns, Nfoils):
    t1 = time.time()
    optimiser = Optimiser(mc.DESVARS_INITIAL, [mc.LC_INFO[2]], mc.GEOM_SOURCE, mc.HYPERPARAMS, mc.MASSES, N, mc.MATERIALS, mc.RES, mc.G0, mc.MTOM, Nfoil, mc.LBUCKLSF,
                        mc.BOUNDS)
    omegan[:, i] = process_aeroelastic_load_case(optimiser.model, optimiser.lcs[0], True,  uc.REFINE_SAVE_PATH+f"{N}\\", mc.RES["kfl"], True)[:nfreq]  
    print(f"Processed {Nfoil} in {time.time()-t1} [s], freqs [rad/s]: {omegan[:, i]}\n")  

for j in range(nfreq): 
    plt.plot(Ns, np.real(omegan[j, :]), label=f"frequency {j+1}")
plt.legend()
plt.ylabel("Natural frequency [rad/s]")
plt.xlabel("Number of nodes per wavelength")
plt.savefig(uc.FW_SAVE_PATH+"FlutterConvergence.pdf")
plt.show()
