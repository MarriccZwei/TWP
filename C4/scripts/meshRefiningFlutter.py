from ..Optimiser import Optimiser
from ..ConfigFiles import mainConfig as mc
from ..ConfigFiles import userConfig as uc
from ..Solution.processLoadCase import process_aeroelastic_load_case

import aerosandbox.numpy as np
import matplotlib.pyplot as plt
import time

import pickle

Ns = [5, 6, 7, 8, 9, 10, 11, 12]
Nfoils = [11]*8
nfreq = 5
omegan = np.zeros((nfreq, len(Ns)), dtype=np.complex64)
res = mc.RES.copy()

for i, N, Nfoil in zip(range(len(Ns)), Ns, Nfoils):
    res["nneighs"] = N**2//4
    t1 = time.time()
    optimiser = Optimiser(mc.DESVARS_INITIAL, [mc.LC_INFO[2]], mc.GEOM_SOURCE, mc.HYPERPARAMS, mc.MASSES, N, mc.MATERIALS, mc.RES, mc.G0, mc.MTOM, Nfoil, mc.LBUCKLSF,
                        mc.BOUNDS, loadCasesJoint=mc.LC_INFO)
    omegas_returned, peigvecs = process_aeroelastic_load_case(optimiser.model, optimiser.lcs[0], True,  uc.REFINE_SAVE_PATH+f"{N}\\", mc.RES["kfl"], False, True)
    omegan[:, i] = omegas_returned[:nfreq]  
    print(f"Processed N: {N}, nfoil: {Nfoil} in {time.time()-t1} [s], freqs [rad/s]: {omegan[:, i]}\n") 

    with open(uc.REFINE_SAVE_PATH+f"{N}\\freqmodes.pcl", 'wb+') as f:
        pickle.dump((omegan[:nfreq, i], peigvecs[:, :nfreq]), f)

for j in range(nfreq): 
    plt.plot(Ns, np.real(omegan[j, :]), label=f"frequency {j+1}")
plt.legend()
plt.ylabel("Natural frequency [rad/s]")
plt.xlabel("Nodes per sheet width")
plt.savefig(uc.FW_SAVE_PATH+"FlutterConvergence.pdf")
plt.show()
