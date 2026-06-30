from ..Optimiser import Optimiser
from ..ConfigFiles import mainConfig as mc
from ..ConfigFiles import userConfig as uc
from ..Solution.processLoadCase import process_aeroelastic_load_case

import aerosandbox.numpy as np
import matplotlib.pyplot as plt
import time

import pickle

def MAC(self, vec1, vec2):
    return (np.dot(vec1, vec2)**2 / np.dot(vec1, vec1) / np.dot(vec2, vec2))

Ns = [12]*8
nfreq = 5
ks = [1e6, 1e8, 1e10, 1e12, 1e14, 1e16, 1e18, 1e20]
omegan = np.zeros((nfreq, len(Ns)), dtype=np.complex64)
res = mc.RES.copy()
nairf = 11

for i, N, kspr in zip(range(len(Ns)), Ns, ks):
    res["sks"] = (kspr, 0., 0., kspr, 0., 0., 0., 1., 0.)
    t1 = time.time()
    optimiser = Optimiser(mc.DESVARS_INITIAL, [mc.LC_INFO[2]], mc.GEOM_SOURCE, mc.HYPERPARAMS, mc.MASSES, N, mc.MATERIALS, res, mc.G0, mc.MTOM, nairf, mc.LBUCKLSF,
                        mc.BOUNDS)
    omegas_returned, peigvecs = process_aeroelastic_load_case(optimiser.model, optimiser.lcs[0], True,  uc.REFINE_SAVE_PATH+f"{i+4}\\", mc.RES["kfl"], False, True)
    omegan[:, i] = omegas_returned[:nfreq]  
    print(f"Processed k: {kspr}, N: {N} in {time.time()-t1} [s], freqs [rad/s]: {omegan[:, i]}\n") 

    with open(uc.REFINE_SAVE_PATH+f"{i+4}\\freqmodes.pcl", 'wb+') as f:
        pickle.dump((omegan[:nfreq, i], peigvecs[:, :nfreq]), f)

for j in range(nfreq): 
    plt.plot(ks, np.real(omegan[j, :]), label=f"frequency {j+1}")
plt.legend()
plt.ylabel("Natural frequency [rad/s]")
plt.xlabel("Nodes per sheet width")
plt.xscale('log')
plt.savefig(uc.FW_SAVE_PATH+"FlutterConvergence.pdf")
plt.show()