from ..Optimiser import Optimiser
from ..ConfigFiles import mainConfig as mc
from ..ConfigFiles import userConfig as uc

import aerosandbox as asb
import gc
import aerosandbox.numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg as ssl

res = 21
k = 2
omegan = np.zeros((res, k))
sos = asb.Atmosphere(7000.).speed_of_sound()
MACHMIN = .6
MACHMAX = .95
VMIN = MACHMIN*sos
VMAX = MACHMAX*sos
N = 6

machs = np.linspace(MACHMIN, MACHMAX, res)
lci = mc.LC_INFO[2]
lci['op'].velocity = VMIN
lci = [lci]
optimiser = Optimiser(mc.DESVARS_INITIAL, lci, mc.GEOM_SOURCE, mc.HYPERPARAMS, mc.MASSES, N, mc.MATERIALS, mc.RES, mc.G0, mc.MTOM, mc.NAIRFS, mc.LBUCKLSF,
                        mc.BOUNDS)
for i, mach in enumerate(machs):
    ratio = (mach/MACHMIN)**2 * np.sqrt((1-MACHMIN**2)/(1-mach**2))
    KAuu = ratio*optimiser.model.uu_matrix(optimiser.lcs[0].KA)   

    print(f"Processing Mach number {mach}...")
    omegan[i, :], _ = ssl.eigs(A=optimiser.model.KC0uu-KAuu, M=optimiser.model.Muu, k=k, sigma=-1., which='LM')
    omegan[i, :] = np.sqrt(omegan[i, :])
    gc.collect()

print("Preparing plots...")
omegan.sort(1)
for j in range(omegan.shape[1]):
    plt.plot(machs, omegan[:, j])
plt.ylabel("Natural Frequency [Hz]")
plt.xlabel("Mach number [-]")

plt.savefig(uc.FW_SAVE_PATH+"OmegaVsVelocity.pdf")
plt.show()