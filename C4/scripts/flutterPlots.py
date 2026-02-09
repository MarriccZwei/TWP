from ..Optimiser import Optimiser
from ..ConfigFiles import mainConfig as mc
from ..ConfigFiles import userConfig as uc

import aerosandbox.numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg as ssl

res = 21
k = 30
omegan = np.zeros((res, k))
VMIN = 200
VMAX = 400

velocities = np.linspace(VMIN, VMAX, res)
lci = mc.LC_INFO[2]
lci['op'].velocity = VMIN
lci = [lci]
optimiser = Optimiser(mc.DESVARS_INITIAL, lci, mc.CAD_STRS, mc.MATERIALS, mc.RES, mc.G0, mc.MTOM, mc.AIRFS, mc.LBUCKLSF,
                        mc.BOUNDS)
for i, velocity in enumerate(velocities):
    ratio = (velocity/VMIN)**2
    KAuu = ratio*optimiser.model.uu_matrix(optimiser.lcs[0].KA)   

    print(f"Processing velocity: {velocity}...")
    omegan[i, :], _ = ssl.eigs(A=optimiser.model.KC0uu-KAuu, M=optimiser.model.Muu, k=k, sigma=-1., which='LM')
    omegan[i, :] = np.sqrt(omegan[i, :])

print("Preparing plots...")
omegan.sort(1)
for j in range(omegan.shape[1]):
    plt.plot(velocities, omegan[:, j])
plt.ylabel("Natural Frequency [Hz]")
plt.xlabel("Airspeed [m/s]")

plt.savefig(uc.FW_SAVE_PATH+"OmegaVsVelocity.pdf")
plt.show()