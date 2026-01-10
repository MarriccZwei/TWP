from ..Optimiser import Optimiser
from ..ConfigFiles import mainConfig as mc
from ..ConfigFiles import userConfig as uc
from ..Solution.processLoadCase import process_aeroelastic_load_case

import aerosandbox.numpy as np
import matplotlib.pyplot as plt

res = 20
k = 20
omegan = np.zeros((res, k), dtype=np.complex64)

velocities = np.linspace(10, mc.LC_INFO[3]['op'].velocity, res)
for i, velocity in enumerate(velocities):
    lci = mc.LC_INFO[3]
    lci['op'].velocity = velocity
    lci = [lci]
    optimiser = Optimiser(mc.DESVARS_INITIAL, lci, mc.CAD_STRS, mc.MATERIALS, mc.RES, mc.G0, mc.MTOM, mc.AIRFS, mc.LBUCKLSF,
                        mc.BOUNDS)

    print(f"Processing velocity: {velocity}...")
    omegan[i, :] = process_aeroelastic_load_case(optimiser.model, optimiser.lcs[0], k=k, returnOmegan=True)

print("Preparing plots...")
for j in range(omegan.shape[1]):
    plt.plot(velocities, omegan[:, j])

plt.savefig(uc.FW_SAVE_PATH+"OmegaVsVelocity.pdf")
plt.show()