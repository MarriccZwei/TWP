from ..Optimiser import Optimiser
from ..ConfigFiles import mainConfig as mc
from ..ConfigFiles import userConfig as uc
from ..Solution.processLoadCase import process_aeroelastic_load_case

import aerosandbox.numpy as np

desvars = {
    '(2t/H)_sq':0.2,
    '(2t/H)_pq':0.5,
    '(2t/H)_aq':0.025,
    'W_bb':0.005,
    'W_mb':0.03,
    'W_lb':0.01
}

optimiser = Optimiser(desvars, [mc.LC_INFO[3]], mc.CAD_STRS, mc.MATERIALS, mc.RES, mc.G0, mc.MTOM, mc.AIRFS, mc.LBUCKLSF,
                      mc.BOUNDS)

resPq = 10
resSq = 10
tPqs = np.linspace(.1, .3, resPq)
tSqs = np.linspace(.1, .3, resSq)
counts = np.zeros((resSq, resPq))

for i in range(resSq):
    desvars['(2t/H)_sq'] = tSqs[i]
    for j in range(resPq):
        print(f"Processing: sq: {tSqs[i]}, pq: {tPqs[j]}")
        desvars['(2t/H)_pq'] = tPqs[j]
        counts[i,j] = optimiser.simulate_constraints(optimiser.desvarvec(desvars))[3]

print(counts)
        