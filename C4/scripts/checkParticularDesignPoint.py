from ..Optimiser import Optimiser
from ..ConfigFiles import mainConfig as mc
from ..ConfigFiles import userConfig as uc

import copy

# rail_d = 0.03
hypers = mc.HYPERPARAMS.copy()
# hypers["d"] = rail_d

res = mc.RES.copy()
# res["klb"] = 0

desvars_initial = {
    '(2t/H)_Sq':0.2031,
    '(2t/H)_Pq':0.2112,
    '(2t/H)_Aq':0.1896,
    'W_bb':0.0173,
    'W_mb':0.0178,
    'W_lb':0.030,
    'ds':0.0129,
    'de':0.0125,
    '(2t/H)_sq':0.1727,
    '(2t/H)_pq':0.1697,
    '(2t/H)_aq':0.1643
}

optimiser = Optimiser(desvars_initial, [mc.LC_INFO[3]], mc.GEOM_SOURCE, hypers, mc.MASSES, mc.N, mc.MATERIALS, res, mc.G0, mc.MTOM, mc.NAIRFS, mc.LBUCKLSF,
                      mc.BOUNDS)
print(optimiser.forward(optimiser.desvarvec(), True, uc.FW_SAVE_PATH))
print(optimiser.objective(optimiser.desvarvec()))