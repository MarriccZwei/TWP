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
    '(2t/H)_Sq':0.249,
    '(2t/H)_Pq':0.263,
    '(2t/H)_Aq':0.096,
    'W_bb':0.016,
    'W_mb':0.015,
    'W_lb':0.030,
    'ds':0.016,
    'de':0.013,
    '(2t/H)_sq':0.168,
    '(2t/H)_pq':0.146,
    '(2t/H)_aq':0.077
}

optimiser = Optimiser(desvars_initial, mc.LC_INFO, mc.GEOM_SOURCE, hypers, mc.MASSES, mc.N, mc.MATERIALS, res, mc.G0, mc.MTOM, mc.NAIRFS, mc.LBUCKLSF,
                      mc.BOUNDS)
print(optimiser.forward(optimiser.desvarvec(), True, uc.FW_SAVE_PATH))
print(optimiser.objective(optimiser.desvarvec()))