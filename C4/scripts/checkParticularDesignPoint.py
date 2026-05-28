from ..Optimiser import Optimiser
from ..ConfigFiles import mainConfig as mc
from ..ConfigFiles import userConfig as uc

import copy

rail_d = 0.01
hypers = mc.HYPERPARAMS.copy()
hypers["d"] = rail_d

res = mc.RES.copy()
res["klb"] = 0

desvars_initial = {
    '(2t/H)_Sq':0.22530687313282435,
    '(2t/H)_Pq':0.1935629980547246,
    '(2t/H)_Aq':0.1672213443406646,
    'W_bb':0.01565030262301346,
    'W_mb':0.015801529483939807,
    'W_lb':0.026136016966417944,
    'ds':0.011931627269955957,
    'de':0.011719753338197511,
    '(2t/H)_sq':0.1604809890242429,
    '(2t/H)_pq':0.16069977670949012,
    '(2t/H)_aq':0.1601870632712459
}

optimiser = Optimiser(desvars_initial, mc.LC_INFO, mc.GEOM_SOURCE, hypers, mc.MASSES, mc.N, mc.MATERIALS, res, mc.G0, mc.MTOM, mc.NAIRFS, mc.LBUCKLSF,
                      mc.BOUNDS)
print(optimiser.simulate_constraints(optimiser.desvarvec(), True, uc.FW_SAVE_PATH))
print(optimiser.objective(optimiser.desvarvec()))