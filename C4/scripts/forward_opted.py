from ..Optimiser import Optimiser
from ..ConfigFiles import mainConfig as mc
from ..ConfigFiles import userConfig as uc
import numpy as np

desvars = {'(2t/H)_Sq': np.float64(0.21326282487490014), '(2t/H)_Pq': np.float64(0.2166493466295755), '(2t/H)_Aq': np.float64(0.16522315578554822), 'W_bb': np.float64(0.016233130279402663), 'W_mb': np.float64(0.015795621594311225), 'W_lb': np.float64(0.026155316722902878), 'ds': np.float64(0.012635661725324059), 'de': np.float64(0.012541686032693776), '(2t/H)_sq': np.float64(0.15018669059398843), '(2t/H)_pq': np.float64(0.1557332235876039), '(2t/H)_aq': np.float64(0.161381268511945)}

optimiser = Optimiser(mc.DESVARS_INITIAL, mc.LC_INFO, mc.GEOM_SOURCE, mc.HYPERPARAMS, mc.MASSES, mc.N, mc.MATERIALS, mc.RES, mc.G0, mc.MTOM, mc.NAIRFS, mc.LBUCKLSF,
                      mc.BOUNDS)
print(optimiser.forward(optimiser.desvarvec(), True, uc.FW_SAVE_PATH))
print(optimiser.objective(optimiser.desvarvec()))