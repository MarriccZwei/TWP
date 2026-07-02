from ..Optimiser import Optimiser
from ..ConfigFiles import mainConfig as mc
from ..ConfigFiles import userConfig as uc
import numpy as np

desvars = {'(2t/H)_Sq': np.float64(0.14779078527131706), '(2t/H)_Pq': np.float64(0.1941012111518347), '(2t/H)_Aq': np.float64(0.17841345464904979), 'W_bb': np.float64(0.01578562454259829), 'W_mb': np.float64(0.018572223880988704), 'W_lb': np.float64(0.01574421166844625), 'ds': np.float64(0.015763858666424515), 'de': np.float64(0.01780049033655861), '(2t/H)_sq': np.float64(0.16046288064007755), '(2t/H)_pq': np.float64(0.10110644736027476), '(2t/H)_aq': np.float64(0.12659052092093015)}

optimiser = Optimiser(desvars, mc.LC_INFO, mc.GEOM_SOURCE, mc.HYPERPARAMS, mc.MASSES, mc.RES, mc.MATERIALS, mc.RES, mc.G0, mc.MTOM, mc.NAIRFS, mc.LBUCKLSF,
                      mc.BOUNDS)
print(optimiser.forward(optimiser.desvarvec(), True, uc.FW_SAVE_PATH))
print(optimiser.objective(optimiser.desvarvec()))