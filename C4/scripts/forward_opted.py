from ..Optimiser import Optimiser
from ..ConfigFiles import mainConfig as mc
from ..ConfigFiles import userConfig as uc
import numpy as np

desvars = {'(2t/H)_Sq': np.float64(0.15836638133674147), '(2t/H)_Pq': np.float64(0.15411829099784008), '(2t/H)_Aq': np.float64(0.14747950614755173), 'W_bb': np.float64(0.016020040088785176), 'W_mb': np.float64(0.01600280320751574), 'W_lb': np.float64(0.016118796257331848), 'ds': np.float64(0.011871654977600083), 'de': np.float64(0.011876414091993087), '(2t/H)_sq': np.float64(0.15713819684923663), '(2t/H)_pq': np.float64(0.11250508084285184), '(2t/H)_aq': np.float64(0.11126672107575558)}

optimiser = Optimiser(desvars, mc.LC_INFO, mc.GEOM_SOURCE, mc.HYPERPARAMS, mc.MASSES, mc.N, mc.MATERIALS, mc.RES, mc.G0, mc.MTOM, mc.NAIRFS, mc.LBUCKLSF,
                      mc.BOUNDS)
print(optimiser.forward(optimiser.desvarvec(), True, uc.FW_SAVE_PATH))
print(optimiser.objective(optimiser.desvarvec()))