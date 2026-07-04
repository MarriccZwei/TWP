from ..Optimiser import Optimiser
from ..ConfigFiles import mainConfig as mc
from ..ConfigFiles import userConfig as uc
from ..Solution.processLoadCase import prep_displacements, plot_nodal_quantity
import pickle
import pyfe3d
import numpy as np

optimiser = Optimiser(mc.DESVARS_INITIAL, [mc.LC_INFO[0]], mc.GEOM_SOURCE, mc.HYPERPARAMS, mc.MASSES, mc.N, mc.MATERIALS, mc.RES, mc.G0, mc.MTOM, mc.NAIRFS, mc.LBUCKLSF,
                      mc.BOUNDS) #we don't need accurate joint mass as we are plotting an already obtained mode

scaling = 100

for i in [0, 3]:
    with open(uc.PCL_PATH+f"\\buckling_modes_lc{i}.pcl", mode="rb") as f:
        eigvecs = pickle.load(f) #selecting the non-spurious mode
        for j in [0,1]:
            eigvec = eigvecs[:, j]
            eigvec2plot = eigvec
            plot_nodal_quantity(
                ncoords=prep_displacements(eigvec2plot, optimiser.model, scaling),
                qty=eigvec2plot[2::pyfe3d.DOF],
                savePath=uc.PCL_PATH,
                plotName=f"buckl_plot{i}{j}",
                model=optimiser.model
            )
            Ue = eigvec[optimiser.model.bu].T @ optimiser.model.KC0uu @ eigvec[optimiser.model.bu]
            print(f"saved {i}-{j}.", Ue)