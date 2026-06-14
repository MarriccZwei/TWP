from ..Optimiser import Optimiser
from ..ConfigFiles import mainConfig as mc
from ..ConfigFiles import userConfig as uc
from ..Solution.processLoadCase import prep_displacements, plot_nodal_quantity
import pickle
import pyfe3d

optimiser = Optimiser(mc.DESVARS_INITIAL, [mc.LC_INFO[0]], mc.GEOM_SOURCE, mc.HYPERPARAMS, mc.MASSES, mc.N, mc.MATERIALS, mc.RES, mc.G0, mc.MTOM, mc.NAIRFS, mc.LBUCKLSF,
                      mc.BOUNDS)

scaling = 100.
for i in [0, 3]:
    with open(uc.PCL_PATH+f"\\buckling_modes_lc{i}.pcl", mode="rb") as f:
        eigvec = pickle.load(f)
        plot_nodal_quantity(
            ncoords=prep_displacements(eigvec, optimiser.model, scaling),
            qty=eigvec[2::pyfe3d.DOF],
            savePath=uc.PCL_PATH,
            plotName=f"buckl_plot{i}",
            model=optimiser.model
        )
    print("saved")