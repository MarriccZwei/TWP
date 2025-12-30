from ..Pyfe3DModel import Pyfe3DModel
from ..LoadCase import LoadCase

import numpy.typing as nt
import numpy as np
import typing as ty
from pypardiso import spsolve

from . import stressRecovery as sr

def process_load_case(model:Pyfe3DModel, lc:LoadCase, materials:ty.Dict[str, float],
                      plot:bool=False, saveDir:str=None)->nt.NDArray[np.float64]:
    '''
    Docstring for process_load_case
    
    :param model: an alerady updated Pyfe3DModel
    :type model: Pyfe3DModel
    :param materials: the dictionary of material properties with keys in form "<PROPERTY>_<MATERIAL>"
    :param lc: the analysed load case, with all unchanging loads already applied
    :type lc: LoadCase
    :param plot: whether or not to plot to results
    :type plot: bool
    :param saveDir: where to save results, if set to None, but plot==True, will show the plots instead
    :type saveDir: str
    :returns: An array with maximum quad stress margin, beam stress margin, buckling load multiplier, flutter bool as float, 0.=>stable
    :type return: NDArray[float64]
    '''
    #1) weight update & static solution
    lc.update_weight(model.M)
    f = lc.loadstack()
    fu = f[model.bu]
    uu = spsolve(model.KC0uu, fu)
    u = np.zeros_like(f)
    u[model.bu] = uu

    #2) post-processing
    failure_margins = np.zeros(4)
    #KG

    Ea = materials["E_ALU"]
    Ef = materials["E_FOAM"]
    nua = materials["NU_ALU"]
    nuf = materials["NU_FOAM"]
    rhoa = materials["RHO_ALU"]
    rhof = materials["RHO_FOAM"]
    
    #2.1) quad postprocessing
    for quad, matdir, shellprop in zip(model.quads, model.matdirs, model.shellprops):
        #general updates
        quad.update_probe_xe(model.ncoords_flatten)
        quad.update_probe_ue(u)
        quad.update_KG

        normal_stresses_s, shear_stress_s, tau_yz_s, tau_xz_s = sr.recover_stresses(sr.strains_quad(model.quadprobe), 
                                                                                    Ea, nua, shellprop.scf_k13)
        normal_stresses_s, shear_stress_s, tau_yz_s, tau_xz_s = sr.recover_stresses(sr.strains_quad(model.quadprobe), 
                                                                                    Ea, nua, shellprop.scf_k13)
    
