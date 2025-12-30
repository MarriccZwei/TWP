from ..Pyfe3DModel import Pyfe3DModel
from ..LoadCase import LoadCase
from .eleProps import quad_stress_recovery

import numpy.typing as nt
import numpy as np
import typing as ty
from pypardiso import spsolve
import pyfe3d as pf3
import scipy.sparse as ss
import scipy.sparse.linalg as ssl

from . import stressRecovery as sr

def process_load_case(model:Pyfe3DModel, lc:LoadCase, materials:ty.Dict[str, float], desvars:ty.Dict[str, float],
                      beamTypes:ty.List[str], quadTypes:ty.List[str], plot:bool=False, saveDir:str=None)->nt.NDArray[np.float64]:
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
    quad_failure_margins = list()
    KGr = np.zeros(model.sizeKG, dtype=pf3.INT)
    KGc = np.zeros(model.sizeKG, dtype=pf3.INT)
    KGv = np.zeros(model.sizeKG, dtype=pf3.DOUBLE)

    Ea = materials["E_ALU"]
    Ef = materials["E_FOAM"]
    nua = materials["NU_ALU"]
    nuf = materials["NU_FOAM"]
    rhoa = materials["RHO_ALU"]
    rhof = materials["RHO_FOAM"]
    
    #2.1) quad postprocessing
    for quad, matdir, shellprop, quadType in zip(model.quads, model.matdirs, model.shellprops, quadTypes):
        quad.update_probe_xe(model.ncoords_flatten)
        quad.update_probe_ue(u)
        quad.update_KG(KGr, KGc, KGv, shellprop)
        quad_failure_margins.append(quad_stress_recovery(desvars, materials, quad, shellprop, matdir, quadType))

    #2.3) buckling
    KG = ss.coo_matrix((KGv, (KGr, KGc)), shape=(model.N, model.N)).tocsc()
    KGuu = model.uu_matrix(KG)
    num_eig_lb = 4 #as per the example in documentation, we don't need excess modes
    eigvecs = np.zeros((model.N, num_eig_lb))
    PREC = np.max(1/model.KC0uu.diagonal())
    eigvals, eigvecsu = ssl.eigsh(A=PREC*KGuu, k=num_eig_lb, which='SM', M=PREC*model.KC0uu, sigma=1., mode='cayley')
    eigvals = -1./eigvals
    eigvecs[model.bu] = eigvecsu
    load_mult = eigvals[0]

    return np.array([
        0.,
        0.,
        load_mult,
        0.
    ])
