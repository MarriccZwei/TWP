import blocks as bl
import scipy.optimize as opt
import functools as ftl
import numpy as np
import numpy.typing as nt
import constants as cst
import elementDefs as ed

@ftl.cache
def _load_c_analysis(optimVars:nt.NDArray[np.float64]):
    sizerVars = {
    #skin
    'tskin':optimVars[0], #[m]
    'csp': optimVars[1], #spacing [m]
    'bsp': optimVars[2], #spacing [m]

    #truss
    'fspar':optimVars[3], #[-], spar thickness/radius fraction
    'frib':optimVars[4], #[-], rib thickness/radius fraction
    'fLETE':optimVars[5], #[-], LETE thickness/radius fraction
    'flg':optimVars[6], #[-], LG truss thickness/radius fraction
    }

    eleDict = ed.eledict(cst.CONSTS, sizerVars, cst.CODES)
    meshOut = bl.mesh_block(cst.CAD_DATA, sizerVars, eleDict, cst.CONSTS, cst.CODES)
    lfems, lppcs = list(), list()
    jmax_omegans, ldms = list(), list()
    for i, lc in enumerate(cst.LOAD_C):
        femres = bl.fem_linear_block(cst.CONSTS, meshOut, lc)
        ppcres = bl.post_processor_block(femres, meshOut, sizerVars, cst.CONSTS)
        lfems.append(femres)
        lppcs.append(ppcres)
    return lfems, lppcs

#def buckling_constraint(optimVars:nt.NDArray[np.float64]):

