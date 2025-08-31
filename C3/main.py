import blocks as bl
import scipy.optimize as opt
import functools as ftl
import numpy as np
import numpy.typing as nt
import constants as cst
import elementDefs as ed
import pyfe3Dgcl as p3g
import pyfe3d as pf3
import geometricClasses as gcl
import typing as ty

@ftl.cache
def _load_c_analysis(optimVars:ty.Tuple[float]):
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
    for i, lc in enumerate(cst.LOAD_C):
        femres = bl.fem_linear_block(cst.CONSTS, meshOut, lc, True)
        try:
            ppcres = bl.post_processor_block(femres, meshOut, sizerVars, cst.CONSTS)
        except:
            print(f"Post-Proc fail at {sizerVars}, load case {i}")
            raise
        lfems.append(femres)
        lppcs.append(ppcres)
    return lfems, lppcs, meshOut

def buckling_constraint(optimVars:nt.NDArray[np.float64]):
   lfems, lppcs, meshOut = _load_c_analysis(tuple(optimVars))
   load_mult = min(lppc["lm"] for lppc in lppcs)
   return load_mult

def aeroelastic_constraint(optimVars:nt.NDArray[np.float64]):
    lfems, lppcs, meshOut = _load_c_analysis(tuple(optimVars)) 
    max_imag = max(max(abs(np.imag(wn)) for wn in lppc["wn"]) for lppc in lppcs)
    return max_imag

#TODO: We will include stress constraints when we have the stresses and the actual model

def weight_objective(optimVars:nt.NDArray[np.float64]):
    lfems, lppcs, meshOut = _load_c_analysis(tuple(optimVars))
    mass = p3g.weight(meshOut["M"], 1, meshOut["N"], pf3.DOF, gcl.Direction3D(0,0,1))
    return np.sum(mass)

if __name__ == "__main__":
    bucC = opt.NonlinearConstraint(buckling_constraint, -np.inf, 1)
    aelC = opt.NonlinearConstraint(aeroelastic_constraint, -.01, .001)
    bounds = opt.LinearConstraint(np.eye(7), [0.001, 0.07, 0.07, 0.01, 0.01, 0.01, 0.01], [.015, np.inf, np.inf, 1, 1, 1, 1])

    optimVars = np.array([cst.INITIAL["tskin"], cst.INITIAL["bsp"], cst.INITIAL["csp"], cst.INITIAL["fspar"],
                          cst.INITIAL["frib"], cst.INITIAL["fLETE"], cst.INITIAL["flg"]])

    res = opt.minimize(weight_objective, optimVars, method="COBYLA", constraints=[bucC, aelC])

    print("=============================================================================================")
    print(f"Success: {res.success}, msg: {res.message}")
    print(f"Values found: {res.x}")
    print(f"Final mass: {res.fun}")
    print("=============================================================================================")
    print("")
    