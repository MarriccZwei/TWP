import typing as ty
import loadsInterpreter as lin

def stresses(eles:ty.List, internalLoads:ty.List[lin.EquivalentPtLoad])->ty.Tuple[ty.List[float], ty.List[float]]:
    '''getting shear and normal stresses'''
    raise NotImplementedError

def buckling(eles:ty.List, normalStresses:ty.List[float], shearstresses:ty.List[float]) -> ty.List[str]:
    '''getting what type of buckling will occur'''
    raise NotImplementedError

def naturalFreqs(solverInterior:ty.Dict)->ty.List[float]:
    '''using computed mass and stiffness matrices, computes natural frequencies'''
    raise NotImplementedError