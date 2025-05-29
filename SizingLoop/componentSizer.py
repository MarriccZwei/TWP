import typing as ty
import componentClasses as ccl

class SimReport():
    def __init__(self, normalStresses:ty.List[float],  shearstresses:ty.List[float], buckling:ty.List[float], natfreq:ty.List[float]):
        self.normalStresses = normalStresses
        self.shearstresses = shearstresses
        self.buckling = buckling
        self.natfreq = natfreq

def newc(components:ty.List[ccl.Component], simReports:ty.List[SimReport])->ty.Tuple[bool, ty.List[ccl.Component]]:
    '''For every load case'''
    '''First checks if local failure modes are satisfied for each element. If not, resizes st they are and sends it back to recalculate.
    If local modes are fine, calculate global modes. The bool return allows the model to determine whether to start the loop again or not'''
    '''Or use some other re-evaluation function'''
    raise NotImplementedError