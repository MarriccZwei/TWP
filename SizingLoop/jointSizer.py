import typing as ty
import loadsInterpreter as lin

def rivets(eles:ty.List, internalLoads:ty.List[lin.EquivalentPtLoad])->ty.List[int]:
    '''filters out the spring elements and then returns the rivet counts based on loads acting on the spring elements'''
    raise NotImplementedError