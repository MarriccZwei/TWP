import loadsInterpreter as lin
import geometricClasses as gcl
import typing as ty

def solve(nodes:ty.List[ty.List[gcl.Point3D]], eles:ty.List, loads:ty.List[ty.List[lin.EquivalentPtLoad]], 
          ids2track:ty.List[int])->ty.Tuple[ty.List[lin.EquivalentPtLoad], ty.List[gcl.Point3D], ty.Dict]:
    '''reformat the system and obtain internal loads at each element and displacements at each probe, also returns dicted solver matrices'''
    raise NotImplementedError