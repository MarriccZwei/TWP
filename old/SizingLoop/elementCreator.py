import componentClasses as ccl
import typing as ty
import geometricClasses as gcl

def eles(nodes:ty.List[ty.List[gcl.Point3D]], components:ty.List[ccl.Component])->ty.List:
    '''This modules creates elements attached to appropriate mesh nodes, based on co-indexed lists of components and parts of mesh corresponding to components of the same index.
    Also, creates mass elements for eles and spring elements for joints'''
    raise NotImplementedError
