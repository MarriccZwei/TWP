import angledRibPattern as arp
import typing as ty
import meshSettings as mst
import componentClasses as ccl
import initialValues as ivl
import geometricClasses as gcl

def mesh(components:ty.List[ccl.Component])->ty.Tuple[ty.List[ty.List[gcl.Point3D]], ty.List[int]]:
    '''creates lists of nodes belonging to the component and retains the ids of loads to track'''
    raise NotImplementedError