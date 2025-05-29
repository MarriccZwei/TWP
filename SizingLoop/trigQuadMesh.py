import angledRibPattern as arp
import typing as ty
import meshSettings as mst
import initialValues as iv
import componentClasses as ccl
import geometricClasses as gcl

def mesh(joints:ty.List[arp.JointPoints])->ty.Tuple[ty.List[ty.List[gcl.Point3D]], ty.List[ccl.Component]]:
    '''creates a list of components and a list of lists of nodes belonging to the component'''
    raise NotImplementedError