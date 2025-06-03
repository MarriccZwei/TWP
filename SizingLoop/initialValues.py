import componentClasses as ccl
import angledRibPattern as arp
import typing as ty
import pointsFromCAD as pfc
import meshSettings as mst
import geometricClasses as gcl

'''generating initial values for sizing loop based on created rib geometry and assumptions'''
def initial_components(joints:ty.List[arp.JointPoints], cadData:pfc.PointsFromCAD, settings:mst.Meshsettings)->ty.Tuple[ty.List[ccl.Component], ty.List[ty.List[gcl.Point3D]], ty.List[int]]:
    '''create the spar, skins, ribs, rivets and lump masses of the wing'''
    components = list()
    
    #1) wing skins

    #2) diagonal ribs
    for j, jointFar in enumerate(joints[1:]):
        jointNear = joints[j]


    raise NotImplementedError