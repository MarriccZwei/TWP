import componentClasses as ccl
import angledRibPattern as arp
import typing as ty
import pointsFromCAD as pfc
import meshSettings as mst
import geometricClasses as gcl

'''generating initial values for sizing loop based on created rib geometry and assumptions'''
def initial_components(joints:ty.List[arp.JointPoints], cadData:pfc.PointsFromCAD, settings:mst.Meshsettings, stiff2Near:bool, startTop:bool)->ty.Tuple[ty.List[ccl.Component], ty.List[int]]:
    '''create the spar, skins, ribs, rivets and lump masses of the wing'''
    components:ty.List[ccl.Component] = list()

    #1) diagonal ribs
    for j, jointFar in enumerate(joints[1:]):
        jointNear = joints[j]
        components.append(ccl.StiffenedRib(jointNear, jointFar, settings, stiff2Near))

    #2) skins

    return components, []


if __name__ == "__main__":
    _pfc = pfc.PointsFromCAD.testpoints()
    import assumptions as asu
    jointWidth = asu.jointWidth

    joints, dihedralDir, _, nearPts, farPts = arp.ray_rib_pattern(jointWidth, _pfc, True, False)
    sets = mst.Meshsettings(10, 10, 2)
    nribs = len(joints)-1
    components, _ = initial_components(joints, _pfc, sets, asu.stiffenerTowardsNear, asu.startTop)
    from mpl_toolkits import mplot3d
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    for i in range(nribs):
        ax.plot(*gcl.pts2coords3D(components[i].net))
    plt.show()