import componentClasses as ccl
import angledRibPattern as arp
import typing as ty
import pointsFromCAD as pfc
import meshSettings as mst
import geometricClasses as gcl
import constants as con

'''generating initial values for sizing loop based on created rib geometry and assumptions'''
def initial_components(joints:ty.List[arp.JointPoints], settings:mst.Meshsettings, stiff2Near:bool, skiDirs:ty.List[gcl.Line2D],
                       startTop:bool)->ty.Tuple[ty.List[ccl.Component], ty.List[int]]:
    '''create the spar, skins, ribs, rivets and lump masses of the wing'''
    components:ty.List[ccl.Component] = list()

    #1) diagonal ribs
    for j, jointFar in enumerate(joints[1:]):
        jointNear = joints[j]
        components.append(ccl.StiffenedRib(jointNear, jointFar, settings, stiff2Near))

    #2) skins
    components.append(ccl.Skin(components, skiDirs))

    #3) masses
    #3.0) setup
    volumes = list()
    projMids = list()
    projNears = list()
    projFars = list()
    def trig_area(ptNear, jointMid, jointFar):
        '''The part responsible for calculating relevant projected points, projected area and volume, abstracted away'''
        projFar = gcl.Point2D(jointFar.fi.y, jointFar.fi.z)
        projNear = gcl.Point2D(ptNear.y, ptNear.z)
        midpoint = jointMid.fi.pt_between(jointMid.fo, .5)
        projMid = gcl.Point2D(midpoint.y, midpoint.z)
        #triangle area
        v1 = [projFar.x-projMid.x, projFar.y-projMid.y]
        v2 = [projNear.x-projMid.x, projNear.y-projMid.y]
        area = .5*abs(v1[0]*v2[1]-v2[0]*v1[1])
        #combining with the constraining chordwise dim for volume
        volumes.append(area*(jointFar.ri.x-jointFar.fi.x))
        #saving some sstuff for later
        projMids.append(projMid)
        projNears.append(projNear)
        projFars.append(projFar)
    #3.1) detrmining the volumes of particular batteries in m^3
    #first battery - using skin dimensions to get the initial corner
    skinRefPtIndex = -1 if startTop else 0 #opposite to where the ribs start
    skinRefPt = components[-1].net[skinRefPtIndex]
    jointMid = joints[0]
    jointFar = joints[1]
    trig_area(skinRefPt, jointMid, jointFar)
    #remaining batteries
    for j, jointFar in enumerate(joints[2:]):
        #relevant joint points
        ptNear = joints[j].fo
        jointMid = joints[j+1]
        trig_area(ptNear, jointMid, jointFar)
    #3.2) converting the volumes to masses
    totVol = sum(volumes)
    masses = [volume/totVol*con.HTBM for volume in volumes]
    #3.3) creating batteries
    for j, mass in enumerate(masses):
        jointFar = joints[j+1]
        components.append(ccl.Battery(mass, jointFar.fi, jointFar.ri, projNears[j], projMids[j], projFars[j], settings))
    #TODO: add the 1st battery, and condition that excludes the engine pts and limits the lg ones
    return components, []


if __name__ == "__main__":
    _pfc = pfc.PointsFromCAD.testpoints()
    import assumptions as asu
    jointWidth = asu.jointWidth

    joints, dihedralDir, skinDirs, nearPts, farPts = arp.ray_rib_pattern(jointWidth, _pfc, True, False)
    sets = mst.Meshsettings(10, 10, 2)
    nribs = 2*len(joints)-2
    components, _ = initial_components(joints, sets, asu.stiffenerTowardsNear, skinDirs, asu.startTop)
    from mpl_toolkits import mplot3d
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    for i in range(nribs, -1, -1):
        ax.plot(*gcl.pts2coords3D(components[i].net))
    plt.show()