import componentClasses as ccl
import angledRibPattern as arp
import typing as ty
import pointsFromCAD as pfc
import meshSettings as mst
import geometricClasses as gcl
import constants as con

'''generating initial values for sizing loop based on created rib geometry and assumptions'''
def initial_components(joints:ty.List[arp.JointPoints], settings:mst.Meshsettings, cadData:pfc.PointsFromCAD, stiff2Near:bool, 
                       skiDirs:ty.List[gcl.Line2D], startTop:bool,
                       weighs:ty.Dict[str, float])->ty.Tuple[ty.List[ccl.Component], ty.List[int]]:
    '''create the spar, skins, ribs, rivets and lump masses of the wing'''
    components:ty.Dict[str, ccl.Component] = dict()

    #1) diagonal ribs
    for j, jointFar in enumerate(joints[1:]):
        jointNear = joints[j]
        components[f'rib{j}'] = ccl.StiffenedRib(jointNear, jointFar, settings, stiff2Near)

    #2) skins
    components['skin'] = ccl.Skin(components.values(), skiDirs)

    #3) masses
    #3.0) setup
    volumes = list()
    projMids = list()
    projNears = list()
    projFars = list()
    xdists=list()
    jointsFarUsed = list()
    def trig_area(ptNear, jointMid, jointFar):
        '''The part responsible for calculating relevant projected points, projected area and volume, abstracted away'''
        projFar = gcl.Point2D(jointFar.fi.y, jointFar.fi.z)
        projNear = gcl.Point2D(ptNear.y, ptNear.z)
        midpoint = jointMid.fi.pt_between(jointMid.fo, .5)
        projMid = gcl.Point2D(midpoint.y, midpoint.z)
        #checking if there is no coincidence with the engine
        for i, engineStart in enumerate(cadData.engineStarts):
            engineEnd = cadData.engineEnds[i]
            if max(projNear.x, engineStart)<=min(projFar.x, engineEnd):
                return #there is no battery if there is motor coincident with the cell
        #triangle area
        v1 = [projFar.x-projMid.x, projFar.y-projMid.y]
        v2 = [projNear.x-projMid.x, projNear.y-projMid.y]
        area = .5*abs(v1[0]*v2[1]-v2[0]*v1[1])
        #chordwise distance based on landing gear location
        xdist = (cadData.lgEnd.x-jointFar.fi.x) if (max(projNear.x, cadData.lgStart.y)
                                                    <=min(projFar.x, cadData.lgEnd.y)) else (jointFar.ri.x-jointFar.fi.x)
        #combining the area with the constraining chordwise dim for volume
        volumes.append(area*(xdist))
        #saving some sstuff for later
        projMids.append(projMid)
        projNears.append(projNear)
        projFars.append(projFar)
        xdists.append(xdist)
        jointsFarUsed.append(jointFar)
    #3.1) detrmining the volumes of particular batteries in m^3
    #first battery - using skin dimensions to get the initial corner
    skinRefPtIndex = -1 if startTop else 0 #opposite to where the ribs start
    skinRefPt = components["skin"].net[skinRefPtIndex]
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
        components[f'bat{j}'] = ccl.Battery(mass, jointsFarUsed[j].fi, xdists[j], projNears[j], projMids[j], projFars[j], settings)

    #4) Other masses
    #4.1) motors
    for j, cg in enumerate(cadData.engineCgs):
        components[f'eng{j}'] = ccl.Mass(weighs['motor'], cg, [], [])
    #4.2) lg
    components['lg'] = ccl.Mass(weighs['lg'], cadData.lgCg, [], [])
    #4.3) hinge
    components['hinge'] = ccl.Mass(weighs['hinge'], cadData.tft.pt_between(cadData.trt, .5), [], [])

    return components, []


    


if __name__ == "__main__":
    _pfc = pfc.PointsFromCAD.testpoints()
    import assumptions as asu
    jointWidth = asu.jointWidth

    joints, dihedralDir, skinDirs, nearPts, farPts = arp.ray_rib_pattern(jointWidth, _pfc, True, False)
    sets = mst.Meshsettings(10, 10, 2)
    components, _ = initial_components(joints, sets, _pfc, asu.stiffenerTowardsNear, skinDirs, asu.startTop, asu.weighs)

    '''uncomment for a 3d plot'''
    # from mpl_toolkits import mplot3d
    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # compval = components.values()
    # for i in range(len(compval)-1, -1, -1):
    #     ax.plot(*gcl.pts2coords3D(list(compval)[i].net))
    # plt.show()

    '''uncomment for projected plot'''
    # import matplotlib.pyplot as plt
    # for c in components.values():
    #     x, y, z = gcl.pts2coords3D(c.net)
    #     plt.subplot(211)
    #     plt.plot(y, x) if len(y)>1 else plt.scatter(y,x)
    #     plt.title("Top View")
    #     plt.subplot(212)
    #     plt.plot(y, z) if len(y)>1 else plt.scatter(y,z)
    #     plt.title("Front View")
    # plt.show()