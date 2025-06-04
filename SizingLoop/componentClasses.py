import abc
import angledRibPattern as arp
import geometricClasses as gcl
import typing as ty
import meshSettings as mst
import numpy as np

class Component(abc.ABC):
    '''In the init, it should have arguments allowing it to obtain global properties of the component,
    such us inertia params per unit length/width in a given direction'''

    @abc.abstractmethod
    def shard(self, nodes:ty.List[gcl.Point3D]):
        '''returns an element connected to the specfied nodes, with properties corresponding of the global elements of the component'''
        raise NotImplementedError
    
    @property
    @abc.abstractmethod
    def net(self)->ty.List[gcl.Point3D]:
        '''returns the mesh on the element'''
        raise NotImplementedError


# For now, we ommit the spar, just to test everything and keep it simple   
class StiffenedRib(Component):
    '''The diagonal rib, bent at the ends, supported with stiffeners'''
    def __init__(self, jointNear:arp.JointPoints, jointFar:arp.JointPoints, settings:mst.Meshsettings, stiffTowardsNear=True):
        #super().__init__() #for now we don't have any parent variables
        #adjusting joints for the connection in the middle
        self.jointNear = arp.JointPoints(jointNear.fi.pt_between(jointNear.fo, .5),
                                         jointNear.ri.pt_between(jointNear.ro, .5),
                                         jointNear.fo, jointNear.ro)
        self.jointFar = arp.JointPoints(jointFar.fi, jointFar.ri,
                                        jointFar.fo.pt_between(jointFar.fi, .5),
                                        jointFar.ro.pt_between(jointFar.ri, .5))
        self.stiffTowardsNear=stiffTowardsNear #direction of stiffeners - towards near direction of far direction
        self.__net =self. _net(settings)

    @property
    def net(self):
        return self.__net

    def _net(self, settings:mst.Meshsettings)->ty.List[gcl.Point3D]:
        nearm = gcl.quad_mesh(self.jointNear.fi, self.jointNear.fo, self.jointNear.ro, self.jointNear.ri,
                              settings.nbf, settings.nc)
        dm = gcl.quad_mesh(self.jointNear.fo, self.jointFar.fi, self.jointFar.ri, self.jointNear.ro,
                           settings.nb, settings.nc, True, False, True, False)
        farm = gcl.quad_mesh(self.jointFar.fi, self.jointFar.fo, self.jointFar.ro, self.jointFar.ri,
                              settings.nbf, settings.nc)
        return  nearm+dm+farm
    
    def shard(self):
        raise NotImplementedError
    
    def meshForSkin(self)->ty.List[gcl.Point3D]:
        return self.net
    

class Skin(Component):
    '''One class for both the lower and upper skin'''
    def __init__(self, ribsMainMesh:ty.List[StiffenedRib], skinDirs:ty.List[gcl.Line2D]):
        #super().__init__() #for now no parent attributes
        #1) obtaining the mesh
        self.topNet = list()
        self.botNet = list()
        for rib in ribsMainMesh:
            for pt in rib.meshForSkin():
                self.topNet.append(gcl.Point3D(pt.x, pt.y, Skin._getz(skinDirs[0], pt.y)))
                self.botNet.append(gcl.Point3D(pt.x, pt.y, Skin._getz(skinDirs[1], pt.y)))

    @property
    def net(self)->ty.List[gcl.Point3D]:
        return self.topNet+self.botNet[::-1]
    
    def shard(self, nodes):
        return super().shard(nodes)


    @staticmethod
    def _getz(line:gcl.Line2D, spanwisePos:float):
        '''getting the z of the top or bottom skin at a given spanwise y'''
        return gcl.Line2D(gcl.Point2D(spanwisePos, 0), gcl.Direction2D(0, 1)).intersect(line).y
    
class Battery(Component):
    def __init__(self, mass, farFront:gcl.Point3D, xdist:float, projNear:gcl.Point2D, projMid:gcl.Point2D, projFar:gcl.Point2D, settings:mst.Meshsettings):
        # super().__init__()
        '''Locates the mass along the centerline of the triangular cross section'''
        centroid = gcl.Point2D.midpoint([projMid, projNear, projFar])
        edgeFront = gcl.Point3D(farFront.x, centroid.x, centroid.y)
        edgeRear = gcl.Point3D(farFront.x+xdist, centroid.x, centroid.y) #to account for that sometimes u have lg
        self.__net = edgeFront.pts_between(edgeRear, settings.nc+1)
        self.massPer = mass/len(self.__net)
        self.settings = settings

    @property
    def net(self)->ty.List[gcl.Point3D]:
        return self.__net
    
    def shard(self):
        raise NotImplementedError
    
class Mass(Component):
    def __init__(self, mass:float, location:gcl.Point3D, connectedTo:ty.List[str], connectedAt:ty.List[ty.List[int]]):
        self.mass = mass
        self.__net = [location]
        self.connectedTo = connectedTo #id of what you connect to
        self.connectedAt = connectedAt #node indics of what you connect to

    @property
    def net(self)->ty.List[gcl.Point3D]:
        return self.__net
    
    def shard(self):
        raise NotImplementedError


if __name__ == "__main__":
    sets = mst.Meshsettings(9, 9, 2)
    jointWidthTest = 0.04
    bot_angle = np.radians(8)
    top_angle = np.radians(2)
    testJointNear = arp.JointPoints(gcl.Point3D(0, 0, 0), gcl.Point3D(2, 0, 0), 
                                    gcl.Point3D(0.02, jointWidthTest*np.cos(bot_angle), jointWidthTest*np.sin(bot_angle)),
                                    gcl.Point3D(1.956, jointWidthTest*np.cos(bot_angle), jointWidthTest*np.sin(top_angle)))
    testJointFar = arp.JointPoints(gcl.Point3D(0.3, 4, 1), gcl.Point3D(1.6, 4, 1), 
                                    gcl.Point3D(0.34, 4+jointWidthTest*np.cos(top_angle), 1+jointWidthTest*np.sin(top_angle)),
                                    gcl.Point3D(1.57, 4+jointWidthTest*np.cos(top_angle), 1+jointWidthTest*np.sin(top_angle)))
    
    #rib creation
    testSrib = StiffenedRib(testJointNear, testJointFar, sets, True)
    testSribNet = testSrib.net

    #skin creation
    skin = Skin([testSrib], [gcl.Line2D(gcl.Point2D(4, 1), gcl.Direction2D.by_angle(top_angle)), gcl.Line2D(gcl.Point2D(0, 0), gcl.Direction2D.by_angle(bot_angle))])

    from mpl_toolkits import mplot3d
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    
    xs = [pt.x for pt in testSribNet]
    ys = [pt.y for pt in testSribNet]
    zs = [pt.z for pt in testSribNet]
    ax.plot(xs, ys, zs)
    ax.plot(*gcl.pts2coords3D(skin.net))
    plt.show()
    print()
    
        


