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
    def net(self):
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
    testSrib = StiffenedRib(testJointNear, testJointFar, sets, True)
    testSribNet = testSrib.net

    from mpl_toolkits import mplot3d
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    
    xs = [pt.x for pt in testSribNet]
    ys = [pt.y for pt in testSribNet]
    zs = [pt.z for pt in testSribNet]
    ax.plot(xs, ys, zs)
    plt.show()
    print()
    
        


