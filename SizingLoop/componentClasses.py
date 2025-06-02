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
    
    @abc.abstractmethod
    def net(self, settings:mst.Meshsettings)->ty.List[gcl.Point3D]:
        '''returns a net of nodes on an element compliant with mesh settings'''
        raise NotImplementedError

# For now, we ommit the spar, just to test everything and keep it simple   
class StiffenedRib(Component):
    '''The diagonal rib, bent at the ends, supported with stiffeners'''
    def __init__(self, jointNear:arp.JointPoints, jointFar:arp.JointPoints, jointWidth, stiffTowardsNear=True):
        #super().__init__() #for now we don't have any parent variables
        self.jointNear = jointNear
        self.jointFar = jointFar
        self.stiffTowardsNear=stiffTowardsNear

        #characteristic dimensions used for meshing
        self.lengthDiag = ((self.jointFar.fo.y-self.jointNear.fo.y)**2+(self.jointFar.fo.z-self.jointNear.fo.z)**2)**.5 #of the diagonal part
        self.widthNear = self.jointNear.ri.x-self.jointNear.fi.x
        self.widthFar = self.jointFar.ro.x-self.jointFar.fo.x
        self.widthMean = self.widthFar/2+self.widthNear/2
        self.jointWidth =jointWidth #should be same for all joints, re-compte it once in initial values

    def net(self, settings:mst.Meshsettings)->ty.List[gcl.Point3D]:
        #1) element sizes and directions
        ld = self.lengthDiag/settings.nb #element length along self.lengthDiag, etc.
        wm = self.widthMean/settings.nc
        jw = self.jointWidth/settings.nbf
        #use 3d vectors for directions

        #2) point generation
        pts = list()
        xCurrent, yCurrent, zCurrent = 0, 0, 0


