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
    def __init__(self, jointNear:arp.JointPoints, jointFar:arp.JointPoints, stiffTowardsNear=True):
        #super().__init__() #for now we don't have any parent variables
        self.jointNear = jointNear
        self.jointFar = jointFar
        self.stiffTowardsNear=stiffTowardsNear #direction of stiffeners - towards near direction of far direction

        #characteristic dimensions used for meshing
        self.lengthDiag = self.jointNear.fo.pythagoras(self.jointFar.fi) #of the diagonal part
        self.widthNear = self.jointNear.ri.x-self.jointNear.fi.x #innermost width
        self.widthFar = self.jointFar.ro.x-self.jointFar.fo.x #outerrmost width
        self.widthND = self.jointNear.ro.x-self.jointNear.fo.x #length of the line on the intersect of near joint and diagonal, etc.
        self.widthFD = self.jointFar.ri.x-self.jointFar.fi.x
        self.widthMean = self.widthND/2+self.widthFD/2
        self.jointWN =self.jointNear.fi.pythagoras(self.jointNear.fo)
        self.jointWF =self.jointFar.fi.pythagoras(self.jointFar.fo) 

    def net(self, settings:mst.Meshsettings)->ty.List[gcl.Point3D]:
        #1) element sizes and directions
        ld = self.lengthDiag/settings.nb #element length along self.lengthDiag, etc.
        jwn = self.jointWN/settings.nbf
        jwf = self.jointWF/settings.nbf
        #use 3d vectors for directions
        vld = gcl.Direction3D.from_pts(self.jointNear.fo, self.jointFar.fi) #vector along self.lengthDiag, etc.
        #vld = gcl.Direction3D(0, vld.y, vld.z) #fixing the x dimension (assuming unswept wing)
        vwm = gcl.Direction3D(1, 0, 0) #chordwise vector
        vbn = gcl.Direction3D.from_pts(self.jointNear.fi, self.jointNear.fo) #near joint spanwise dir
        #vbn = gcl.Direction3D(0, vbn.y, vbn.z)
        vbf = gcl.Direction3D.from_pts(self.jointFar.fi, self.jointFar.fo) #far joint spanwise dir
        #vbf = gcl.Direction3D(0, vbf.y, vbf.z)

        #2) point generation 
        #2.1) tool
        def ptgen(initialPoint, nb, vc, vb, wb, widthInitial, widthFinal):
            newPts = list()
            for i in range(nb):
                spanwiseTransl = vb.step(initialPoint, i*wb)
                wc = (widthInitial + i/nb*(widthFinal-widthInitial))/settings.nc
                for j in range(settings.nc+1):
                    newPts.append(vc.step(spanwiseTransl, j*wc))
            return newPts
        #2.2) assembling the point list
        pts = ptgen(self.jointNear.fi, settings.nbf, vwm, vbn, jwn, self.widthNear, self.widthND)
        pts += ptgen(self.jointNear.fo, settings.nb, vwm, vld, ld, self.widthND, self.widthFD)
        #the +1 is to generate also the edge points
        pts += ptgen(self.jointFar.fi, settings.nbf+1, vwm, vbf, jwf, self.widthFD, self.widthFar)

        return pts
    
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
    testSrib = StiffenedRib(testJointNear, testJointFar, True)
    testSribNet = testSrib.net(sets)

    from mpl_toolkits import mplot3d
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    
    xs = [pt.x for pt in testSribNet]
    ys = [pt.y for pt in testSribNet]
    zs = [pt.z for pt in testSribNet]
    ax.scatter(xs, ys, zs)
    plt.show()
    print()
    
        


