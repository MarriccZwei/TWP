import numpy as np
import Analytical as an
import SizingLoop.geometricClasses as gcl

"bending stresses obtained with pfe3D to be compared with the analytical case"
class Sheet():
    def __init__(self, wfix:float, wfree:float, length:float, h:float, stiffener:an.RealStiffener, nstiff:int):
        self.wfix = wfix
        self.wfree = wfree
        self.length = length
        self.h = h #sheet thickness, abbreviated due to being used many times
        self.stiffener = stiffener
        self.nstiff = nstiff

    def default_mesh(self, nw, nl): #numbers of elements in width-wise and chord-wise dirs
        ncoordslist = list()
        for y in np.linspace(0, self.length, nl+1):
            zOverLen = y/self.length
            w = self.wfix*(1-zOverLen)+zOverLen*self.wfree #linear interpol between the edge widths
            for x in np.linspace(-w/2, w/2, nw+1):
                ncoordslist.append(np.array([x, y, 0]))
        #applying pyfe3d
        ncoords = np.array(ncoordslist)
        x = ncoords[:, 0]
        y = ncoords[:, 1]
        z = ncoords[:, 2]
        ncoords_flatten = ncoords.flatten()
        return x, y, z, ncoords, ncoords_flatten
        