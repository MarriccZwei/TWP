import abc
import numpy as np
"Analytical formula for a 1-side fixed sheet response to pure bending"
'''ASSUMPTIONS AS IN BOOK OF HPB SET 7'''

class RealStiffener(abc.ABC):
    xst = NotImplemented #position of the cg wrt outer edge of the stiffener along x
    #outer edge is the edge that's on the boundary of the sheet - i.e the vertical for an L stiffener
    hst = NotImplemented #position of the cg wrt lower edge of the stiffener along x
    Ast = NotImplemented #cross-section area
    Ixx = NotImplemented
    Iyy = NotImplemented
    Ixy = NotImplemented


class RealLStiffener(RealStiffener):
    def __init__(self, L, t):
        self.L = L
        self.t = t
        self.xst = L/4
        self.hst = L/4
        self.Ast = 2*L*t
        self.Ixx = 5*t*L**3/24
        self.Iyy = self.Ixx
        self.Ixy = t*L**3/8


class RealSheet():
    def __init__(self, wfix:float, wfree:float, length:float, h:float, stiffener:RealStiffener, nstiff:int):
        self.wfix = wfix
        self.wfree = wfree
        self.length = length
        self.h = h #sheet thickness, abbreviated due to being used many times
        self.stiffener = stiffener
        self.nstiff = nstiff

    def sheet_properties(self, zOverLen:float):
        w = self.wfix*(1-zOverLen)+zOverLen*self.wfree #linear interpol between the edge widths
        ycentr = ((self.h**2*w/2+(self.h+self.stiffener.hst)*self.nstiff*self.stiffener.Ast)/
                  (w*self.h+self.nstiff*self.stiffener.Ast))
        xcentr = 0 #should be a relatively accurate assumtions although one of the stiffs is inverted
        Ixx = (self.nstiff*(self.stiffener.Ixx+self.stiffener.Ast*(self.h+self.stiffener.hst-ycentr)**2)
             +w*self.h*(self.h*(self.h/2-ycentr)**2))
        xcentrStiffs = np.linspace(-w/2+self.stiffener.xst, w/2-self.stiffener.xst, self.nstiff)
        IyyPS = np.sum(self.stiffener.Ast*(xcentrStiffs-xcentr)**2) #parallel axis due to stiffeners
        Iyy = self.nstiff*self.stiffener.Iyy+IyyPS+w**3*self.h/12
        #we assume Ixy to be 0 due to the sheet being basically symmetric
        return w, xcentr, ycentr, Ixx, Iyy
    
    def property_arrays(self, zOverLens):
        prop_list = [self.sheet_properties(zOverLen) for zOverLen in zOverLens]
        w = np.array([prop[0] for prop in prop_list])
        xcentr = np.array([prop[1] for prop in prop_list])
        ycentr = np.array([prop[2] for prop in prop_list])
        Ixx = np.array([prop[3] for prop in prop_list])
        Iyy = np.array([prop[4] for prop in prop_list])
        return w, xcentr, ycentr, Ixx, Iyy

        
    

if __name__=="__main__":
    length = 10
    stiff = RealLStiffener(0.05, 0.005)
    sheet = RealSheet(1, .7, length, 0.01, stiff, 5)
    zOverLens = np.linspace(0, 1, 1000)
    zs = length*zOverLens
    w, xcentr, ycentr, Ixx, Iyy = sheet.property_arrays(zOverLens)

    import matplotlib.pyplot as plt
    plt.subplot(2,2,1)
    plt.plot(zs, w)
    plt.title("w")
    plt.subplot(2,2,2)
    plt.plot(zs, ycentr)
    plt.title("ycentr")
    plt.subplot(2,2,3)
    plt.plot(zs, Ixx)
    plt.title("Ixx")
    plt.subplot(2,2,4)
    plt.plot(zs, Iyy)
    plt.title("Iyy")
    plt.show()

    "Iyy is parabolic - indeed just an object getting squeezed so it's moment of inerti will decrease linearly"
    "Ixx decreases rapidly as for less w the centroid location is closer to stiffeners hence worse"
    "ycentr increases - same amount of stiffener area, less sheet area"