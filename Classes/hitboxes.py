from typing import Tuple, List
import unittest
import numpy as np

class Hitbox():
    '''A rectangular Hitbox'''
    def __init__(self, corner1:Tuple[float], corner2:Tuple[float]):
        self.corner1 = corner1
        self.corner2 = corner2

    #checks if this hitbox is outside another hitbox
    def outside(self, refHitbox)->bool:
        refc1 = refHitbox.corner1
        refc2 = refHitbox.corner2

        #if any of the coords does not match, then the hitboxes are completely seperate
        for i in range(3):
            xmin = min(self.corner1[i], self.corner2[i])
            xmax = max(self.corner1[i], self.corner2[i])
            refxmin = min(refHitbox.corner1[i], refHitbox.corner2[i])
            refxmax = max(refHitbox.corner1[i], refHitbox.corner2[i])
            if (refxmin > xmax) or (refxmax < xmin):
                return True #if one of the coords is separate
        return False #if all coordinates are nested then the hitboxes ain't separate

class Battery(Hitbox):
    '''Battery representation'''
    #x, y, z dim describe the setting of the battery and include the vent pipe
    def __init__(self, corner1, xdim, ydim, zdim):
        corner2 = (corner1[0]+xdim, corner1[1]+ydim, corner1[2]+zdim)
        super().__init__(corner1, corner2)
        self.xdim = xdim
        self.ydim = ydim
        self.zdim = zdim

def uncodePoints(ptStr:str, masterSep="|", coordSep=";"):
    '''Uncodes the points from CATIA and returns them as a list of tuples'''

    codedPts = ptStr.split(masterSep)
    uncodedPts = list()
    for pt in codedPts:
        ptCoords = pt.split(coordSep) #obtaining the coords
        uncodedPts.append((float(ptCoords[0]), float(ptCoords[1]), float(ptCoords[2]))) #will throw error if too little data
    return uncodedPts

def point_from_point(point:Tuple[float], x, y, z):
    return (point[0]+x, point[1]+y, point[2]+z)

class Cell():
    """A representation of the prismatic cell being the main storage unit for the batteries"""
    def __init__(self, botPTs:List[Tuple[float]], topPts:List[Tuple[float]], resolution=10):
        self.botCorners = botPTs
        self.topCorners = topPts
        #checking if the cell has a sharp corner at the z axis or elsewhere - checking which type of cell we deal with 
        self.atZ = topPts[1][0] < resolution and topPts[1][0] > resolution
        #generating the line between the 2 side corners x = a*z + b
        topLineA = (topPts[0][0]-topPts[2][0])/(topPts[0][2]-topPts[2][2])
        topLineB = topPts[0][0] - topLineA*topPts[0][2]
        self.midline = lambda z: topLineA*z+topLineB

    #code from stack overflow - adapted to python and used in zx plane for our top and bottom triangles
    def populate(self, battery:Battery, margin:float, hitboxes2avoid:List[Hitbox]):
        '''1st see how much fits in the xz top triangle, 2nd for each of the battery corner xz find all applicable ys, 
        return a set of battery objects'''
        batXZs = list()

        #first cell type - unshared corner around z = 0
        if self.atZ:
            layerPt = point_from_point(self.topCorners[1], 0, 0, margin) #the point marking a z-layer of bats
            maxLayer = self.topCorners[0][2] - margin+battery.zdim #the maximum z a layer can have
            while layerPt[2] < maxLayer:
                zwidth = margin+battery.zdim
                #generating a layer pt rectngle
                layerEndPt = point_from_point(layerPt, 0, 0, zwidth)
                #here the y coord doesn't matter
                xwidth = layerPt[0]-self.midline(layerEndPt[2])
                #how many batteries ar in a layer in the x direction
                batEffXdim = battery.xdim+margin #effective x dimension of the batteries
                n = int((xwidth-margin)/(batEffXdim))
                for i in range(1, n+1):
                    batXZs.append(point_from_point(layerPt, -i*batEffXdim, 0, 0))
                #to generate the next layer, we move on to the next layer point
                layerPt = layerEndPt



            
            
        


if __name__ == "__main__":
    class TestHitbox(unittest.TestCase):
        def test_outside(self):
            hb1 = Hitbox((9,9,7), (1, 2, 3))
            hb2 = Hitbox((2, 3, 4), (7, 6, 7))
            hb3 = Hitbox((0, 0.5, 1), (9, 9, -8))
            self.assertTrue(not hb1.outside(hb2))
            self.assertTrue(hb3.outside(hb1))

    unittest.main()