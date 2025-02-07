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


def uncodePoints(ptStr:str, masterSep="|", coordSep=";"):
    '''Uncodes the points from CATIA and returns them as a list of tuples'''

    codedPts = ptStr.split(masterSep)
    uncodedPts = list()
    for pt in codedPts:
        ptCoords = pt.split(coordSep) #obtaining the coords
        uncodedPts.append((float(ptCoords[0]), float(ptCoords[1]), float(ptCoords[2]))) #will throw error if too little data
    return uncodedPts


class Cell():
    """A representation of the prismatic cell being the main storage unit for the batteries"""
    def __init__(self, botPTs:List[Tuple[float]], topPts:List[Tuple[float]]):
        self.botCorners = botPTs
        self.topCorners = topPts

    #code from stack overflow - adapted to python and used in zx plane for our top and bottom triangles
    def _halfplaneSign(self, pt1:Tuple[float], pt2:Tuple[float], pt3:Tuple[float]):
        return (pt1[0]-pt3[0])    
            
            
        


if __name__ == "__main__":
    class TestHitbox(unittest.TestCase):
        def test_outside(self):
            hb1 = Hitbox((9,9,7), (1, 2, 3))
            hb2 = Hitbox((2, 3, 4), (7, 6, 7))
            hb3 = Hitbox((0, 0.5, 1), (9, 9, -8))
            self.assertTrue(not hb1.outside(hb2))
            self.assertTrue(hb3.outside(hb1))

    unittest.main()