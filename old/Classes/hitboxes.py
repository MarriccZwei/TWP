from typing import Tuple, List
import unittest
import math

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

'''Coordinate geometry needed to generate batteries inside the cell'''
def point_from_point(point:Tuple[float], x, y, z):
    return (point[0]+x, point[1]+y, point[2]+z)

def vector(pt1, pt2):
    return (pt2[0]-pt1[0], pt2[1]-pt1[1], pt2[2]-pt1[2])

def cross_product(vec1, vec2):
    return (vec1[1]*vec2[2]-vec1[2]*vec2[1], vec2[0]*vec1[2]-vec1[0]*vec2[2], vec1[0]*vec2[1]-vec1[1]*vec2[0])

def plane_3_points_xz(p1:Tuple[float], p2:Tuple[float], p3:Tuple[float]): #the plane equation with x and z as variables
    vec1 = vector(p1, p2)
    vec2 = vector(p2, p3)
    cp = cross_product(vec1, vec2)
    C = -cp[0]*p1[0]-cp[1]*p1[1]-cp[2]*p1[2] #evaluate at p1 to get the equation constant
    return lambda x, z:-(cp[0]*x+cp[2]*z+C)/cp[1]

class Cell():
    """A representation of the prismatic cell being the main storage unit for the batteries"""
    def __init__(self, pts:List[Tuple[float]]):
        #points assigned by their signatures
        self.fit = pts[0]
        self.fib = pts[1]
        self.fot = pts[2]
        self.fob = pts[3]
        self.rot = pts[4]
        self.rob = pts[5]
        #generating top and bottom planes
        self.topPlane = plane_3_points_xz(self.fit, self.fot, self.rot)
        self.botPlane = plane_3_points_xz(self.fib, self.fob, self.rob)

        #generating corner points
        #z is the spanwise direction here. Thus x and y are copied from the rect corner with z of the offseted corner
        self.cft = (self.fot[0], self.fot[1], self.fit[2])
        self.cfb = (self.fob[0], self.fob[1], self.fib[2])

        # #accounting for the root cell
        # if self.atZ and topPts[2][2] > topPts[1][2]:
        #     self.topCorners[1] = (topPts[1][0], topPts[1][1], topPts[2][2]) #shrinking the root cell to a higher (lower negative) z value

    #code from stack overflow - adapted to python and used in zx plane for our top and bottom triangles
    def populate(self, battery:Battery, margin:float, hitboxes2avoid:List[Hitbox]) -> List[Battery]:
        '''1st see how much fits in the xz top triangle, 2nd for each of the battery corner xz find all applicable ys, 
        return a set of battery objects'''
        batXZs = list() #the list of battery least positive corners on the XZ plane
        bats = list() #the final list that shall be returned

        #top dimensions seem to be a bit more constraining so we will do in terms of them
        zwidth = self.cft[2]-self.fot[2]
        xwidth = self.rot[0]-self.fot[0]

        #how many batteries would fit
        nz = int((zwidth+margin)/(battery.zdim+margin))
        nx = int((xwidth+margin)/(battery.xdim+margin))

        #fitting batteries in the XZ plane
        for ix in range(nx):
            for iz in range(nz):
                #translate towards negative z and towards positive x, with closer corner in x and further in z 
                #to satisfy the batXZs sign convention
                batXZs.append(point_from_point(self.cft, (battery.xdim+margin)*ix, 0, -battery.zdim*(iz+1)-margin*iz))

        for p in batXZs:
            #obtaining the most constraining y height for the battery stack
            #evaluating for each corner and if more constraining than the previous one - replacing as the new minimum
            starty = self.topPlane(p[0], p[2])
            startHeight = abs(self.botPlane(p[0], p[2])-starty) #abs comes with some risks but since the planes never intersect...

            xcornHeight = abs(self.botPlane(p[0]+battery.xdim, p[2])-self.topPlane(p[0]+battery.xdim, p[2]))
            if xcornHeight<startHeight:
                starty = self.topPlane(p[0]+battery.xdim, p[2])
                startHeight = xcornHeight

            zcornHeight = abs(self.botPlane(p[0], p[2]+battery.zdim)-self.topPlane(p[0], p[2]+battery.zdim))
            if zcornHeight<startHeight:
                starty = self.topPlane(p[0], p[2]+battery.zdim)
                startHeight = zcornHeight

            oppCornHeight = abs(self.botPlane(p[0]+battery.xdim, p[2]+battery.zdim)-self.topPlane(p[0]+battery.xdim, p[2]+battery.zdim))
            if xcornHeight<startHeight:
                starty = self.topPlane(p[0]+battery.xdim, p[2]+battery.zdim)
                startHeight = oppCornHeight

            batEffY = margin+battery.ydim #effective dimension
            m = int((startHeight+margin)/batEffY) #counting batteries           
            #generating batteries
            for i in range(m):
                corner1 = (p[0], starty-batEffY-i*(batEffY+margin), p[2]) #y-axis direction dependent !!!
                bat = Battery(corner1, battery.xdim, battery.ydim, battery.zdim)
               
                #checking if the batteries are not inside or touching any hitbox
                append = True
                for hbx in hitboxes2avoid:
                    if not bat.outside(hbx):
                        append = False

                if append: bats.append(bat)
            
        return bats

    # def optimised_populate(self, battery:Battery, margin:float, hitboxes2avoid:List[Hitbox]) -> List[Battery]:
    #     batVol = battery.ydim*battery.ydim*battery.zdim
    #     bats = self.populate(battery, margin, hitboxes2avoid) #populate with battery in the original direction

        


'''OUTDATED'''
if __name__ == "__main__":
    class TestHitbox(unittest.TestCase):
        def test_outside(self):
            hb1 = Hitbox((9,9,7), (1, 2, 3))
            hb2 = Hitbox((2, 3, 4), (7, 6, 7))
            hb3 = Hitbox((0, 0.5, 1), (9, 9, -8))
            self.assertTrue(not hb1.outside(hb2))
            self.assertTrue(hb3.outside(hb1))

    class TestCell(unittest.TestCase):
        def test_populate(self):
            c1 = Cell([(-2, 175, 2500), (-1002, 225, 2500), (-1002, 200, 2000)], [(-1, -125, 2500), (-1005, -150, 2500), (-1000, -100, 2000)])
            batteries = c1.populate(Battery((0, 0, 0), 125, 40, 75), 20, [])
            import matplotlib.pyplot as plt
            corn1xs = [bat.corner1[0] for bat in batteries]
            corn1zs = [bat.corner1[1] for bat in batteries]
            corn2xs = [bat.corner2[0] for bat in batteries]
            corn2zs = [bat.corner2[1] for bat in batteries]
            cellTopxs = [p[0] for p in c1.topCorners]+[c1.topCorners[0][0]]
            cellTopzs = [p[1] for p in c1.topCorners]+[c1.topCorners[0][1]]
            cellBotxs = [p[0] for p in c1.botCorners]+[c1.botCorners[0][0]]
            cellBotzs = [p[1] for p in c1.botCorners]+[c1.botCorners[0][1]]
            plt.plot(corn1xs, corn1zs)
            plt.plot(corn2xs, corn2zs)
            plt.plot(cellTopxs, cellTopzs)
            plt.plot(cellBotxs, cellBotzs)
            plt.show()


    unittest.main()