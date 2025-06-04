import numpy as np
import typing as ty

class Point2D():
    def __init__(self, x:float, y:float):
        self.x = x
        self.y = y

    @classmethod
    def midpoint(cls, pts):
        return cls(sum([pt.x for pt in pts])/len(pts), sum([pt.y for pt in pts])/len(pts))
    
    def pythagoras(self, pt):
        return((self.x-pt.x)**2+(self.y-pt.y)**2)**.5

class Direction2D():
    def __init__(self, dx:float, dy:float):
        magn = (dx**2+dy**2)**.5
        self.x = dx/magn
        self.y = dy/magn

    def step(self, pt:Point2D, dist:float): #moving a point a certain distance along the direction
        return Point2D(pt.x+self.x*dist, pt.y+self.y*dist)
    
    @classmethod
    def by_angle(cls, angle:float, radians=True):
        if not radians:
            angle *= np.pi/180
        return Direction2D(np.cos(angle), np.sin(angle))
    
    @classmethod #used to obtain directions between adjacent points
    def from_pts(cls, p1:Point2D, p2:Point2D):
        delta_y = p2.y-p1.y
        delta_x = p2.x-p1.x
        return cls(delta_x, delta_y)
    
    @classmethod
    def rotate(cls, dir, angle):
        sine = np.sin(angle)
        cosine = np.cos(angle)
        rotation_matrix = np.asmatrix([[cosine, -sine], [sine, cosine]])
        #to unpack we need to convert it back to an array
        rotated = np.array(np.matmul(rotation_matrix, np.asmatrix([[dir.x], [dir.y]])))
        return cls(rotated[0][0], rotated[1][0])


class Line2D():
    def __init__(self, pt:Point2D, dir:Direction2D):
        self.pt = pt
        self.dir = dir

    @classmethod
    def from_pts(cls, pt1:Point2D, pt2:Point2D):
        return cls(pt1, Direction2D.from_pts(pt1, pt2))
    
    def intersect(self, line):
        m = (-line.dir.x*(line.pt.y-self.pt.y)+line.dir.y*(line.pt.x-self.pt.x))/(-self.dir.y*line.dir.x+self.dir.x*line.dir.y)
        return self.dir.step(self.pt, m)
    

class Point3D():
    def __init__(self, x:float, y:float, z:float):
        self.x = x
        self.y = y
        self.z = z

    def pythagoras(self, pt):
        return((self.x-pt.x)**2+(self.y-pt.y)**2+(self.z-pt.z)**2)**.5
    
    def pt_between(self, pt, ratio=.5):
        sr = 1-ratio
        return Point3D(sr*self.x+ratio*pt.x, sr*self.y+ratio*pt.y, sr*self.z+ratio*pt.z)
    
    def pts_between(self, pt, n):
        ratios = np.linspace(0, 1, n)
        return [self.pt_between(pt, ratio) for ratio in ratios]

class Direction3D():
    def __init__(self, dx:float, dy:float, dz:float):
        magn = (dx**2+dy**2+dz**2)**.5
        self.x = dx/magn
        self.y = dy/magn
        self.z = dz/magn

    def step(self, pt:Point3D, dist:float): #moving a point a certain distance along the direction
        return Point3D(pt.x+self.x*dist, pt.y+self.y*dist, pt.z+self.z*dist)
    
    @classmethod #used to obtain directions between adjacent points
    def from_pts(cls, p1:Point3D, p2:Point3D):
        delta_y = p2.y-p1.y
        delta_x = p2.x-p1.x
        delta_z = p2.z-p1.z
        return cls(delta_x, delta_y, delta_z)

'''Utility functions with geometric classes'''
def quad_mesh(pt1:Point3D, pt2:Point3D, pt3:Point3D, pt4:Point3D,
              ne12:int, ne14:int, end12=True, end23=True, end34=True, end41=True):
    '''PLZ INPUT PTS ALONG THE PERIMETER! USE CONVEX QUADRANGLES!'''
    '''creates a grid of nodes between the 4 points'''
    #0)number of elements -> number of nodes
    n12 = ne12+1
    n14 = ne14+1

    #1) point creation
    points12 = pt1.pts_between(pt2, n12)
    points43 = pt4.pts_between(pt3, n12) #to make it in same direction
    pts = list()
    #including 23 and 41 boundaries
    looprange = range(n12)
    if not end23:
        looprange = looprange[:-1]
    if not end41:
        looprange = looprange[1:]
    for i in looprange:
        newPts = points12[i].pts_between(points43[i], n14)
        #adjusting the ends
        if not end12:
            newPts = newPts[1:]
        if not end34:
            newPts = newPts[:-1]
        pts += newPts
    
    return pts

def pts2coords2D(pts:ty.List[Point2D]):
    return [pt.x for pt in pts], [pt.y for pt in pts]

def pts2coords3D(pts:ty.List[Point3D]):
    return [pt.x for pt in pts], [pt.y for pt in pts], [pt.z for pt in pts]

if __name__ == "__main__":
    pt1 = Point3D(0, 0, 0)
    pt2 = Point3D(1, -1, -1)
    pt3 = Point3D(-4, -6, -6)
    pt4 = Point3D(-5, -4, -5)
    mesh = quad_mesh(pt1, pt2, pt3, pt4, 5, 7)
    meshne = quad_mesh(pt1, pt2, pt3, pt4, 5, 7, False, False, False, False)
    from mpl_toolkits import mplot3d
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    plt.plot(*pts2coords3D(mesh))
    plt.plot(*pts2coords3D(meshne))
    plt.show()
