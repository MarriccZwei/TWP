import numpy as np

class Point2D():
    def __init__(self, x:float, y:float):
        self.x = x
        self.y = y

    @classmethod
    def midpoint(cls, pts):
        return cls(sum([pt.x for pt in pts])/len(pts), sum([pt.y for pt in pts])/len(pts))

class Direction2D():
    def __init__(self, dx:float, dy:float):
        magn = (dx**2+dy**2)**.5
        self.ux = dx/magn
        self.uy = dy/magn

    def step(self, pt:Point2D, dist:float): #moving a point a certain distance along the direction
        return Point2D(pt.y+self.y*dist, pt.z+self.z*dist, pt.top)
    
    @classmethod #used to obtain directions between adjacent points
    def from_pts(cls, p1:Point2D, p2:Point2D):
        delta_y = p2.y-p1.y
        delta_x = p2.x-p1.x
        return cls(delta_y, delta_x)
    
    @classmethod
    def rotate(cls, dir, angle):
        sine = np.sin(angle)
        cosine = np.cos(angle)
        rotation_matrix = np.asmatrix([[cosine, -sine], [sine, cosine]])
        return np.matmul(rotation_matrix, np.asmatrix([dir.x, dir.y]))


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