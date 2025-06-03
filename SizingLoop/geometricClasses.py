import numpy as np

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