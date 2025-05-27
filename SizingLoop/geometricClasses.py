class Point2D():
    def __init__(self, x:float, y:float):
        self.x = x
        self.y = y

class direction2D():
    def __init__(self, dx:float, dy:float):
        magn = (dx**2+dy**2)**.5
        self.ux = dx/magn
        self.uy = dy/magn


class Line2D():
    def __init__(self, pt:Point2D, dir:direction2D):
        self.pt = pt
        self.dir = dir

class Point3D():
    def __init__(self, x:float, y:float, z:float):
        self.x = x
        self.y = y
        self.z = z