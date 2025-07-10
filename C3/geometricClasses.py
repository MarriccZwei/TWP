import numpy as np
import typing as ty
import numpy.typing as nt

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

class MeshConn3D():
    "a struct used for connection list by Mesh3D"
    def __init__(self, ids:ty.List[int], elename:str, eleid:str):
        self.ids = ids
        self.elename = elename
        self.eleid = eleid

class Mesh3D():
    def __init__(self):
        self.nodes:ty.List[Point3D] = list()
        self.connections:ty.List[MeshConn3D] = list()

    def pyfe3D(self):
        "here be all the re-formatting of the code so it can be used with pyfe3D"
        ncoords = np.vstack(([node.x for node in self.nodes],
                             [node.y for node in self.nodes],
                             [node.z for node in self.nodes]))
        x = ncoords[:, 0]
        y = ncoords[:, 1]
        z = ncoords[:, 2]
        ncoords_flatten = ncoords.flatten()
        return ncoords, x, y, z, ncoords_flatten
    
    def register(self, pts:ty.List[Point3D]):
        "registers a list of points into the mesh, returns the ID space they occupy"
        "whatever data structure is used for the points, has to be list compatible and flattened before registration"
        "TODO: returns a list of special indices"
        firstIndex = len(self.nodes)
        self.nodes += pts
        lastIndex = len(self.nodes)-1
        return np.linspace(firstIndex, lastIndex, len(pts), dtype=int)
    
    def quad_interconnect(self, ids, eleid):
        "creates a set of quad connections between elements taking the four adjacent elements on the list"
        "assumes a list where nodes next to each other are next to each other on the array"
        n1s = ids[:-1, :-1].flatten()
        n2s = ids[1:, :-1].flatten()
        n3s = ids[1:, 1:].flatten()
        n4s = ids[:-1, 1:].flatten()
        for i in range(len(n1s)):
            self.connections.append(MeshConn3D([n1s[i], n2s[i], n3s[i], n4s[i]], "quad", eleid))
    
    def quad_connect(self, ids1, ids2, eleid):
        n1s = ids1[:-1]
        n2s = ids1[1:]
        n3s = ids2[1:]
        n4s = ids2[:-1]
        for i in range(len(n1s)):
            self.connections.append(MeshConn3D([n1s[i], n2s[i], n3s[i], n4s[i]], "quad", eleid))

    def spring_connect(self, ids1, ids2, eleid):
        for id1, id2 in zip(ids1, ids2):
            self.connections.append(MeshConn3D([id1, id2], "spring", eleid))

    def spring_interconnect(self, ids, eleid):
        for id1, id2 in zip(ids[:-1], ids[1:]):
            self.connections.append(MeshConn3D([id1, id2], "spring", eleid))

    def pointmass_attach(self, id_, eleid): #used to attach extra point mass to a given point
        self.connections.append(MeshConn3D([id_], "mass", eleid))


'''Utility functions with geometric classes'''

def pts2coords2D(pts:ty.List[Point2D]):
    return [pt.x for pt in pts], [pt.y for pt in pts]

def pts2coords3D(pts:ty.List[Point3D]):
    return [pt.x for pt in pts], [pt.y for pt in pts], [pt.z for pt in pts]

def multi_section_sheet3D(pt1s:ty.List[Point3D], pt2s:ty.List[Point3D], nsect:int, nLs:ty.List[int]):
    #make sure that all pt1s are on the same side of the sheet, same with pt2s!
    #any n must be greater or equal to 2, one less intem on nLs than on pt1s
    #required list sizes, you need at least 2 segments:
    assert len(nLs)+1==len(pt1s)==len(pt2s)
    sections = [pt1.pts_between(pt2, nsect) for pt1, pt2 in zip(pt1s, pt2s)]
    points = list() #the final list of created points
    for i in range(nsect):
        pointRow = sections[0][i].pts_between(sections[1][i], nLs[0]) #the first segment include both corners
        for j in range(1, len(sections)-1): #all other sections - ommit the 1st point 2 avoid dupes
            pointRow += sections[j][i].pts_between(sections[j+1][i], nLs[j])[1:]
        points.append(pointRow)
    pts = np.array(points)

    assert pts.shape[0] == ns
    assert pts.shape[1] == sum(nLs)-len(nLs)+1
    #first dimension is ns, the other is sum of nLs +1 - len(nLs)!
    return pts

if __name__ == "__main__":
    pt1s = [Point3D(0, 0, 0), Point3D(3, 0, .5), Point3D(4, -1, -.5), Point3D(4, 0, 4)]
    pt2s = [Point3D(0, 6, 1), Point3D(2.5, 7, 0), Point3D(3, 6, -.5), Point3D(4, 5, 5)]
    ns = 7
    nLs = [2, 13, 7]
    msp = multi_section_sheet3D(pt1s, pt2s, ns, nLs)
    from mpl_toolkits import mplot3d
    import matplotlib.pyplot as plt
    # fig = plt.figure()
    # ax = plt.axes(projection='3d')

    msp_flat = msp.flatten()
    # plt.plot(*pts2coords3D(msp_flat))
    # plt.plot(*pts2coords3D(msp[1:-1, 1:-1].flatten()))

    # plt.plot(*pts2coords3D(msp[0, :]))
    # plt.plot(*pts2coords3D(msp[-1, :]))
    # plt.plot(*pts2coords3D(msp[:, 0]))
    # plt.plot(*pts2coords3D(msp[:, -1]))

    #using the sheet test the mesh
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    mesh = Mesh3D()
    idspace = mesh.register(list(msp_flat))
    idgrid = idspace.reshape(msp.shape)
    #print(idgrid)
    mesh.quad_interconnect(idgrid, 'test')
    mesh.spring_interconnect(idgrid[0, :], 'test')
    for i in idspace[::4]: mesh.pointmass_attach(i, 'test') 
    mesh.quad_connect(idgrid[:4, 0], idgrid[:4, -1], 'test')
    mesh.spring_connect(idgrid[4:, 0], idgrid[4:, -1], 'test')

    scatter = []
    for conn in mesh.connections[::4]:
        pts = [mesh.nodes[i] for i in conn.ids]
        if len(pts)>1:
            plt.plot(*pts2coords3D(pts))
        else:
            scatter+=pts
    plt.scatter(*pts2coords3D(scatter))
    plt.plot(*pts2coords3D([mesh.nodes[i] for i in mesh.connections[-1].ids]))

    plt.show()