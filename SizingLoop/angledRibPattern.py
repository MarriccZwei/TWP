import pointsFromCAD as pfc
import geometricClasses as gcl
import typing as ty

class JointPoints(): 
    def __init__(self, fi:gcl.Point3D, ri:gcl.Point3D, fo:gcl.Point3D, ro:gcl.Point3D): #front-inner/rear-outer
        #TODO: implement a class made of 3d mid points that will represent a very idealised, outer dims only, version of the diagonal rib
        self.fi = fi
        self.ri = ri
        self.fo = fo
        self.ro = ro

def ray_rib_pattern(startTop=True, endTop=True) -> ty.Tuple[ty.List[JointPoints], gcl.direction2D, ty.List[gcl.Line2D]]:
    '''Generates the diagonal ribs. Return points grouped in joint points plus the dihedral vector, upper and lower spanwise directions'''
    raise NotImplementedError