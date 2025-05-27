import pointsFromCAD as pfc
import geometricClasses as gcl
import typing as ty

class JointPoints(): 
    def __init__(self, fi, ri, fo, ro): #front-inner/rear-outer
        #TODO: implement a class made of 3d mid points that will represent a very idealised, outer dims only, version of the diagonal rib
        self.fi = fi
        self.ri = ri
        self.fo = fo
        self.ro = ro

def ray_rib_pattern(startTop=True, endTop=True) -> ty.Tuple[ty.List[JointPoints], ty.List[gcl.Line2D]]:
    '''Generates the diagonal ribs. Return points grouped in joint points plus the upper and lower spanwise directions'''
    raise NotImplementedError