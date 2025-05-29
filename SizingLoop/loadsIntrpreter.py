import loadModel as ldm
import typing as ty
import geometricClasses as gcl

def aerodynamic(n, nodes:ty.List[ty.List[gcl.Point3D]], eles:ty.List, dihedralVec:gcl.direction2D):
    #using dihedralVec assumes unswept wing; n is load factor
    '''Returns a list of nodes with the same format as nodes, 
    with the equivalent aerodynamic load acting on these nodes in the main reference frame,
    which is why the dihedral direction is needed'''
    raise NotImplementedError

def fixed_inertial(n, nodes:ty.List[ty.List[gcl.Point3D]], eles:ty.List, dihedralVec:gcl.direction2D):
    '''Inertial loads of everything else than components'''
    raise NotImplementedError

def prop_tt(n, nodes:ty.List[ty.List[gcl.Point3D]], eles:ty.List, dihedralVec:gcl.direction2D):
    '''Thrust and torque from propellers'''
    raise NotImplementedError

def tot_fixed_load(n, nodes:ty.List[ty.List[gcl.Point3D]], eles:ty.List, dihedralVec:gcl.direction2D):
    '''a complete set of unchanging loads'''
    raise NotImplementedError