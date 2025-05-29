import loadModel as ldm
import typing as ty
import geometricClasses as gcl

class EquivalentPtLoad():
    def __init__(self, Fx:float, Fy:float, Fz:float, Mx:float, My:float, Mz:float):
        self.Fx = Fx
        self.Fy = Fy
        self.Fz = Fz
        self.Mx = Mx
        self.My = My
        self.Mz = Mz
    
    @classmethod
    def superimpose(cls, depth, *loadlists):
        '''allow for superposition of load lists with our shapes'''
        raise NotImplementedError

def aerodynamic(n:float, nlg:float, nodes:ty.List[ty.List[gcl.Point3D]], eles:ty.List, dihedralVec:gcl.direction2D)->ty.List[ty.List[EquivalentPtLoad]]:
    #using dihedralVec assumes unswept wing; n is load factor
    '''Returns a list of nodes with the same format as nodes, 
    with the equivalent aerodynamic load acting on these nodes in the main reference frame,
    which is why the dihedral direction is needed'''
    '''takes a 1/nodes portion of the load acting on an element to the node'''
    raise NotImplementedError

#TODO: see if you need it - re-computing all inertial is much simpler and probs fine
def fixed_inertial(n:float, nlg:float, nodes:ty.List[ty.List[gcl.Point3D]], eles:ty.List, dihedralVec:gcl.direction2D)->ty.List[ty.List[EquivalentPtLoad]]:
    '''Inertial loads of everything else than components, including an approximate spread of load from batteries (inc. in eles)'''
    raise NotImplementedError

def prop_tt(n:float, nlg:float, nodes:ty.List[ty.List[gcl.Point3D]], eles:ty.List, dihedralVec:gcl.direction2D)->ty.List[ty.List[EquivalentPtLoad]]:
    '''Thrust and torque from propellers'''
    raise NotImplementedError

def lg(n:float, nlg:float, nodes:ty.List[ty.List[gcl.Point3D]], eles:ty.List, dihedralVec:gcl.direction2D)->ty.List[ty.List[EquivalentPtLoad]]:
    '''load on the landing gear while landing'''
    raise NotImplementedError

def tot_fixed_load(n:float, nlg:float, nodes:ty.List[ty.List[gcl.Point3D]], eles:ty.List, dihedralVec:gcl.direction2D)->ty.List[ty.List[EquivalentPtLoad]]:
    '''a complete set of unchanging loads'''
    raise NotImplementedError

def var_load(n:float, nlg:float, nodes:ty.List[ty.List[gcl.Point3D]], eles:ty.List, dihedralVec:gcl.direction2D)->ty.List[ty.List[EquivalentPtLoad]]:
    '''the variable inertial load that has to be updated every loop iteration'''
    raise NotImplementedError