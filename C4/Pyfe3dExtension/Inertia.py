from dataclasses import dataclass
import numpy as np
import numpy.typing as nt

@dataclass
class InertiaProp:
    '''
    Representing a mass and an inertia tensor with centroid at the xcg, ycg, zcg coordinates from attachment point
    '''
    m: float
    dxcg: float = 0.0
    dycg: float = 0.0
    dzcg: float = 0.0
    Ixx: float = 0.0
    Ixy: float = 0.0
    Ixz: float = 0.0
    Iyy: float = 0.0
    Iyz: float = 0.0
    Izz: float = 0.0

    def __post_init__(self):
        for name, value in vars(self).items():
            if not isinstance(value, float):
                raise TypeError(f"{name} must be float, got {type(value).__name__}")

        if self.m <= 0.0:
            raise ValueError("m (mass) must be positive")
        
@dataclass
class InertiaData:
    M_SPARSE_SIZE = 24
        

class Inertia:
    '''
    Element representing translated rigid body inertia.
    '''
    def __init__(self, data:InertiaData):
        self.M_SPARSE_SIZE = data.M_SPARSE_SIZE
        self.n1:int = None
        self.c1:int = None
        self.init_k_M:int = None
    
    def update_m(self, Mr:nt.NDArray[np.int32], Mc:nt.NDArray[np.int32], Mv:nt.NDArray[np.float64], prop:InertiaProp):
        '''
        Docstring for update_m
        
        :param Mr: Array to store row positions of sparse values
        :type Mr: nt.NDArray[np.int32]
        :param Mc: Array to store column positions of sparse values
        :type Mc: nt.NDArray[np.int32]
        :param Mv: Array to store sparse values
        :type Mv: nt.NDArray[np.float64]
        :param prop: inertia prop to be used during mass matrix updates
        :type prop: InertiaProp
        '''
        m = prop.m
        rx = prop.dxcg
        ry = prop.dycg
        rz = prop.dzcg
        Ixx = prop.Ixx
        Ixy = prop.Ixy
        Ixz = prop.Ixz
        Iyy = prop.Iyy
        Iyz = prop.Iyz
        Izz = prop.Izz

        k = self.init_k_M

        #mass terms
        for i in range(3):
            Mr[k] = i+self.c1
            Mc[k] = i+self.c1
            Mv[k] = m
            k+=1

        #inertia->mass coupling terms
        for r, c, v in zip([0, 0, 1, 1, 2, 2], [4, 5, 3, 5, 3, 4], [m*rz, -m*ry, -m*rz, m*rx, m*ry, -m*rx]):
            Mr[k] = r+self.c1
            Mc[k] = c+self.c1
            Mv[k] = v
            k+=1

        #mass->inertia coupling terms
        for r, c, v in zip([3, 3, 4, 4, 5, 5], [1, 2, 0, 2, 0, 1], [-m*rz, m*ry, m*rz, -m*rx, -m*ry, m*rx]):
            Mr[k] = r+self.c1
            Mc[k] = c+self.c1
            Mv[k] = v
            k+=1

        #inertia terms
        for r, c, v in zip([3, 3, 3, 4, 4, 4, 5, 5, 5], [3, 4, 5, 3, 4, 5, 3, 4, 5], 
                           [Ixx+m*(ry**2+rz**2), Ixy-m*rx*ry, Ixz-m*rx*rz, 
                            Ixy-m*rx*ry, Iyy+m*(rx**2+rz**2), Iyz-m*ry*rz, 
                            Ixz-m*rx*rz, Iyz-m*ry*rz, Izz+m*(rx**2+ry**2)]):
            Mr[k] = r+self.c1
            Mc[k] = c+self.c1
            Mv[k] = v
            k+=1
