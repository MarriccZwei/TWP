from typing import Tuple, List, Dict
import numpy as np
#structural elements

class Joint():
    def __init__(self, point:Tuple[float], extLoad:Tuple[float]=(0,0,0)):
        self.x = point[0]
        self.y = point[1]
        self.z = point[2]
        self.Fx = extLoad[0]
        self.Fy = extLoad[1]
        self.Fz = extLoad[2]

class Truss():
    def __init__(self, node1key:str, node2key:str, rout:float, rin:float):
        self.node1key = node1key
        self.node2key = node2key
        self.rout = rout
        self.rin = rin
        self.L = None
    
    def init_len(self, nodedict:Dict[str, Joint]):
        j1 = nodedict[self.node1key]
        j2 = nodedict[self.node2key]
        self.L = np.sqrt((j1.x-j2.x)**2+(j1.y-j2.y)**2+(j1.z-j2.z)**2)