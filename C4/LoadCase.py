import aerosandbox as asb
import numpy as np
import numpy.typing as nt
import scipy.sparse as ss

class LoadCase():
    def __init__(self, n, N_DOF, g0, thrust_per_motor, op_point:asb.OperatingPoint):
        self.n = n #load factor
        self.N = N_DOF #total number of degrees of freedom in the model
        self.g0 = g0 #ravitational acceleration
        self.Ft = thrust_per_motor
        self.op = op_point
        self._F = np.zeros(N_DOF) #here be the unchanging loads
        self._W = np.zeros(N_DOF) #the current value of the model's weight
        #TODO: change to each of the forces being accessible as a property

    def apply_Fext(self, ncoords_skin:nt.NDArray[np.float32], nid_pos_affected:nt.NDArray[np.int32]):
        '''Conducting an aerodynamic simulation based on skin nodes, returning Fext AND KA
        Includes updatingself._F'''
        Fext = np.zeros(self.N)
        #obtain just Fext from an aerodynamic simulation
        return Fext
    
    def apply_Thrust(self, DOF:int, nid_pos_affected:nt.NDArray[np.int32]):
        '''Applying the thrust at the affected nodes
        Includes updating self._F'''
        Fthrust = np.zeros(self.N)
        return Fthrust
    
    def update_Weight(self, M:ss.coo_matrix):
        '''Applying weight to every node using a passed mass matrix, inclues weight rotation wrt aoa
        Includes updating self._W'''
        W = np.zeros(self.N)
        return W
    
    def loadstack(self):
        return self._F+self._W
