import aerosandbox as asb
import numpy as np
import numpy.typing as nt
import scipy.sparse as ss
import typing as ty
import scipy.spatial as ssp

class LoadCase():
    def __init__(self, n:float, nlg:float, DOF:float, N:float, g0:float, thrust_per_motor:float, op_point:asb.OperatingPoint, aeroelastic:bool=False):
        self.n = n #load factor
        self.nlg = nlg #landing gear normal force load factor
        self.N = N #total number of degrees of freedom in the model
        self.DOF = DOF #number of degrees of freedom per node
        assert N%DOF==0
        self.g0 = g0 #ravitational acceleration
        self.Ft = thrust_per_motor
        self.op = op_point
        self.aeroelastic = aeroelastic #do we evaluate flutter for this load case

        self.A = np.zeros(N) #here be the aerodynamic loads (basically lift)
        self.W = np.zeros(N) #the current value of the model's weight
        self.T = np.zeros(N) #here be the thrust
        self.L = np.zeros(N) #here be the landing loads
        #self.KA = np.zeros((N, N)) #here be the aerodynamic matrix


    def aerodynamic_matrix(self, ncoords_affected:nt.NDArray[np.float32], nid_pos_affected:nt.NDArray[np.int32],
                   les:ty.List, tes:ty.List, airfs:ty.List[asb.Airfoil],
                   ymin:float, bres:int=20, cres:int=10):
        '''Conducting an aerodynamic simulation based on skin nodes, returning Fext and KA'''
        #TODO: remove gcl refs
        vlm_, dFv_dp = self._calc_dFv_dp(les, tes, airfs, self.op, ymin, bres, cres, np.zeros(len(airfs)*2))
        W, self.A = self._aero2fem(vlm_, ncoords_affected, nid_pos_affected, self.N, self.DOF)
        W_u_to_p = self._fem2aero(les, np.zeros(len(airfs)*2), ncoords_affected, nid_pos_affected, self.N, self.DOF)
        self.KA = W @ dFv_dp @ W_u_to_p 
    
    def apply_aero(self, nid_pos_affected:nt.NDArray[np.int32], coords_affect:nt.NDArray[np.float64]):
        '''
        Applying the aerodynamic load to the given subset of nodes

        :param nid_pos_affected: matrix indices of the affected nodes
        :type nid_pos_affected: nt.NDArray[np.int32]
        :param coords_affect: ncoords-formatted coordinates of the nodes affected
        :type coords_affect: nt.NDArray[np.float64]
        '''

    def apply_thrust(self, nid_pos_affected:nt.NDArray[np.int32]):
        '''Applying the thrust at the affected nodes'''
    

    def update_weight(self, M:ss.coo_matrix):
        '''Applying weight to every node using a passed mass matrix, inclues weight rotation wrt aoa'''
        aoa = np.deg2rad(self.op.alpha)
        nNodes = self.N//self.DOF
        ntot = self.n+self.nlg
        gvect = [0]*self.DOF
        gvect[0] = ntot*self.g0*np.sin(aoa)
        gvect[2] = -ntot*self.g0*np.cos(aoa) 
        gvect = np.array(gvect*nNodes)  
        self.W = M@gvect

    def apply_landing(self, nid_pos_affected:nt.NDArray[np.int32]):
        '''Applying the landing load to every node affected, assume uniformly distributed, against the direction of weight'''
    
    def loadstack(self):
        return self.A+self.T+self.W+self.L

    #=====IMPLEMENTATION OF ROTATION WRT AOA==============================
    def _rotate_4_aoa(self, load:nt.NDArray[np.float32]):
        '''rotates a load from a global coordinate system to the body-centered coordinate system'''
    
    #=====IMPLEMENTATION AERODYNAMIC LOAD==================================
    def _aero2fem(self, vlm:asb.VortexLatticeMethod, ncoords_s:nt.NDArray[np.float32], ids_s:nt.NDArray[np.int32], N:int, DOF:int,
             number_of_neighbors=100):
        """ncoords selected - only selected nodes to which we apply the loads - will have to be tracked externally with ids_s"""

        # x, y, z coordinates of each node, NOTE assuming same order for DOFs
        vortex_valid = vlm.vortex_centers[:, 1] > ncoords_s[:, 1].min() # region represented by the FE model

        tree = ssp.cKDTree(ncoords_s)
        d, node_indices = tree.query(vlm.vortex_centers[vortex_valid], k=number_of_neighbors)

        power = 2
        weights = (1/d**power)/((1/d**power).sum(axis=1)[:, None])
        assert np.allclose(weights.sum(axis=1), 1)

        v = vortex_valid.sum()*3
        W = np.zeros((N, v))

        W *= 0
        for j in range(number_of_neighbors):
            for i, node_index in enumerate(node_indices[:, j]):
                #Here we cannot use the node_index directly, as we need to match the node indexing in the global mesh
                W[DOF*ids_s[node_index] + 0, i*3 + 0] += weights[i, j]
                W[DOF*ids_s[node_index] + 1, i*3 + 1] += weights[i, j]
                W[DOF*ids_s[node_index] + 2, i*3 + 2] += weights[i, j]
        assert np.allclose(W.sum(axis=0), 1.)

        Fv = vlm.forces_geometry[vortex_valid].flatten()
        Fext = W @ Fv #external forces to be applied to the fem model

        assert np.isclose(Fext[0::DOF].sum(), Fv[0::3].sum())
        assert np.isclose(Fext[1::DOF].sum(), Fv[1::3].sum())
        assert np.isclose(Fext[2::DOF].sum(), Fv[2::3].sum())

        return ss.csc_matrix(W), Fext

    def _fem2aero(self, les:ty.List, p:nt.NDArray[np.float32], ncoords_s:nt.NDArray[np.float32], ids_s:nt.NDArray[np.int32], N:int, DOF:int,
                number_of_neighbors=2):
        twists = p[len(les):]#also skipping the twist for now
        deform_dir = gcl.Direction3D(0,0,1)
        heave_displs =  p[:len(les)]
        leds = [deform_dir.step(le, p_) for le, p_ in zip(les, heave_displs)]

        tree = ssp.cKDTree(ncoords_s)
        d, node_indices = tree.query(gcl.pts2numpy3D(leds), k=number_of_neighbors)
        power = 2
        weights2 = (1/d**power)/((1/d**power).sum(axis=1)[:, None])
        assert np.allclose(weights2.sum(axis=1), 1)

        W_u_to_p = np.zeros((len(p)*3, N))

        for j in range(number_of_neighbors):
            for i, node_index in enumerate(node_indices): #we have 3 forces for 2 p vars per node, so 6
                W_u_to_p[i*6 + 0, DOF*node_index + 0] = weights2[i, j]
                W_u_to_p[i*6 + 1, DOF*node_index + 1] = weights2[i, j]
                W_u_to_p[i*6 + 2, DOF*node_index + 2] = weights2[i, j]
                W_u_to_p[i*6 + 3, DOF*node_index + 0] = weights2[i, j]
                W_u_to_p[i*6 + 4, DOF*node_index + 1] = weights2[i, j]
                W_u_to_p[i*6 + 5, DOF*node_index + 2] = weights2[i, j]
            
        return ss.csc_matrix(W_u_to_p)

    def _vlm(self, les:ty.List, tes:ty.List, airfs:ty.List[asb.Airfoil],
            op:asb.OperatingPoint, bres:int=20, cres:int=10, displs:ty.List[float]=None, return_sol=False):
        #allowing for initial displacements for flutter. Yet, for static analysis we dont's need displacements
        if displs is None:
            displs = np.zeros(len(airfs)*2)
        heaveDir = gcl.Direction3D(0,0,1)
        cs = [te.x-le.x for te, le in zip(tes, les)] #obtaining the chord list
        ptles = [heaveDir.step(le, displ) for le, displ in zip(les, displs[:len(airfs)])]


        airplane = asb.Airplane("E9X", xyz_ref=[0,0,0], wings = [
                                    asb.Wing(
                                        name="E9X_WING",
                                        symmetric=True,
                                        xsecs=[asb.WingXSec(
                                            xyz_le=[ptle.x, ptle.y, ptle.z],
                                            chord=c,
                                            twist=twist, 
                                            airfoil=airf
                                        )for ptle, c, airf, twist in zip(ptles, cs, airfs, displs[len(airfs):])],
                                    )
                                ])
        vlm = asb.VortexLatticeMethod(airplane, op, spanwise_resolution=bres, chordwise_resolution=cres)
        sol = vlm.run()
        forces = sol["F_g"]
        moments = sol["M_g"]

        if return_sol:
            return airplane, vlm, forces, moments, sol
        else:
            return airplane, vlm, forces, moments


    def _calc_dFv_dp(self, les:ty.List, tes:ty.List, airfs:ty.List[asb.Airfoil], #TODO: refactor the displs
            op:asb.OperatingPoint, ymin:float, bres:int=20, cres:int=10, displs:ty.List[float]=None, return_sol=False, epsilon=.01):
        
        airplane, vlm_, forces, moments = self._vlm(les, tes, airfs, op, bres, cres, displs, return_sol)

        vortex_valid = vlm_.vortex_centers[:, 1] > ymin
        v = vortex_valid.sum()*3
        Fv = vlm_.forces_geometry[vortex_valid].flatten()

        dFv_dp = np.zeros((v, len(displs)*3)) # NOTE remember to pass both heave and twist displacements
        for i in range(len(displs)-1):
            p_DOF = 3*(i + 1) + 2 # heave DOF starting at second airfoil
            p2 = displs.copy()
            p2[i+1] += epsilon

            plane2, vlm2, f2, M2 = self._vlm(les, tes, airfs, op, bres, cres, p2, return_sol)
            Fv2 = vlm2.forces_geometry[vortex_valid].flatten()
            dFv_dp[:, p_DOF] += (Fv2 - Fv)/epsilon
        dFv_dp = ss.csc_matrix(dFv_dp)
        return vlm_, dFv_dp
