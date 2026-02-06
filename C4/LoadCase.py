import aerosandbox as asb
import aerosandbox.numpy as np
import numpy.typing as nt
import scipy.sparse as ss
import typing as ty
import scipy.spatial as ssp
import pyfe3d as pf3
import scipy.sparse.linalg as ssl

class LoadCase():
    def __init__(self, n:float, MTOM:float, N:float, g0:float, 
                 thrust_total:float, op_point:asb.OperatingPoint, les:nt.NDArray, tes:nt.NDArray,
                 airfs:ty.List[asb.Airfoil], bres:int=20, cres:int=10, nneighs:int=100, nneighs_p:int=10,
                 aeroelastic:bool=False):
        self.n = n #load factor
        self.N = N #total number of degrees of freedom in the model
        assert N%pf3.DOF==0
        self.g0 = g0 #gravitational acceleration
        self.Ft = thrust_total
        self.op = op_point
        self.MTOM = MTOM
        self.aeroelastic = aeroelastic #do we evaluate flutter for this load case

        self.les = les
        self.tes = tes
        self.airfs = airfs
        self.bres = bres
        self.cres = cres
        self.nneighs = nneighs
        self.nneighs_p = nneighs_p

        self._cs = [self.tes[i,0]-self.les[i,0] for i in range(self.les.shape[0])] #obtaining the chord list

        self.A = np.zeros(N) #here be the aerodynamic loads (basically lift)
        self.W = np.zeros(N) #the current value of the model's weight
        self.T = np.zeros(N) #here be the thrust


    def aerodynamic_matrix(self, nid_pos_affected:nt.NDArray[np.int32], ncoords_affected:nt.NDArray[np.float32]):
        '''Conducting an aerodynamic simulation based on skin nodes, updating A and KA'''

        vlm_, dFv_dp = self._calc_dFv_dp(ncoords_affected[:,1].min())
        W, self.A = self._aero2fem(vlm_, ncoords_affected, nid_pos_affected, 1.)
        W_u_to_p = self._fem2aero(ncoords_affected, nid_pos_affected)
        self.KA = W @ dFv_dp @ W_u_to_p


    def apply_aero(self, nid_pos_affected:nt.NDArray[np.int32], coords_affected:nt.NDArray[np.float64]):
        '''
        Applying the aerodynamic load to the given subset of nodes

        :param nid_pos_affected: matrix indices of the affected nodes
        :type nid_pos_affected: nt.NDArray[np.int32]
        :param coords_affect: ncoords-formatted coordinates of the nodes affected
        :type coords_affect: nt.NDArray[np.float64]
        '''
        _, vlm, forces, _ = self._vlm()
        ratio = self.MTOM*self.g0*self.n*np.cos(np.deg2rad(self.op.alpha))/forces[2]
        _, self.A = self._aero2fem(vlm, coords_affected, nid_pos_affected, ratio)


    def apply_thrust(self, nid_pos_affected:nt.NDArray[np.int32]):
        '''Applying the thrust at the affected nodes'''
        self.T[0::pf3.DOF][nid_pos_affected] = -self.Ft/2/len(nid_pos_affected)
    

    def update_weight(self, M:ss.coo_matrix):
        '''Applying weight to every node using a passed mass matrix, inclues weight rotation wrt aoa'''
        aoa = np.deg2rad(self.op.alpha)
        nNodes = self.N//pf3.DOF
        gvect = [0]*pf3.DOF
        gvect[0] = self.n*self.g0*np.sin(aoa)
        gvect[2] = -self.n*self.g0*np.cos(aoa) 
        gvect = np.array(gvect*nNodes)  
        self.W = M@gvect


    def loadstack(self):
        return self.A+self.T+self.W
    

    #=====IMPLEMENTATION AERODYNAMIC LOAD==================================
    def _aero2fem(self, vlm:asb.VortexLatticeMethod, ncoords_s:nt.NDArray[np.float32], ids_s:nt.NDArray[np.int32], ratio:int):
        """ncoords selected - only selected nodes to which we apply the loads - will have to be tracked externally with ids_s"""

        # x, y, z coordinates of each node, NOTE assuming same order for DOFs
        vortex_valid = vlm.vortex_centers[:, 1] > ncoords_s[:, 1].min() # region represented by the FE model

        tree = ssp.cKDTree(ncoords_s)
        d, node_indices = tree.query(vlm.vortex_centers[vortex_valid], k=self.nneighs)

        power = 2
        weights = (1/d**power)/((1/d**power).sum(axis=1)[:, None])
        assert np.allclose(weights.sum(axis=1), 1)

        v = vortex_valid.sum()*3
        W = np.zeros((self.N, v))

        for j in range(self.nneighs):
            for i, node_index in enumerate(node_indices[:, j]):
                #Here we cannot use the node_index directly, as we need to match the node indexing in the global mesh
                W[pf3.DOF*ids_s[node_index] + 0, i*3 + 0] += weights[i, j]
                W[pf3.DOF*ids_s[node_index] + 1, i*3 + 1] += weights[i, j]
                W[pf3.DOF*ids_s[node_index] + 2, i*3 + 2] += weights[i, j]
        assert np.allclose(W.sum(axis=0), 1.)

        Fv = vlm.forces_geometry[vortex_valid].flatten()
        #re-scaling the aerodynamic load to match the specified load factor, done due to some load factors not being achievable w\o hlds
        Fv = ratio*Fv
        Fext = W @ Fv #external forces to be applied to the fem model

        assert np.isclose(Fext[0::pf3.DOF].sum(), Fv[0::3].sum())
        assert np.isclose(Fext[1::pf3.DOF].sum(), Fv[1::3].sum())
        assert np.isclose(Fext[2::pf3.DOF].sum(), Fv[2::3].sum())

        return ss.csc_matrix(W), Fext


    def _fem2aero(self, ncoords_s:nt.NDArray[np.float32], ids_s:nt.NDArray[np.int32]):
        displs = np.vstack((self.les, self.tes))
        assert displs.shape == (len(self.les)*2, 3)

        tree = ssp.cKDTree(ncoords_s)
        d, node_indices = tree.query(displs, k=self.nneighs_p)
        power = 2
        weights2 = (1/d**power)/((1/d**power).sum(axis=1)[:, None])
        assert np.allclose(weights2.sum(axis=1), 1)

        W_u_to_p = np.zeros((len(displs)*3, self.N))

        for j in range(self.nneighs_p):
            for i, node_index in enumerate(node_indices[:, j]): #we have 3 forces for 2 p vars per node, so 6
                W_u_to_p[i*3+0, pf3.DOF*ids_s[node_index] + 0] += weights2[i, j]
                W_u_to_p[i*3+1, pf3.DOF*ids_s[node_index] + 1] += weights2[i, j]
                W_u_to_p[i*3+2, pf3.DOF*ids_s[node_index] + 2] += weights2[i, j]
            
        return ss.csc_matrix(W_u_to_p)


    def _vlm(self, displs:ty.List[float]=None, return_sol=False):
        #allowing for initial displacements for flutter. Yet, for static analysis we dont's need displacements
        if displs is None:
            displs = np.zeros(len(self.airfs)*2)

        #converting LE-TE displs into LE displs and twists
        ledispls = displs[:self.les.shape[0]]
        tedispls = displs[self.les.shape[0]:len(displs)]
        ptles = [self.les[i,:]+np.array([0,0,displs[i]]) for i, led in enumerate(ledispls)]
        crange = .55 #from .15 to .7 of chord
        twists = [np.arctan((led-ted)/c/crange) for led, ted, c in zip(ledispls, tedispls, self._cs)]

        airplane = asb.Airplane("E9X", xyz_ref=[0,0,0], wings = [
                                    asb.Wing(
                                        name="E9X_WING",
                                        symmetric=True,
                                        xsecs=[asb.WingXSec(
                                            xyz_le=[ptle[0], ptle[1], ptle[2]],
                                            chord=c,
                                            twist=np.rad2deg(twist), #asb has aoas and twists in degrees 
                                            airfoil=airf
                                        )for ptle, c, airf, twist in zip(ptles, self._cs, self.airfs, twists)],
                                    )
                                ])
        vlm = asb.VortexLatticeMethod(airplane, self.op, spanwise_resolution=self.bres, chordwise_resolution=self.cres)
        sol = vlm.run()
        forces = sol["F_g"]
        moments = sol["M_g"]

        if return_sol:
            return airplane, vlm, forces, moments, sol
        else:
            return airplane, vlm, forces, moments


    def _calc_dFv_dp(self, ymin:float, epsilon=.01):
        p = np.zeros(len(self.airfs)*2)
        airplane, vlm_, forces, moments = self._vlm(p)
        #ratio = self.MTOM*self.g0*self.n*np.cos(np.deg2rad(self.op.alpha))/forces[2]

        vortex_valid = vlm_.vortex_centers[:, 1] > ymin
        v = vortex_valid.sum()*3
        Fv = vlm_.forces_geometry[vortex_valid].flatten()

        dFv_dp = np.zeros((v, len(p)*3)) # NOTE remember to pass both heave and twist displacements
        for i in range(len(p)-1):
            if i != len(self.les)-1: #we skip the te displ of the first foil, just as we skip its le displ
                p_DOF = 3*(i+1)+2 # heave DOF starting at second airfoil
                p2 = p.copy()
                p2[i+1] += epsilon

                plane2, vlm2, f2, M2 = self._vlm(p2)
                Fv2 = vlm2.forces_geometry[vortex_valid].flatten()
                deriv = (Fv2 - Fv)/epsilon
                dFv_dp[:, p_DOF] += deriv

        dFv_dp = ss.csc_matrix(dFv_dp)
        return vlm_, dFv_dp