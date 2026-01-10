import aerosandbox as asb
import aerosandbox.numpy as np
import numpy.typing as nt
import scipy.sparse as ss
import typing as ty
import scipy.spatial as ssp
import pyfe3d as pf3
import scipy.sparse.linalg as ssl

class LoadCase():
    def __init__(self, n:float, nlg:float, MTOM:float, N:float, g0:float, 
                 thrust_total:float, op_point:asb.OperatingPoint, les:nt.NDArray, tes:nt.NDArray,
                 airfs:ty.List[asb.Airfoil], bres:int=20, cres:int=10, nneighs:int=100,
                 aeroelastic:bool=False):
        self.n = n #load factor
        self.nlg = nlg #landing gear normal force load factor
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
        self.nneighs = nneighs #numer of neighbours

        self.A = np.zeros(N) #here be the aerodynamic loads (basically lift)
        self.W = np.zeros(N) #the current value of the model's weight
        self.T = np.zeros(N) #here be the thrust
        self.L = np.zeros(N) #here be the landing loads


    def aerodynamic_matrix(self, nid_pos_affected:nt.NDArray[np.int32], ncoords_affected:nt.NDArray[np.float32]):
        '''Conducting an aerodynamic simulation based on skin nodes, updating A and KA'''

        vlm_, dFv_dp = self._calc_dFv_dp(ncoords_affected[:,1].min())
        W, self.A = self._aero2fem(vlm_, ncoords_affected, nid_pos_affected, 1.)
        W_u_to_p = self._fem2aero(np.zeros(len(self.airfs)*2), ncoords_affected, nid_pos_affected)
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
        ntot = self.n+self.nlg
        gvect = [0]*pf3.DOF
        gvect[0] = ntot*self.g0*np.sin(aoa)
        gvect[2] = -ntot*self.g0*np.cos(aoa) 
        gvect = np.array(gvect*nNodes)  
        self.W = M@gvect


    def apply_landing(self, nid_pos_affected:nt.NDArray[np.int32]):
        '''Applying the landing load to every node affected, assume uniformly distributed, against the direction of weight'''
        nnodes = len(nid_pos_affected)
        fl = np.zeros(nnodes*pf3.DOF)
        NL = self.MTOM*self.g0*self.nlg/2/nnodes #overall landing load per node in the earth upwards direction
        NLx = -NL*np.sin(np.deg2rad(self.op.alpha))
        NLz = NL*np.cos(np.deg2rad(self.op.alpha))
        fl[0::pf3.DOF] = NLx
        fl[2::pf3.DOF] = NLz

        self.L = np.zeros(self.N)
        self.L[0::pf3.DOF][nid_pos_affected] = fl[0::pf3.DOF]
        self.L[2::pf3.DOF][nid_pos_affected] = fl[2::pf3.DOF]
    

    def loadstack(self):
        return self.A+self.T+self.W+self.L
    

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


    def _fem2aero(self, p:nt.NDArray[np.float32], ncoords_s:nt.NDArray[np.float32], ids_s:nt.NDArray[np.int32]):
        heave_displs =  p[:len(self.les)]
        leds = [self.les[i, :]+np.array([0.,0.,heave_displs[i]]) for i in range(self.les.shape[0])]
        #ledts = leds+[np.array([0., 0., 0.])]*(len(self.les)-1) #adding non-displaced points 4 twist

        tree = ssp.cKDTree(ncoords_s)
        d, node_indices = tree.query(leds, k=self.nneighs)
        power = 2
        weights2 = (1/d**power)/((1/d**power).sum(axis=1)[:, None])
        assert np.allclose(weights2.sum(axis=1), 1)

        W_u_to_p = np.zeros((len(p)*3, self.N))

        for j in range(self.nneighs):
            for i, node_index in enumerate(node_indices): #we have 3 forces for 2 p vars per node, so 6
                W_u_to_p[i*3+0, pf3.DOF*ids_s[node_index] + 0] += weights2[i, j]
                W_u_to_p[i*3+1, pf3.DOF*ids_s[node_index] + 1] += weights2[i, j]
                W_u_to_p[i*3+2, pf3.DOF*ids_s[node_index] + 2] += weights2[i, j]
                W_u_to_p[i*3+0+len(heave_displs), pf3.DOF*ids_s[node_index] + 3] += weights2[i, j]
                W_u_to_p[i*3+1+len(heave_displs), pf3.DOF*ids_s[node_index] + 4] += weights2[i, j]
                W_u_to_p[i*3+2+len(heave_displs), pf3.DOF*ids_s[node_index] + 5] += weights2[i, j]
            
        return ss.csc_matrix(W_u_to_p)


    def _vlm(self, displs:ty.List[float]=None, return_sol=False):
        #allowing for initial displacements for flutter. Yet, for static analysis we dont's need displacements
        if displs is None:
            displs = np.zeros(len(self.airfs)*2)
        cs = [self.tes[i,0]-self.les[i,0] for i in range(self.les.shape[0])] #obtaining the chord list
        ptles = [self.les[i,:]+np.array([0,0,displs[i]]) for i in range(self.les.shape[0])]


        airplane = asb.Airplane("E9X", xyz_ref=[0,0,0], wings = [
                                    asb.Wing(
                                        name="E9X_WING",
                                        symmetric=True,
                                        xsecs=[asb.WingXSec(
                                            xyz_le=[ptle[0], ptle[1], ptle[2]],
                                            chord=c,
                                            twist=np.rad2deg(twist), 
                                            airfoil=airf
                                        )for ptle, c, airf, twist in zip(ptles, cs, self.airfs, displs[len(self.airfs):])],
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


    def _calc_dFv_dp(self, ymin:float, displs:ty.List[float]=None, epsilon=.01):
        if displs is None:
            displs = np.zeros(len(self.airfs)*2)
        airplane, vlm_, forces, moments = self._vlm(displs)
        #ratio = self.MTOM*self.g0*self.n*np.cos(np.deg2rad(self.op.alpha))/forces[2]

        vortex_valid = vlm_.vortex_centers[:, 1] > ymin
        v = vortex_valid.sum()*3
        Fv = vlm_.forces_geometry[vortex_valid].flatten()

        dFv_dp = np.zeros((v, len(displs)*3)) # NOTE remember to pass both heave and twist displacements
        for i in range(len(displs)-1):
            if i != len(self.les)-1: #we skip the twist of the first foil, just as we skip its displacement
                p_DOF = (3*i+2) if (i<len(self.les)-1) else (3*i+1) # heave DOF starting at second airfoil
                p2 = displs.copy()
                p2[i+1] += epsilon

                plane2, vlm2, f2, M2 = self._vlm(p2)
                Fv2 = vlm2.forces_geometry[vortex_valid].flatten()
                deriv = (Fv2 - Fv)/epsilon
                dFv_dp[:, p_DOF] += deriv

        dFv_dp = ss.csc_matrix(dFv_dp)
        return vlm_, dFv_dp