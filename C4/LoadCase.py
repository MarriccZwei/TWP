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
                 airfs:ty.List[asb.Airfoil], bres:int=20, cres:int=10, nneighs:int=100,
                 aeroelastic:bool=False, nlg:float=0., bank:float=0.):
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

        self.nlg = nlg #landing load factor, applied to one gear
        self.bank = bank #bank angle, give in degrees, internally converted to rad

        self._cs = [self.tes[i,0]-self.les[i,0] for i in range(self.les.shape[0])] #obtaining the chord list

        self.A = np.zeros(N) #here be the aerodynamic loads (basically lift)
        self.W = np.zeros(N) #the current value of the model's weight
        self.T = np.zeros(N) #here be the thrust
        self.L = np.zeros(N) #here be the landing load

        self.pg = 1/np.sqrt(1-(self.op.velocity/self.op.atmosphere.speed_of_sound())**2) #Prandtl-Glauert correction


    def aerodynamic_matrix(self, nid_pos_affected:nt.NDArray[np.int32], ncoords_affected:nt.NDArray[np.float32], debug=False):
        '''Conducting an aerodynamic simulation based on skin nodes, updating A and KA'''

        vlm_, dFv_dp = self._calc_dFv_dp(ncoords_affected[:,1].min(), debug=debug)
        W, self.A = self._aero2fem(vlm_, ncoords_affected, nid_pos_affected, 1.)
        W_u_to_p = self._fem2aero(ncoords_affected, nid_pos_affected)
        self.KA = W @ dFv_dp @ W_u_to_p


    def apply_aero(self, nid_pos_affected:nt.NDArray[np.int32], coords_affected:nt.NDArray[np.float64], debug=False):
        '''
        Applying the aerodynamic load to the given subset of nodes

        :param nid_pos_affected: matrix indices of the affected nodes
        :type nid_pos_affected: nt.NDArray[np.int32]
        :param coords_affect: ncoords-formatted coordinates of the nodes affected
        :type coords_affect: nt.NDArray[np.float64]
        '''
        _, vlm, _, _, lift = self._vlm(return_lift=True, debug=debug)
        ratio = self.MTOM*self.g0*self.n/lift
        _, self.A = self._aero2fem(vlm, coords_affected, nid_pos_affected, ratio)


    def apply_thrust(self, nid_pos_affected:nt.NDArray[np.int32]):
        '''Applying the thrust at the affected nodes'''
        self.T[0::pf3.DOF][nid_pos_affected] = -self.Ft/2/len(nid_pos_affected)
    

    def update_weight(self, M:ss.coo_matrix):
        '''Applying weight to every node using a passed mass matrix, inclues weight rotation wrt aoa'''
        aoa = np.deg2rad(self.op.alpha)
        bank = np.deg2rad(self.bank)
        nNodes = self.N//pf3.DOF
        gmagn = (self.n+self.nlg)*self.g0
        gvect = [0]*pf3.DOF
        gvect[0] = gmagn*np.sin(aoa)*np.cos(bank)
        gvect[1] = gmagn*np.sin(bank)
        gvect[2] = -gmagn*np.cos(aoa)*np.cos(bank) 
        gvect = np.array(gvect*nNodes)  
        self.W = M@gvect


    def apply_landing(self, nid_pos_affected:nt.NDArray[np.int32]):
        '''Applying the landing load to the affected nodes
        NOTE: there is no moment correction so centroid of the nodes has to be at the lg centroid!'''
        nNodes = len(nid_pos_affected)
        aoa = np.deg2rad(self.op.alpha)
        bank = np.deg2rad(self.bank)
        lPerNode = self.nlg*self.g0*self.MTOM/nNodes/2 #NOTE division by 2 - we assume 2-point landing
        self.L[0::pf3.DOF][nid_pos_affected] = -lPerNode*np.sin(aoa)*np.cos(bank)
        self.L[1::pf3.DOF][nid_pos_affected] = -lPerNode*np.sin(bank)
        self.L[2::pf3.DOF][nid_pos_affected] = lPerNode*np.cos(aoa)*np.cos(bank)


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
        n_points = node_indices.shape[0]
        nnz = 3 * self.nneighs * n_points

        Wr = np.zeros(nnz, dtype=int)
        Wc = np.zeros(nnz, dtype=int)
        Wv = np.zeros(nnz)

        k = 0
        for j in range(self.nneighs):
            for i, node_index in enumerate(node_indices[:, j]):
                base_row = pf3.DOF * ids_s[node_index]
                base_col = i * 3
                w = weights[i, j]

                Wr[k] = base_row + 0
                Wc[k] = base_col + 0
                Wv[k] = w
                k += 1

                Wr[k] = base_row + 1
                Wc[k] = base_col + 1
                Wv[k] = w
                k += 1

                Wr[k] = base_row + 2
                Wc[k] = base_col + 2
                Wv[k] = w
                k += 1

        # build sparse matrix
        W = ss.coo_matrix((Wv, (Wr, Wc)), shape=(self.N, 3 * n_points)).tocsr()

        # same check as before
        assert np.allclose(np.array(W.sum(axis=0)).ravel(), 1.)

        Fv = vlm.forces_geometry[vortex_valid].flatten()
        #re-scaling the aerodynamic load to match the specified load factor, done due to some load factors not being achievable w\o hlds
        Fv = ratio*Fv
        Fext = W @ Fv #external forces to be applied to the fem model

        assert np.isclose(Fext[0::pf3.DOF].sum(), Fv[0::3].sum())
        assert np.isclose(Fext[1::pf3.DOF].sum(), Fv[1::3].sum())
        assert np.isclose(Fext[2::pf3.DOF].sum(), Fv[2::3].sum())

        return ss.csc_matrix(W), Fext


    def _fem2aero(self, ncoords_s:nt.NDArray[np.float32], ids_s:nt.NDArray[np.int32]):
        NAIRFS = self.les.shape[0]
        W_u_to_p = np.zeros((NAIRFS*6, self.N))

        #leading edge points
        for i in range(NAIRFS):
            tree = ssp.cKDTree(ncoords_s)
            d, node_indices = tree.query(self.les[None, i], k=1)
            node_index = node_indices[0]
            weigh = 2

            W_u_to_p[i*3+0, pf3.DOF*ids_s[node_index]+0] += weigh
            W_u_to_p[i*3+1, pf3.DOF*ids_s[node_index]+1] += weigh
            W_u_to_p[i*3+2, pf3.DOF*ids_s[node_index]+2] += weigh

        #trailing edge points
        for i in range(NAIRFS):
            tree = ssp.cKDTree(ncoords_s)
            d, node_indices = tree.query(self.tes[None, i], k=1)
            node_index = node_indices[0]
            weigh = 2

            W_u_to_p[(NAIRFS+i)*3+0, pf3.DOF*ids_s[node_index]+0] += weigh
            W_u_to_p[(NAIRFS+i)*3+1, pf3.DOF*ids_s[node_index]+1] += weigh
            W_u_to_p[(NAIRFS+i)*3+2, pf3.DOF*ids_s[node_index]+2] += weigh
  
        return ss.csc_matrix(W_u_to_p)


    def _vlm(self, displs:ty.List[float]=None, return_lift=False, debug=False):
        #allowing for initial displacements for flutter. Yet, for static analysis we dont's need displacements
        if displs is None:
            displs = np.zeros(len(self.airfs)*2)

        #converting LE-TE displs into LE displs and twists
        rowlen = self.les.shape[0]
        ledispls = displs[:rowlen]
        #leshifts = displs[2*rowlen:3*rowlen]
        tedispls = displs[rowlen:2*rowlen]
        #teshifts = displs[3*rowlen:4*rowlen]
        # ptles = [self.les[i,:]+np.array([displs[2*rowlen+i], 0, displs[i]]) for i in range(len(ledispls))]
        ptles = [self.les[i,:]+np.array([0, 0, displs[i]]) for i in range(len(ledispls))]
        #cs = [c+tes-les for c, les, tes in zip(self._cs, leshifts, teshifts)]
        cs = self._cs
        crange = .55 #from .15 to .7 of chord
        twists = [np.arctan((led-ted)/c/crange) for led, ted, c in zip(ledispls, tedispls, cs)]

        airplane = asb.Airplane("E9X", xyz_ref=[0,0,0], wings = [
                                    asb.Wing(
                                        name="E9X_WING",
                                        symmetric=True,
                                        xsecs=[asb.WingXSec(
                                            xyz_le=[ptle[0], ptle[1], ptle[2]],
                                            chord=c,
                                            twist=np.rad2deg(twist), #asb has aoas and twists in degrees 
                                            airfoil=airf
                                        )for ptle, c, airf, twist in zip(ptles, cs, self.airfs, twists)],
                                    )
                                ])
        vlm = asb.VortexLatticeMethod(airplane, self.op, spanwise_resolution=self.bres, chordwise_resolution=self.cres)
        sol = vlm.run()
        forces = sol["F_g"]
        moments = sol["M_g"]
        lift = sol['L']

        # #compressibility corrections
        vlm.forces_geometry *= self.pg
        forces *= self.pg
        moments *= self.pg
        lift *= self.pg


        if debug: #printing ratio
            ratio = self.MTOM*self.g0*self.n/lift
            print(f"nominal load to calculated load: {ratio}")

        if return_lift:
            return airplane, vlm, forces, moments, lift
        else:
            return airplane, vlm, forces, moments


    def _karman_thiessen(self, cp:nt.NDArray[np.float64]):
        beta = np.sqrt(1-self.m2)
        return cp/(beta+self.m2/(1+beta)*cp/2)


    def _calc_dFv_dp(self, ymin:float, epsilon=.01, debug=False):
        rowlen = len(self.airfs)
        p = np.zeros(rowlen*2)
        airplane, vlm_, forces, moments = self._vlm(p, debug=debug)

        vortex_valid = vlm_.vortex_centers[:, 1] > ymin
        v = vortex_valid.sum()*3
        Fv = vlm_.forces_geometry[vortex_valid].flatten()

        dFv_dp = np.zeros((v, rowlen*6)) # NOTE remember to pass both heave and twist displacements
        for i in range(len(p)):
            p_DOF = (3*i+2) #if (i<2*rowlen) else (3*(i-2*rowlen))
            p2 = p.copy()
            p2[i] += epsilon

            plane2, vlm2, f2, M2 = self._vlm(p2, debug=debug)
            Fv2 = vlm2.forces_geometry[vortex_valid].flatten()
            deriv = (Fv2 - Fv)/epsilon
            dFv_dp[:, p_DOF] += deriv

        dFv_dp = ss.csc_matrix(dFv_dp)
        return vlm_, dFv_dp