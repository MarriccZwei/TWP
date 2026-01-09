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
        self.g0 = g0 #ravitational acceleration
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
        #self.KA = np.zeros((N, N)) #here be the aerodynamic matrix


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
        print(ratio)
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


    def _fem2aero(self, p:nt.NDArray[np.float32], ncoords_s:nt.NDArray[np.float32], ids_s:nt.NDArray[np.int32],
                number_of_neighbors=2):
        number_of_neighbors = self.nneighs#NOTE: remove when testing done
        heave_displs =  p[:len(self.les)]
        leds = [self.les[i, :]+np.array([0.,0.,heave_displs[i]]) for i in range(self.les.shape[0])]
        #ledts = leds+[np.array([0., 0., 0.])]*(len(self.les)-1) #adding non-displaced points 4 twist

        tree = ssp.cKDTree(ncoords_s)
        d, node_indices = tree.query(leds, k=number_of_neighbors)
        print(node_indices.shape, p.shape, d.shape)
        power = 2
        weights2 = (1/d**power)/((1/d**power).sum(axis=1)[:, None])
        assert np.allclose(weights2.sum(axis=1), 1)

        W_u_to_p = np.zeros((len(p)*3, self.N))

        for j in range(number_of_neighbors):
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


    def _calc_dFv_dp(self, ymin:float, displs:ty.List[float]=None, return_sol=False, epsilon=.01):
        print("CalcdFvdP")
        if displs is None:
            displs = np.zeros(len(self.airfs)*2)
        airplane, vlm_, forces, moments = self._vlm(displs, return_sol)
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

                plane2, vlm2, f2, M2 = self._vlm(p2, return_sol)
                Fv2 = vlm2.forces_geometry[vortex_valid].flatten()
                deriv = (Fv2 - Fv)/epsilon
                dFv_dp[:, p_DOF] += deriv
                print(np.max(deriv))

        dFv_dp = ss.csc_matrix(dFv_dp)
        return vlm_, dFv_dp


    # def compute_KA(self, coords_affected:nt.NDArray[np.float64]):
    #     span = 42 # [m]
    #     sweep_rad = np.deg2rad(7)
    #     x_root = 0. # [m]
    #     z_root = 0.

    #     airfoil_chords = np.linspace(5., 2., 9)
    #     airfoil_ys = np.linspace(0, span/2, 9)
    #     airfoil_xs = x_root + airfoil_ys*np.tan(sweep_rad)

    #     var_hd0 = 0
    #     var_hd1 = 0
    #     var_hd2 = 0
    #     var_hd3 = 0
    #     var_hd4 = 0
    #     var_hd5 = 0
    #     var_hd6 = 0
    #     var_hd7 = 0
    #     var_hd8 = 0
    #     var_twist0 = 5.2
    #     var_twist1 = 5.2
    #     var_twist2 = 5.2
    #     var_twist3 = 5.2
    #     var_twist4 = 5.2
    #     var_twist5 = 5.2
    #     var_twist6 = 5.2
    #     var_twist7 = 5.2
    #     var_twist8 = 5.2

    #     p = np.array([
    #     var_hd0,
    #     var_hd1,
    #     var_hd2,
    #     var_hd3,
    #     var_hd4,
    #     var_hd5,
    #     var_hd6,
    #     var_hd7,
    #     var_hd8,
    #     var_twist0,
    #     var_twist1,
    #     var_twist2,
    #     var_twist3,
    #     var_twist4,
    #     var_twist5,
    #     var_twist6,
    #     var_twist7,
    #     var_twist8])

    #     def make_wing(p):
    #         heave_displs = p[:9]
    #         twists = p[9:]
    #         xsecs = []
    #         for i in range(9):
    #             airfoil = asb.Airfoil(f'naca24{18-i}')
    #             xsec = asb.WingXSec(
    #                 xyz_le=[airfoil_xs[i], airfoil_ys[i], z_root+heave_displs[i]],
    #                 chord=airfoil_chords[i],
    #                 twist=twists[i],
    #                 airfoil=airfoil,
    #             )
    #             xsecs.append(xsec)
            
    #         wing = asb.Wing(
    #                     name="E9X-Wing",
    #                     symmetric=True,  # Should this wing be mirrored across the XZ plane?
    #                     xsecs=xsecs)
    #         return wing

    #     wing = make_wing(p)
    #     wing.draw()

    #     vlm = asb.VortexLatticeMethod(
    #         airplane=asb.Airplane(
    #             name="E9X",
    #             xyz_ref=[0, 0, 0],
    #             wings=[wing],
    #         ),
    #         op_point=self.op
    #     )
    #     aero = vlm.run()  # Returns a dictionary
    #     for k, v in aero.items():
    #         print(f"{k.rjust(4)} : {v}")
    #     vlm.draw()

    #     vortex_valid = vlm.vortex_centers[:, 1] > coords_affected[:, 1].min() # region represented by the FE model

    #     tree = ssp.cKDTree(coords_affected)
    #     d, node_indices = tree.query(vlm.vortex_centers[vortex_valid], k=self.nneighs)
    #     power = 2
    #     weights = (1/d**power)/((1/d**power).sum(axis=1)[:, None])
    #     assert np.allclose(weights.sum(axis=1), 1)
    #     print('distances', d, d.min(), d.max())

    #     v = vortex_valid.sum()*3
    #     W_v_to_u = np.zeros((self.N, v))
    #     print(W_v_to_u.shape, node_indices.shape, weights.shape)

    #     for j in range(self.nneighs):
    #         for i, node_index in enumerate(node_indices[:, j]):
    #             W_v_to_u[pf3.DOF*node_index + 0, i*3 + 0] += weights[i, j]
    #             W_v_to_u[pf3.DOF*node_index + 1, i*3 + 1] += weights[i, j]
    #             W_v_to_u[pf3.DOF*node_index + 2, i*3 + 2] += weights[i, j]

    #     assert np.allclose(W_v_to_u.sum(axis=0), 1.)
    #     W_v_to_u = ss.csc_matrix(W_v_to_u)

    #     print(vlm.forces_geometry[vortex_valid].shape)
    #     print(vlm.forces_geometry[:3])
    #     Fv = vlm.forces_geometry[vortex_valid].flatten()
    #     print(Fv[:9])
    #     Fext = W_v_to_u @ Fv
    #     print(Fext.shape)

    #     number_of_neighbors2 = 2

    #     heave_displs = p[:9]
    #     # twists = p[4:] NOTE ignoring twists for now
    #     airfoil_coords = np.vstack((airfoil_xs, airfoil_ys, z_root+heave_displs)).T
    #     print(airfoil_coords)
    #     tree = ssp.cKDTree(coords_affected)
    #     d, node_indices = tree.query(airfoil_coords, k=number_of_neighbors2)
    #     power = 2
    #     weights2 = (1/d**power)/((1/d**power).sum(axis=1)[:, None])
    #     assert np.allclose(weights2.sum(axis=1), 1)
    #     print('distances', d, d.min(), d.max())

    #     W_u_to_p = np.zeros((len(heave_displs)*3, self.N))
    #     print(W_u_to_p.shape, node_indices.shape, weights2.shape)

    #     for j in range(number_of_neighbors2):
    #         for i, node_index in enumerate(node_indices):
    #             W_u_to_p[i*3 + 0, pf3.DOF*node_index + 0] = weights2[i, j]
    #             W_u_to_p[i*3 + 1, pf3.DOF*node_index + 1] = weights2[i, j]
    #             W_u_to_p[i*3 + 2, pf3.DOF*node_index + 2] = weights2[i, j]

    #     W_u_to_p = ss.csc_matrix(W_u_to_p)

    #     def calc_dFv_dp(velocity):
    #         epsilon = 0.01
    #         dFv_dp = np.zeros((v, len(heave_displs)*3))
    #         wing = make_wing(p)
    #         vlm = asb.VortexLatticeMethod(
    #             airplane=asb.Airplane(
    #                 name="AIAA dummy plane",
    #                 xyz_ref=[0, 0, 0],
    #                 wings=[wing],
    #             ),
    #             op_point=asb.OperatingPoint(
    #                 velocity=velocity,  # m/s
    #                 alpha=self.op.alpha,  # degree
    #             )
    #         )
    #         vlm.run()
    #         for i in range(4):
    #             p_DOF = 3*(i + 1) + 2 # heave DOF starting at second airfoil
    #             p2 = p.copy()
    #             p2[i+1] += epsilon
    #             wing2 = make_wing(p2)
    #             vlm2 = asb.VortexLatticeMethod(
    #                 airplane=asb.Airplane(
    #                     name="AIAA dummy plane",
    #                     xyz_ref=[0, 0, 0],
    #                     wings=[wing2],
    #                 ),
    #                 op_point=asb.OperatingPoint(
    #                     velocity=velocity,  # m/s
    #                     alpha=self.op.alpha,  # degree
    #                 )
    #             )
    #             vlm2.run()
    #             Fv2 = vlm2.forces_geometry[vortex_valid].flatten()
    #             dFv_dp[:, p_DOF] += (Fv2 - Fv)/epsilon
    #         dFv_dp = ss.csc_matrix(dFv_dp)
    #         return vlm, dFv_dp
        
    #     vlm, dFv_dp = calc_dFv_dp(velocity=self.op.velocity)
    #     KA = W_v_to_u @ dFv_dp @ W_u_to_p
    #     print(KA.shape)
    #     self.KA = KA