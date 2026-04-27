import aerosandbox.numpy as np
import numpy.typing as nt
import pyvista as pv
import typing as ty
import pypardiso as ppd
import pyfe3d as pf3

from ..Pyfe3DModel import Pyfe3DModel
from .Mesher import Mesher
from ..LoadCase import LoadCase
from .joints_available import JointsAvailable

class InertiaSubmesh():
    def __init__(self, scaffold:nt.NDArray[np.float64], HYPERPARAMS:dict[str, float], MASSES:dict[str, float], c_at_y:ty.Callable[[float], float], eqpt_dict:dict[str, ty.Any], 
                 lcs:list[LoadCase], G0:float, MTOM:float, plot_joint_loading=False):
        self.eleTypes:list[str] = list()
        self.eleArgs:list[list[float]] = list()
        self.eleNodes:list[list[tuple[float]]] = list()

        self.battery_masses:list[float] = list()
        self.tot_computed_bat_mass:float = 0.
        self.batteryVertices:list[nt.NDArray[np.float64]] = list()

        delta = HYPERPARAMS["delta"]
        Hpcsq = HYPERPARAMS["(H/c)_sq"]
        Hpcaq = HYPERPARAMS["(H/c)_aq"]
        D = HYPERPARAMS["D"]
        Delta_b = HYPERPARAMS["Delta b"]

        rho_bat = MASSES["rho_bat"]
        mhi = MASSES["hi"]
        mLE = MASSES["LE"]
        mTE = MASSES["TE"]
        m_bat_nominal = MASSES["bi"]
        mmi = MASSES["mi"]
        mli = MASSES["li"]

        #1) battery packing
        bat_centroids:list[tuple[float]] = list()
        bat_inbconns:list[tuple[float]] = list()
        bat_peakconns:list[tuple[float]] = list()
        bat_foreconns:list[tuple[float]] = list()

        bat_masses_in_range:list[float] = list()
        s3 = np.sqrt(3)

        #1.1) battery connection point coordinates
        y_oub = scaffold[:, 1:-3, 1]
        y_inb = np.zeros(y_oub.shape)
        x_inb = np.zeros(y_oub.shape)
        z_inb = np.zeros(y_oub.shape)
        y_inb[1:, :] = y_oub[:-1, :]
        y_oub = y_oub.flatten()
        y_inb = y_inb.flatten()
        x_oub_fore = scaffold[:, 1:-3, 0].flatten()
        z_oub_fore = scaffold[:, 1:-3, 2].flatten()
        x_oub_peak = scaffold[:, 2:-2, 0].flatten()
        z_oub_peak = scaffold[:, 2:-2, 2].flatten()
        x_oub_rear = scaffold[:, 3:-1, 0].flatten()
        z_oub_rear = scaffold[:, 3:-1, 2].flatten()
        x_inb[1:, :] = scaffold[:-1, 2:-2, 0]
        x_inb = x_inb.flatten()
        z_inb[1:, :] = scaffold[:-1, 2:-2, 2]
        z_inb = z_inb.flatten()
        y_fus = scaffold[:, :, 1].min()
        
        for yok, xof, zof, xok, zok, xor, zor, yia, xia, zia in zip(
            y_oub, x_oub_fore, z_oub_fore, x_oub_peak, z_oub_peak,
            x_oub_rear, z_oub_rear, y_inb, x_inb, z_inb
        ):
            #obtaining battery area by subtracting the area of spacings
            h_fore = abs(zok-zof)
            h_rear = abs(zok-zor)
            h = min(h_fore, h_rear)
            a = 2*h/s3

            c = c_at_y(yok)
            delta_d = Hpcsq*c/2+delta+D
            delta_a = Hpcaq*c/2+delta
            delta_s = Hpcsq*c/2+delta
            
            A1 = delta_d**2/s3
            A2 = delta_a*(a-2*delta_d/s3)
            A3 = delta_s*(a-4*delta_a/s3-delta_s/s3)
            A = h**2/s3-A1-2*A2-A3

            #obtaining battery area by calculation of height and smaller side of the battery trapezoid
            a_prime = 2/s3*(delta_d-2*delta_a)
            h_prime = h-delta_d-delta_s
            A_prime = a_prime*h_prime+h_prime**2/s3
            assert np.isclose(A, A_prime)

            if (A>0) and (A2>0) and (A3>0): #only count in the mass of the batteries that can physically fit into their bays
                bat_mass = rho_bat*A*(yok-yia-Delta_b)
                self.battery_masses.append(bat_mass)
                if yok > y_fus: #only mesh batteries inboard from the fuselage plane
                    bat_masses_in_range.append(bat_mass)
                    
                    #centroid computation.
                    centr_oub_z = zok-2*h/3*np.sign(zok-zof)
                    centr_inb_z = zia+centr_oub_z-zok
                    bat_centroids.append(((xia+xok)/2, (yok+yia)/2, (centr_oub_z+centr_inb_z)/2))
                    bat_inbconns.append((xia, yia, zia))
                    bat_foreconns.append((xof, yok, zof))
                    bat_peakconns.append((xok, yok, zok))
        
        #1.5) adjusting for the nominal battery mass
        self.tot_computed_bat_mass = sum(self.battery_masses)
        ratio_bat_mass = m_bat_nominal/self.tot_computed_bat_mass
        self.battery_masses = [m_bat_raw*ratio_bat_mass for m_bat_raw in self.battery_masses]
        bat_masses_in_range = [m_bat_raw*ratio_bat_mass for m_bat_raw in bat_masses_in_range]

        #1.6) battery mesh
        for forec, peakc, inbc, centr, bat_mass in zip( 
            bat_foreconns, bat_peakconns, bat_inbconns, bat_centroids, bat_masses_in_range
        ): 
            #1.5.1) battery inertia
            self.eleTypes.append("bi")
            self.eleArgs.append([bat_mass])
            self.eleNodes.append([centr])
            #1.5.2) spring mesh - connection to all outboard points and the inboard rail
            self.eleTypes.extend(["ms"]*3)
            self.eleArgs.extend([[]]*3)
            self.eleNodes.append([forec, centr])
            self.eleNodes.append([peakc, centr])
            self.eleNodes.append([inbc, centr]) 

        #2) wingtip and equipment inertia
        #2.1) wingtip
        #creating hinge+wingtip centroid
        wingtip_pt = (scaffold[-1, 0, :]+scaffold[-1, -1, :])/2
        delta_y_hn = (scaffold[-1, -1, 0]-scaffold[-1, 0, 0])
        wingtip_pt[1] += delta_y_hn
        wingtip_pt[2] += delta_y_hn/2
        wingtip_pt = (wingtip_pt[0], wingtip_pt[1], wingtip_pt[2])
        #asigning the inertia element
        self.eleTypes.append("hi")
        self.eleArgs.append([mhi])
        self.eleNodes.append([wingtip_pt])
        #assigning the spring elements
        self.eleTypes.extend(["ms"]*3)
        self.eleArgs.extend([[]]*3)
        self.eleNodes.append([(scaffold[-1, 0, 0], scaffold[-1, 0, 1], scaffold[-1, 0, 2]), wingtip_pt])
        self.eleNodes.append([(scaffold[-1, 1, 0], scaffold[-1, 1, 1], scaffold[-1, 1, 2]), wingtip_pt])
        self.eleNodes.append([(scaffold[-1, -1, 0], scaffold[-1, -1, 1], scaffold[-1, -1, 2]), wingtip_pt])

        #2.2) LE-TE equipment
        nodes_LE_top = scaffold[:, 0, :]
        nodes_LE_bot = scaffold[:, 1, :]
        self._LE_or_TE_inertia(nodes_LE_top, nodes_LE_bot, mLE)
        nodes_TE_last = scaffold[:, -1, :]
        nodes_TE_semilast = scaffold[:, -2, :]
        self._LE_or_TE_inertia(nodes_TE_last, nodes_TE_semilast, mTE)

        #3) motor inertia
        assert len(eqpt_dict["motor_pts"]) == len(eqpt_dict['motor_is'])
        for mcoords, midx in zip(eqpt_dict["motor_pts"], eqpt_dict['motor_is']):
            self.eleTypes.extend(['mi', 'ms', 'ms', 'ms'])
            self.eleArgs.extend([[mmi], [], [], []])
            self.eleNodes.append([mcoords])
            self.eleNodes.append([(scaffold[midx-1, 1, 0], scaffold[midx-1, 1, 1], scaffold[midx-1, 1, 2]), mcoords])
            self.eleNodes.append([(scaffold[midx, 0, 0], scaffold[midx, 0, 1], scaffold[midx, 0, 2]), mcoords])
            self.eleNodes.append([(scaffold[midx, 1, 0], scaffold[midx, 1, 1], scaffold[midx, 1, 2]), mcoords])

        #4) landing gear inertia
        lgcoords = eqpt_dict["lg_pt"]
        assert len(eqpt_dict['lg_is']) == 2
        lgi_inb = eqpt_dict["lg_is"][0]
        lgi_oub = eqpt_dict["lg_is"][1]
        self.eleTypes.extend(['li', 'ms', 'ms', 'ms', 'li', 'ms', 'ms', 'ms'])
        self.eleArgs.extend([[mli/2], [], [], [], [mli/2], [], [], []])
        #inboard landing gear
        lgcoords_inb = (lgcoords[0], scaffold[lgi_inb, -1, 1], lgcoords[2])
        self.eleNodes.append([lgcoords_inb])
        self.eleNodes.append([(scaffold[lgi_inb, -1, 0], scaffold[lgi_inb, -1, 1], scaffold[lgi_inb, -1, 2]), lgcoords_inb])
        self.eleNodes.append([(scaffold[lgi_inb, -2, 0], scaffold[lgi_inb, -2, 1], scaffold[lgi_inb, -2, 2]), lgcoords_inb])
        self.eleNodes.append([(scaffold[lgi_oub, -2, 0], scaffold[lgi_oub, -2, 1], scaffold[lgi_oub, -2, 2]), lgcoords_inb])
        #outboard landing gear
        lgcoords_oub = (lgcoords[0], scaffold[lgi_oub, -1, 1], lgcoords[2])
        self.eleNodes.append([lgcoords_oub])
        self.eleNodes.append([(scaffold[lgi_oub, -1, 0], scaffold[lgi_oub, -1, 1], scaffold[lgi_oub, -1, 2]), lgcoords_oub])
        self.eleNodes.append([(scaffold[lgi_oub, -2, 0], scaffold[lgi_oub, -2, 1], scaffold[lgi_oub, -2, 2]), lgcoords_oub])
        self.eleNodes.append([(scaffold[lgi_inb, -2, 0], scaffold[lgi_inb, -2, 1], scaffold[lgi_inb, -2, 2]), lgcoords_oub])

        #5) Joint inertia
        self.tot_joint_mass = 0
        
        #running a simulation on how the loads on engines, lg, hinge and the batteries will translate into loads @ joints
        collisionDecimalPlaces = 8
        mesher = Mesher(collisionDecimalPlaces)
        for eT, eA, eN in zip(self.eleTypes, self.eleArgs, self.eleNodes):
            mesher.load_ele(eN, eT, eA)

        ncoords = np.array(mesher.nodes)

        #boundary condition  scaffold points are fixed
        to_tuple_entry = lambda _float:int(round(_float*10**collisionDecimalPlaces))
        to_tuple = lambda x, y, z: (to_tuple_entry(x), to_tuple_entry(y), to_tuple_entry(z))
        scaffold_flat = scaffold.reshape(scaffold.shape[0]*scaffold.shape[1], scaffold.shape[2])
        scaffold_hash = set([to_tuple(x, y, z) for x, y, z in zip(scaffold_flat[:, 0], scaffold_flat[:, 1], scaffold_flat[:, 2])])
        boundary = lambda x,y,z:tuple([to_tuple(x, y, z) in scaffold_hash]*pf3.DOF)

        model = Pyfe3DModel(np.array(mesher.nodes), boundary)

        #populating the model inertia
        inertia_vals:list[float] = list() 
        for eleType, eleArg, eleNodePoses in zip(self.eleTypes, self.eleArgs, mesher.eleNodePoses):
            if eleType[1] == 'i':
                inertia_vals.append(eleArg[0])
                model.load_inertia(eleNodePoses[0])
            elif eleType == 'ms':
                model.load_spring(*eleNodePoses, 1e6, 0., 0., 1e6, 0., 0., 0., 1., 0.)

        model.KC0_M_update([], [], [], [], inertia_vals)

        #control sum of the DOF remaining free
        ctrl_sum = pf3.DOF*(len(eqpt_dict["motor_is"])+len(eqpt_dict["lg_is"])+1+len(bat_centroids))
        assert np.count_nonzero(model.bu) == ctrl_sum, f"dofs: {np.count_nonzero(model.bu)}, ctrl sum: {ctrl_sum}, N:{model.N}"
        
        #load application
        fints:list[nt.NDArray[np.float_]] = list()
        fexts:list[nt.NDArray[np.float_]] = list()
        for lcinfo in lcs:
            lc = LoadCase(lcinfo["n"], MTOM, model.N, G0, lcinfo["Ttot"], lcinfo["op"], np.array([]), np.array([]), np.array([]), nlg=lcinfo["nlg"])
            lc.apply_thrust(mesher.get_submesh('mi')[0])
            lc.apply_landing(mesher.get_submesh('li')[0])
            lc.update_weight(model.M)
            fext = lc.loadstack()
            fu = fext[model.bu]
            uu = ppd.spsolve(model.KC0uu, fu)
            u = np.zeros_like(fext)
            u[model.bu] = uu

            fint = np.zeros_like(fext)
            for spring in model.springs:
                spring.update_probe_ue(u)
                spring.update_fint(fint)
            
            fints.append(fint)
            fexts.append(fext)

        #create fint_envelope fint at joint accessible by coordinates
        self.rjperc = 0.
        fint_envelope = np.vstack(np.abs(fints)).max(axis=0)
        fint_dict:dict[tuple[float], tuple[float]] = dict()
        for i, x, y, z in zip(range(len(model.x)), model.x, model.y, model.z):
            fint_dict[to_tuple(x, y, z)] = (fint_envelope[pf3.DOF*i+0], fint_envelope[pf3.DOF*i+1], fint_envelope[pf3.DOF*i+2])

        #translate to joint shear and normal forces
        for x, y, z in zip(scaffold_flat[:, 0], scaffold_flat[:, 1], scaffold_flat[:, 2]):
            if to_tuple(x, y, z) in fint_dict: #equipment is not attached to every scaffold point out there
                Vx, Vy, Nz = fint_dict[to_tuple(x, y, z)]
                V = np.sqrt(Vx**2+Vy**2)

                lj, mj = JointsAvailable.size_joint(Nz, V, Hpcsq)
                self.rjperc = max(lj/c_at_y(y), self.rjperc)
                self.tot_joint_mass += mj
                
                #joint element creation
                self.eleTypes.append('ji')
                self.eleArgs.append([mj])
                self.eleNodes.append([(x, y, z)])

        self.rjperc /= 2 #accounting for the fact that rj = .5 lj
    
        #6) Consistency check
        assert len(self.eleTypes) == len(self.eleArgs) == len(self.eleNodes)
        #motors, landing gear, wingtip and battery points are the added nodes in this submesh, alls else should coincide with the structural submesh.
        self.expected_node_count = len(eqpt_dict["motor_is"])+len(eqpt_dict["lg_is"])+1+len(bat_centroids)

        #7) plotting if so requested
        if plot_joint_loading:
            for fvect in fexts+fints+[fint_envelope]:
                self._plot_force_vector(model.ncoords, fvect)


    def _LE_or_TE_inertia(self, nodes1:nt.NDArray[np.float64], nodes2:nt.NDArray[np.float64], totmass:float):
        dists = abs(nodes1[:, 2]-nodes2[:, 2]) #the points are end of a verical line
        masses = totmass*dists/dists.sum()/2 #half of the mass goes to the top, half goes to the bottom
        for i, m in enumerate(masses):
            self.eleTypes.extend(["ei"]*2)
            self.eleArgs.extend([[m]]*2)
            self.eleNodes.append([(nodes1[i, 0], nodes1[i, 1], nodes1[i, 2])])
            self.eleNodes.append([(nodes2[i, 0], nodes2[i, 1], nodes2[i, 2])])

    def _plot_force_vector(self, ncoords:nt.NDArray[np.float64], force_vec:nt.NDArray[np.float64]):
        fx = force_vec[0::pf3.DOF]
        fy = force_vec[1::pf3.DOF]
        fz = force_vec[2::pf3.DOF]

        vectors = np.column_stack((fx, fy, fz))  # shape (n_points, 3)

        # compute magnitudes
        magnitudes = np.linalg.norm(vectors, axis=1)

        # avoid division by zero
        nonzero = magnitudes > 0

        unit_vectors = np.zeros_like(vectors)
        unit_vectors[nonzero] = vectors[nonzero] / magnitudes[nonzero][:, None]

        mesh = pv.PolyData(ncoords)
        mesh["vectors"] = unit_vectors      # direction (normalized)
        mesh["magnitude"] = magnitudes      # scalar for coloring

        arrows = mesh.glyph(
            orient="vectors",
            scale=False,        # ensures all arrows same size
            factor=0.2          # adjust arrow length globally
        )

        plotter = pv.Plotter()
        plotter.add_mesh(arrows, scalars="magnitude", cmap="viridis")
        plotter.show()