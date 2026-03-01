import aerosandbox.numpy as np
import numpy.typing as nt
import pyvista as pv
import typing as ty

class InertiaSubmesh():
    def __init__(self, scaffold:nt.NDArray[np.float64], HYPERPARAMS:dict[str, float], MASSES:dict[str, float], c_at_y:ty.Callable[[float], float], eqpt_dict:dict[str, ty.Any]):
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
        bat_rearconns:list[tuple[float]] = list()

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
            h_fore = abs(zok-zof)
            h_rear = abs(zok-zor)
            h = h_fore if (h_fore<h_rear) else h_rear
            a = 2*h/s3

            c = c_at_y(yok)
            delta_d = Hpcsq*c/2+delta+D
            delta_a = Hpcaq*c/2+delta
            delta_s = Hpcsq*c/2+delta
            
            A1 = delta_d**2/s3
            A2 = delta_a*(a-2*delta_d/s3)
            A3 = delta_s*(a-4*delta_a/s3-delta_s/s3)
            A = h**2/s3-A1-2*A2-A3
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
                    bat_rearconns.append((xor, yok, zor))
                    bat_peakconns.append((xok, yok, zok))
        
        #1.5) adjusting for the nominal battery mass
        self.tot_computed_bat_mass = sum(self.battery_masses)
        ratio_bat_mass = m_bat_nominal/self.tot_computed_bat_mass
        self.battery_masses = [m_bat_raw*ratio_bat_mass for m_bat_raw in self.battery_masses]
        bat_masses_in_range = [m_bat_raw*ratio_bat_mass for m_bat_raw in bat_masses_in_range]

        #1.6) battery mesh
        for forec, peakc, rearc, inbc, centr, bat_mass in zip( 
            bat_foreconns, bat_peakconns, bat_rearconns, bat_inbconns, bat_centroids, bat_masses_in_range
        ): 
            #1.5.1) battery inertia
            self.eleTypes.append("bi")
            self.eleArgs.append([bat_mass])
            self.eleNodes.append([centr])
            #1.5.2) spring mesh - connection to all outboard points and the inboard rail
            self.eleTypes.extend(["ms"]*4)
            self.eleArgs.extend([[]]*4)
            self.eleNodes.append([forec, centr])
            self.eleNodes.append([peakc, centr])
            self.eleNodes.append([rearc, centr])
            self.eleNodes.append([inbc, centr]) 

        #2) hinge and equipment inertia
        #2.1) hinge
        hnPts = scaffold[-1, ::2, :]
        nhnpts = hnPts.shape[0]
        mhi_per = mhi/nhnpts
        for i in range(nhnpts):
            self.eleTypes.append("hi")
            self.eleArgs.append([mhi_per])
            self.eleNodes.append([(hnPts[i, 0], hnPts[i, 1], hnPts[i, 2])])

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
            self.eleTypes.extend(['mi', 'ms', 'ms', 'ms', 'ms'])
            self.eleArgs.extend([[mmi], [], [], [], []])
            self.eleNodes.append([mcoords])
            self.eleNodes.append([(scaffold[midx-1, 1, 0], scaffold[midx-1, 1, 1], scaffold[midx-1, 1, 2]), mcoords])
            self.eleNodes.append([(scaffold[midx, 0, 0], scaffold[midx, 0, 1], scaffold[midx, 0, 2]), mcoords])
            self.eleNodes.append([(scaffold[midx, 1, 0], scaffold[midx, 1, 1], scaffold[midx, 1, 2]), mcoords])
            self.eleNodes.append([(scaffold[midx, 2, 0], scaffold[midx, 2, 1], scaffold[midx, 2, 2]), mcoords])

        #4) landing gear inertia
        lgcoords = eqpt_dict["lg_pt"]
        assert len(eqpt_dict['lg_is']) == 2
        lgi_inb = eqpt_dict["lg_is"][0]
        lgi_oub = eqpt_dict["lg_is"][1]
        self.eleTypes.extend(['li', 'ms', 'ms', 'ms', 'ms'])
        self.eleArgs.extend([[mli], [], [], [], []])
        self.eleNodes.append([lgcoords])
        self.eleNodes.append([(scaffold[lgi_inb, -1, 0], scaffold[lgi_inb, -1, 1], scaffold[lgi_inb, -1, 2]), lgcoords])
        self.eleNodes.append([(scaffold[lgi_inb, -2, 0], scaffold[lgi_inb, -2, 1], scaffold[lgi_inb, -2, 2]), lgcoords])
        self.eleNodes.append([(scaffold[lgi_oub, -1, 0], scaffold[lgi_oub, -1, 1], scaffold[lgi_oub, -1, 2]), lgcoords])
        self.eleNodes.append([(scaffold[lgi_oub, -2, 0], scaffold[lgi_oub, -2, 1], scaffold[lgi_oub, -2, 2]), lgcoords])
    
        #5) Consistency check
        assert len(self.eleTypes) == len(self.eleArgs) == len(self.eleNodes)
        #motors, landing gear and battery points are the added nodes in this submesh, alls else should coincide with the structural submesh.
        self.expected_node_count = len(eqpt_dict["motor_is"])+1+len(bat_centroids)


    def _LE_or_TE_inertia(self, nodes1:nt.NDArray[np.float64], nodes2:nt.NDArray[np.float64], totmass:float):
        dists = abs(nodes1[:, 2]-nodes2[:, 2]) #the points are end of a verical line
        masses = totmass*dists/dists.sum()/2 #half of the mass goes to the top, half goes to the bottom
        for i, m in enumerate(masses):
            self.eleTypes.extend(["ei"]*2)
            self.eleArgs.extend([[m]]*2)
            self.eleNodes.append([(nodes1[i, 0], nodes1[i, 1], nodes1[i, 2])])
            self.eleNodes.append([(nodes2[i, 0], nodes2[i, 1], nodes2[i, 2])])