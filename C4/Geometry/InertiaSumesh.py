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
        bat_masses_in_range:list[float] = list()
        s3 = np.sqrt(3)

        #1.1) battery connection point coordinates
        y_oub = scaffold[:, 1:-3, 1]
        y_inb = np.zeros(y_oub.shape)
        y_inb[1:, :] = y_oub[:-1, :]
        x_oub_fore = scaffold[:, 1:-3, 0]
        z_oub_fore = scaffold[:, 1:-3, 2]
        x_oub_peak = scaffold[:, 2:-2, 0]
        z_oub_peak = scaffold[:, 2:-2, 2]
        x_oub_rear = scaffold[:, 3:-1, 0]
        z_oub_rear = scaffold[:, 3:-1, 2]
        y_fus = scaffold[:, :, 1].min()
        dz_dy = (scaffold[0, 0, 2]-scaffold[1, 0, 2])/(scaffold[0, 0, 1]-scaffold[1, 0, 1])
        
        for yok, xof, zof, xok, zok, xor, zor, yia in zip(
            y_oub.flatten(), x_oub_fore.flatten(), z_oub_fore.flatten(), x_oub_peak.flatten(), z_oub_peak.flatten(),
            x_oub_rear.flatten(), z_oub_rear.flatten(), y_inb.flatten()
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
            assert A2 > 0
            assert A3 > 0
            A = h**2/s3-A1-2*A2-A3
            if A>0:
                bat_mass = rho_bat*A*(y_oub-y_inb-Delta_b)
                self.battery_masses.append(bat_mass)
                if yok > y_fus:
                    bat_masses_in_range.append(bat_mass)
                    #TODO: centroid, coordinate lists to be used 4 meshing (directly tuples), remove battery plotting
        
        #1.5) adjusting for the nominal battery mass
        self.tot_computed_bat_mass = sum(self.battery_masses)
        ratio_bat_mass = m_bat_nominal/self.tot_computed_bat_mass
        self.battery_masses = [m_bat_raw*ratio_bat_mass for m_bat_raw in self.battery_masses]
        bat_masses_in_range = [m_bat_raw*ratio_bat_mass for m_bat_raw in bat_masses_in_range]

        #1.6) battery mesh
        for yok, xof, zof, xok, zok, xor, zor, xia, yia, zia, centr, bat_mass in zip( #skipping the 1st row of bats that's inside the fuselage
            y_oub[1:, :].flatten(), x_oub_fore[1:, :].flatten(), z_oub_fore[1:, :].flatten(),
            x_oub_peak[1:, :].flatten(), z_oub_peak[1:, :].flatten(), x_oub_rear[1:, :].flatten(), z_oub_rear[1:, :].flatten(),
            x_oub_peak[:-1, :].flatten(), y_oub[:-1, :].flatten(), z_oub_peak[:-1, :].flatten(), 
            bat_centroids[y_oub.shape[1]:], self.battery_masses[y_oub.shape[1]:] #removing the 1st raw of batteries from the already flattened mass lists
        ): 
            if centr[1] > scaffold[:, :, 1].min(): #we don't add the in-fuselage batteries to the mesh
                #1.5.1) battery inertia
                self.eleTypes.append("bi")
                self.eleArgs.append([bat_mass])
                self.eleNodes.append([centr])
                #1.5.2) spring mesh - connection to all outboard points and the inboard rail
                self.eleTypes.extend(["ms"]*4)
                self.eleArgs.extend([[]]*4)
                self.eleNodes.append([(xof, yok, zof), centr])
                self.eleNodes.append([(xok, yok, zok), centr])
                self.eleNodes.append([(xor, yok, zor), centr])
                self.eleNodes.append([(xia, yia, zia), centr]) 

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


    def _LE_or_TE_inertia(self, nodes1:nt.NDArray[np.float64], nodes2:nt.NDArray[np.float64], totmass:float):
        dists = abs(nodes1[:, 2]-nodes2[:, 2]) #the points are end of a verical line
        masses = totmass*dists/dists.sum()/2 #half of the mass goes to the top, half goes to the bottom
        for i, m in enumerate(masses):
            self.eleTypes.extend(["ei"]*2)
            self.eleArgs.extend([[m]]*2)
            self.eleNodes.append([(nodes1[i, 0], nodes1[i, 1], nodes1[i, 2])])
            self.eleNodes.append([(nodes2[i, 0], nodes2[i, 1], nodes2[i, 2])])


    @staticmethod
    def _polygon_area_centroid_3d(points):
        """
        Compute area and centroid of a planar 3D polygon.
        Assumes points are ordered and planar.
        """
        n = len(points)

        # Compute normal using Newell's method
        normal = np.zeros(3)
        for i in range(n):
            p0 = points[i]
            p1 = points[(i + 1) % n]
            normal += np.cross(p0, p1)

        area = np.linalg.norm(normal) / 2.0
        normal = normal / np.linalg.norm(normal)

        # Project polygon to dominant plane for centroid computation
        ax = np.argmax(np.abs(normal))

        if ax == 0:      # project to yz
            coords = points[:, 1:]
        elif ax == 1:    # project to xz
            coords = points[:, [0, 2]]
        else:            # project to xy
            coords = points[:, :2]

        # 2D centroid formula
        A2 = 0.0
        C2 = np.zeros(2)

        for i in range(n):
            p0 = coords[i]
            p1 = coords[(i + 1) % n]
            cross = p0[0]*p1[1] - p1[0]*p0[1]
            A2 += cross
            C2 += (p0 + p1) * cross

        A2 *= 0.5
        C2 /= (6*A2)

        # Reconstruct 3D centroid
        centroid = np.zeros(3)
        if ax == 0:
            centroid[1:] = C2
            centroid[0] = points[:,0].mean()
        elif ax == 1:
            centroid[0] = C2[0]
            centroid[2] = C2[1]
            centroid[1] = points[:,1].mean()
        else:
            centroid[:2] = C2
            centroid[2] = points[:,2].mean()

        return abs(area), centroid


    @classmethod
    def _centroid_of_loft_analytical(cls, points):
        """
        Analytical centroid of a loft defined by 8x3 array.
        First 4 rows = top trapezoid
        Last 4 rows  = bottom trapezoid
        """

        if points.shape != (8, 3):
            raise ValueError("Input must be 8x3.")

        top = points[:4]
        bottom = points[4:]

        A1, c1 = cls._polygon_area_centroid_3d(top)
        A0, c0 = cls._polygon_area_centroid_3d(bottom)

        denom = A0 + 2*np.sqrt(A0*A1) + A1

        C = (
            A0 * c0
            + 2*np.sqrt(A0*A1) * (c0 + c1)/2
            + A1 * c1
        ) / denom

        return C


    @staticmethod
    def _plot_bat(points: np.ndarray):
        """
        Create a lofted solid between two trapezoids stored in an 8x3 array.

        Parameters
        ----------
        points : np.ndarray
            8x3 array of vertices.
            First 4 rows = trapezoid at larger y.
            Last 4 rows  = corresponding trapezoid at smaller y.
        plot : bool
            If True, plot the result.

        Returns
        -------
        mesh : pv.PolyData
            The resulting surface mesh.
        """

        if points.shape != (8, 3):
            raise ValueError("Input must be an 8x3 array.")

        # Build face list
        faces = []

        # Top trapezoid
        faces.extend([4, 0, 1, 2, 3])

        # Bottom trapezoid
        faces.extend([4, 4, 5, 6, 7])

        # Side faces
        for i in range(4):
            top0 = i
            top1 = (i + 1) % 4
            bot0 = i + 4
            bot1 = ((i + 1) % 4) + 4

            faces.extend([4, top0, top1, bot1, bot0])

        faces = np.array(faces)

        # Create mesh
        mesh = pv.PolyData(points, faces)

        return mesh


    def plot_bats(self, plotter:pv.Plotter):
        for vs in self.batteryVertices:
           plotter.add_mesh(self._plot_bat(vs), color="lightblue", show_edges=True)