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
        Hpcsk = HYPERPARAMS["(H/c)_sq"]
        Hpcas = HYPERPARAMS["(H/c)_aq"]
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
        #1.1) determination of the hypotetical root coordinates
        pre_root_x = scaffold[0, :, 0]
        pre_root_y = scaffold[0, 0, 1]
        pre_root_z = scaffold[0, :, 2]
        pre2_root_x = scaffold[1, :, 0]
        pre2_root_y = scaffold[1, 0, 1]
        pre2_root_z = scaffold[1, :, 2]
        delta_y = pre2_root_y-pre_root_y

        #b = f(y)-a*y from linear function; shape to match the scaffold shape
        a_x = (pre2_root_x-pre_root_x)/delta_y
        a_z = (pre2_root_z-pre_root_z)/delta_y
        b_x = (pre_root_x-a_x*pre_root_y)
        b_z = (pre_root_z-a_z*pre_root_y)
        
        #1.2) outboard and inboard battery scaffold coordinates
        xscf_oub = scaffold[:, :, 0]
        yscf_oub = scaffold[:, :, 1]
        zscf_oub = scaffold[:, :, 2]

        #inboard attachment points for the batteries
        yscf_inb_no_delta = np.vstack((np.zeros((1, scaffold.shape[1])), yscf_oub[:-1, :]))
        xscf_inb_no_delta = a_x[None, :]*yscf_inb_no_delta+b_x[None, :]
        zscf_inb_no_delta = a_z[None, :]*yscf_inb_no_delta+b_z[None, :]

        #offset on inboard coordinates for Delta_b
        yscf_inb = yscf_inb_no_delta+Delta_b
        xscf_inb = a_x[None, :]*yscf_inb+b_x[None, :]
        zscf_inb = a_z[None, :]*yscf_inb+b_z[None, :]

        x_oub_fore = xscf_oub[:, 1:-3].flatten()
        y_oub_fore = yscf_oub[:, 1:-3].flatten()
        z_oub_fore = zscf_oub[:, 1:-3].flatten()
        x_inb_fore = xscf_inb[:, 1:-3].flatten()
        y_inb_fore = yscf_inb[:, 1:-3].flatten()
        z_inb_fore = zscf_inb[:, 1:-3].flatten()

        x_oub_peak = xscf_oub[:, 2:-2].flatten()
        y_oub_peak = yscf_oub[:, 2:-2].flatten()
        z_oub_peak = zscf_oub[:, 2:-2].flatten()
        x_inb_peak = xscf_inb[:, 2:-2].flatten()
        y_inb_peak = yscf_inb[:, 2:-2].flatten()
        z_inb_peak = zscf_inb[:, 2:-2].flatten()

        x_oub_rear = xscf_oub[:, 3:-1].flatten()
        y_oub_rear = yscf_oub[:, 3:-1].flatten()
        z_oub_rear = zscf_oub[:, 3:-1].flatten()
        x_inb_rear = xscf_inb[:, 3:-1].flatten()
        y_inb_rear = yscf_inb[:, 3:-1].flatten()
        z_inb_rear = zscf_inb[:, 3:-1].flatten()

        x_inb_peak_scaffold = xscf_inb_no_delta[:, 2:-2].flatten()
        y_inb_peak_scaffold = yscf_inb_no_delta[:, 2:-2].flatten()
        z_inb_peak_scaffold = zscf_inb_no_delta[:, 2:-2].flatten()
        
        #1.3)per-coordinate set battery geometry computation
        for xof, yof, zof, xif, yif, zif, xok, yok, zok, xik, yik, zik, xor, yor, zor, xir, yir, zir, xia, yia, zia in zip(
            x_oub_fore, y_oub_fore, z_oub_fore, x_inb_fore, y_inb_fore, z_inb_fore, 
            x_oub_peak, y_oub_peak, z_oub_peak, x_inb_peak, y_inb_peak, z_inb_peak,
            x_oub_rear, y_oub_rear, z_oub_rear, x_inb_rear, y_inb_rear, z_inb_rear,
            x_inb_peak_scaffold, y_inb_peak_scaffold, z_inb_peak_scaffold):
            
            #intermediate expressions
            batdir = np.sign(zok-zor)
            c = c_at_y(yok)
            Hsk = Hpcsk*c
            Has = Hpcas*c
            delta_zok = -batdir*(delta+D+max(Hsk/2, Has))
            abs_delta_xok = abs(delta_zok)/np.sqrt(3)-Has-2*delta
            assert abs_delta_xok > 0, abs_delta_xok

            #outboard coordinates
            zok_n = zok+delta_zok
            xokf_n = xok-abs_delta_xok
            xokr_n = xok+abs_delta_xok
            zofr_n = abs(max(batdir*zof, batdir*zor))+batdir*(Hsk/2+delta)
            abs_delta_xorf = abs(zofr_n-zok_n)/np.sqrt(3)
            xof_n = xokf_n-abs_delta_xorf #to correct: must be symmetric
            xor_n = xokr_n+abs_delta_xorf #to correct: must be symmetric

            #inboard coordinates
            zik_n = zik+delta_zok
            xikf_n = xik-abs_delta_xok
            xikr_n = xik+abs_delta_xok
            zifr_n = zik_n+zofr_n-zok_n
            xif_n = xikf_n-abs_delta_xorf
            xir_n = xikr_n+abs_delta_xorf

            #1.4) battery creation
            if (zok_n-zofr_n)*batdir > 0: #only make batteries if there is enough space
                #battery point assembly direction 0-of, 1-okf, 2-okr, 3-or, 4-if, 5-ikf, 6-ikr, 7-ir
                vertices = np.array([
                    [xof_n, yof, zofr_n], [xokf_n, yok, zok_n], [xokr_n, yok, zok_n], [xor_n, yof, zofr_n],
                    [xif_n, yif, zifr_n], [xikf_n, yik, zik_n], [xikr_n, yik, zik_n], [xir_n, yir, zifr_n]
                ])
                self.batteryVertices.append(vertices)
                bat_mass = rho_bat*(xokr_n-xokf_n+xor_n-xof_n)*(yok-yik)*abs(zofr_n-zok_n)/2
                self.battery_masses.append(bat_mass)

                #1.5) battery mesh
                x_centr_bat = tuple(self._centroid_of_loft_analytical(vertices))
                if x_centr_bat[1] > scaffold[:, :, 1].min(): #we don't add the in-fuselage batteries to the mesh
                    #1.5.1) battery inertia
                    self.eleTypes.append("bi")
                    self.eleArgs.append([bat_mass])
                    self.eleNodes.append([x_centr_bat])
                    #1.5.2) spring mesh - connection to all outboard points and the inboard rail
                    self.eleTypes.extend(["ms"]*4)
                    self.eleArgs.extend([[]]*4)
                    self.eleNodes.append([(xof, yof, zof), x_centr_bat])
                    self.eleNodes.append([(xok, yok, zok), x_centr_bat])
                    self.eleNodes.append([(xor, yor, zor), x_centr_bat])
                    self.eleNodes.append([(xia, yia, zia), x_centr_bat]) 

            #1.5) adjusting for the nominal battery mass
            self.tot_computed_bat_mass = sum(self.battery_masses)
            ratio_bat_mass = m_bat_nominal/self.tot_computed_bat_mass
            self.battery_masses = [m_bat_raw*ratio_bat_mass for m_bat_raw in self.battery_masses]

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