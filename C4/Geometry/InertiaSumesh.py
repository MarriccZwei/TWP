import aerosandbox.numpy as np
import numpy.typing as nt
import pyvista as pv
import typing as ty

class InertiaSubmesh():
    def __init__(self, scaffold:nt.NDArray[np.float64], HYPERPARAMS:dict[str, float], MASSES:dict[str, float], c_at_y:ty.Callable[[float], float]):
        self.eleTypes:list[str] = list()
        self.eleArgs:list[list[float]] = list()
        self.eleNodes:list[list[tuple[float]]] = list()

        self.battery_masses:list[float] = list()
        self.battery_centroids:list[tuple[float]] = list()
        self.batteryVertices:list[nt.NDArray[np.float64]] = list()

        delta = HYPERPARAMS["delta"]
        Hpcsk = HYPERPARAMS["(H/c)_sk"]
        Hpcas = HYPERPARAMS["(H/c)_as"]
        D = HYPERPARAMS["D"]
        d = HYPERPARAMS["d"]
        Delta_b = HYPERPARAMS["Delta b"]

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

        #offset on inboard coordinates for Delta_b
        yscf_inb = np.vstack((np.zeros((1, scaffold.shape[1])), yscf_oub[:-1, :]))+Delta_b
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
        
        #1.3)per-coordinate set battery geometry computation
        for xof, yof, zof, xif, yif, zif, xok, yok, zok, xik, yik, zik, xor, yor, zor, xir, yir, zir in zip(
            x_oub_fore, y_oub_fore, z_oub_fore, x_inb_fore, y_inb_fore, z_inb_fore, 
            x_oub_peak, y_oub_peak, z_oub_peak, x_inb_peak, y_inb_peak, z_inb_peak,
            x_oub_rear, y_oub_rear, z_oub_rear, x_inb_rear, y_inb_rear, z_inb_rear):
            
            #intermediate expressions
            batdir = np.sign(zok-zor)
            c = c_at_y(yok)
            Hsk = Hpcsk*c
            Has = Hpcas*c
            delta_zok = -batdir*(delta+D+max(Hsk/2, Has))
            abs_delta_xok = abs(delta_zok)/np.sqrt(3)-Has-2*delta
            assert abs_delta_xok > 0, abs_delta_xok
            abs_delta_xorf = (Hsk/2+Has+3*delta)/np.sqrt(3)

            #outboard coordinates
            zok_n = zok+delta_zok
            xokf_n = xok-abs_delta_xok
            xokr_n = xok+abs_delta_xok
            zofr_n = abs(max(batdir*zof, batdir*zor))+batdir*(Hsk/2+delta)
            xof_n = xof+abs_delta_xorf #to correct: must be symmetric
            xor_n = xor-abs_delta_xorf #to correct: must be symmetric

            #inboard coordinates
            zik_n = zik+delta_zok
            xikf_n = xik-abs_delta_xok
            xikr_n = xik+abs_delta_xok
            zifr_n = zik_n+zofr_n-zok_n
            xif_n = xik+xof_n-xok
            xir_n = xik+xor_n-xok

            if (zok_n-zofr_n)*batdir > 0: #only make batteries if there is enough space
                #battery point assembly direction 0-of, 1-okf, 2-okr, 3-or, 4-if, 5-ikf, 6-ikr, 7-ir
                vertices = np.array([
                    [xof_n, yof, zofr_n], [xokf_n, yok, zok_n], [xokr_n, yok, zok_n], [xor_n, yof, zofr_n],
                    [xif_n, yif, zifr_n], [xikf_n, yik, zik_n], [xikr_n, yik, zik_n], [xir_n, yif, zifr_n]
                ])

                self.batteryVertices.append(vertices)
            else:
                print(f"size z: {zok_n}, {zofr_n}")


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