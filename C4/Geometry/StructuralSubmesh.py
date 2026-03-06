from .ElysianWing import ElysianWing

import aerosandbox.numpy as np
import numpy.typing as nt

class StructuralSubmesh():
    def __init__(self, wing:ElysianWing, HYPERPARAMS:dict[str, float], n:int):
        self.eleTypes:list[str] = list()
        self.eleArgs:list[list[float]] = list()
        self.eleNodes:list[list[tuple[float]]] = list()
        assert n >= 4 #to allow for joint representation
        self.n = n
        self.wing = wing
        self.HYPERPARAMS = HYPERPARAMS

        #1) computation of number of y coordinates per space between two scaffold ys
        # Differences in y between adjacent stations
        y_difference = wing.scaffold[1:, 0, 1] - wing.scaffold[:-1, 0, 1]

        # x differences
        x_difference_oub = wing.scaffold[1:, -1, 0] - wing.scaffold[1:, 0, 0]
        x_difference_inb = wing.scaffold[:-1, -1, 0] - wing.scaffold[:-1, 0, 0]

        x_difference_avg = (x_difference_inb + x_difference_oub) / 2

        n_x_tot = (n-1) * (wing.scaffold[:, ::2, 0].shape[1]-1) + 1 #not to double count the nodes

        ys_per_bay = np.ceil(n_x_tot * y_difference / x_difference_avg).astype(np.int8)

        eq_dict = self.wing.large_equipment_summary()
        furthest_motor_i = eq_dict["motor_is"][-1]
        furthest_lg_i = eq_dict["lg_is"][-1]

        #2)bay-by bay mesh creation
        for i, ny in enumerate(ys_per_bay):
            self._sheet_creation(wing.scaffold[i, 0, :], wing.scaffold[i, 1, :], wing.scaffold[i+1, 0, :], wing.scaffold[i+1, 1, :], ny, "pq") #fore spar
            self._sheet_creation(wing.scaffold[i, -1, :], wing.scaffold[i, -2, :], wing.scaffold[i+1, -1, :], wing.scaffold[i+1, -2, :], ny, "pq") #rear spar

            for inbf, inbr, oubf, oubr in zip(self.wing.scaffold[i, 1:-2, :], self.wing.scaffold[i, 2:-1, :], self.wing.scaffold[i+1, 1:-2, :], self.wing.scaffold[i+1, 2:-1, :]):
                self._sheet_creation(inbf, inbr, oubf, oubr, ny, "aq") #angled spars

            for inbf, inbr, oubf, oubr in zip(self.wing.scaffold[i, :-2:2, :], self.wing.scaffold[i, 2::2, :], self.wing.scaffold[i+1, :-2:2, :], self.wing.scaffold[i+1, 2::2, :]):
                self._sheet_creation(inbf, inbr, oubf, oubr, ny, "sq", True) #top skin

            for inbf, inbr, oubf, oubr in zip(self.wing.scaffold[i, 1:-2:2, :], self.wing.scaffold[i, 3::2, :], self.wing.scaffold[i+1, 1:-2:2, :], self.wing.scaffold[i+1, 3::2, :]):
                self._sheet_creation(inbf, inbr, oubf, oubr, ny, "sq", False) #lower skin

            self._rib_creation(i+1)

            if i < furthest_motor_i: #NOTE: for i = furthest i we have our inboard at the boundary so the bay is outboard of the boundary
                self._railing_creation(self.wing.scaffold[i, 0, :], self.wing.scaffold[i+1, 0, :], ny, [], "sb") #top motor support
                self._railing_creation(self.wing.scaffold[i, 1, :], self.wing.scaffold[i+1, 1, :], ny, [], "sb") #bottom motor support

            if i < furthest_lg_i: #same for landing gear supports
                self._railing_creation(self.wing.scaffold[i, -1, :], self.wing.scaffold[i+1, -1, :], ny, [], "sb")
                self._railing_creation(self.wing.scaffold[i, -2, :], self.wing.scaffold[i+1, -2, :], ny, [], "sb")

            for inb, oub in zip(self.wing.scaffold[i, 2:-2, :], self.wing.scaffold[i+1, 2:-2, :]):
                self._railing_creation(inb, oub, ny, [HYPERPARAMS['d']], "rb") #battery rails

        #3) consistency check
        assert len(self.eleTypes) == len(self.eleArgs) == len(self.eleNodes)

        #3.1) getting the node count that one should obtain if this part of mesh is resolved correctly
        n_top_skin_panels = self.wing.scaffold[0, :-2:2, 0].shape[0]
        n_low_skin_panels = self.wing.scaffold[0, 1:-2:2, 0].shape[0]
        n_spars = 2
        n_angled_spars = self.wing.scaffold[0, 1:-2, 0].shape[0]
        #corner nodes plus intermediate nodes
        nodes_per_sec = self.wing.scaffold.shape[1]+(self.n-2)*(n_spars+n_angled_spars+n_top_skin_panels+n_low_skin_panels)
        #at all but first bay, one onde is repeated
        overall_y_count = ys_per_bay.sum()-ys_per_bay.shape[0]+1
        self.expected_node_count = nodes_per_sec*overall_y_count


    def _railing_creation(self, inb:nt.NDArray[np.float64], oub:nt.NDArray, ny:int, eleArg:list[float], eleType:str):
        c_inb = self.wing.c_at_y(inb[1])
        c_oub = self.wing.c_at_y(oub[1])
        rj_inb = self.HYPERPARAMS["rj/c"]*c_inb
        rj_oub = self.HYPERPARAMS["rj/c"]*c_oub
        dj_avg = rj_oub+rj_inb
        delta_y = oub[1]-inb[1]
        nj_n = dj_avg/delta_y
        nj_signle_side = int(np.round(nj_n/2*(ny-4))) #how many ns to fill each joint with, we have 2 edge joints and joint boundaries, so we need at least 4 nodes

        ys = np.array(np.linspace(inb[1], inb[1]+rj_inb, 2+nj_signle_side)[:-1].tolist()+np.linspace(inb[1]+rj_inb, oub[1]-rj_oub, ny-2*nj_signle_side-2).tolist()+
                      np.linspace(oub[1]-rj_oub, oub[1], 2+nj_signle_side)[1:].tolist())
        fracs = (ys-inb[1])/delta_y #projecting the distorted mesh
        xs = inb[0]+fracs*(oub[0]-inb[0])
        zs = inb[2]+fracs*(oub[2]-inb[2])
        assert ys.shape == xs.shape == zs.shape == (ny,)

        for x1, y1, z1, x2, y2, z2 in zip(xs[:-1], ys[:-1], zs[:-1], xs[1:], ys[1:], zs[1:]):
           self.eleTypes.append(eleType) 
           self.eleArgs.append(eleArg)
           self.eleNodes.append([(x1, y1, z1), (x2, y2, z2)])
    
    
    def _sheet_creation(self, inbf:nt.NDArray[np.float64], inbr:nt.NDArray[np.float64], oubf:nt.NDArray[np.float64], oubr:nt.NDArray[np.float64], ny:int, eleType:str, skin:bool=None):
        """
        zproj - None for no projection; True for upper skin, false for lower skin
        """
        Hpc = self.HYPERPARAMS[f"(H/c)_{eleType}"]

        c_inb = self.wing.c_at_y(inbf[1])
        c_oub = self.wing.c_at_y(oubf[1])
        rj_inb = self.HYPERPARAMS["rj/c"]*c_inb
        rj_oub = self.HYPERPARAMS["rj/c"]*c_oub
        dj_avg = rj_oub+rj_inb

        delta_y = oubf[1]-inbf[1]
        nyj_n = dj_avg/delta_y
        nyj_signle_side = int(np.round(nyj_n/2*(ny-4))) #how many ns to fill each joint with, we have 2 edge joints and joint boundaries, so we need at least 4 nodes

        yvals = np.array(np.linspace(inbf[1], inbf[1]+rj_inb, 2+nyj_signle_side)[:-1].tolist()+np.linspace(inbf[1]+rj_inb, oubf[1]-rj_oub, ny-2*nyj_signle_side-2).tolist()+
                         np.linspace(oubf[1]-rj_oub, oubf[1], 2+nyj_signle_side)[1:].tolist())
        # Spanwise interpolation parameter (0=inboard, 1=outboard)
        eta = ((yvals-inbf[1])/delta_y)[:, None]   # shape (ny, 1)

        #z values depend on whether we are creating a spar or a skin - a spar can just follow linear interpolation,
        #while the skin needs to follow the wing curvature at the stations and interpolate linearly for remaining zs
        if skin is None:
            xoubs, zoubs = self._ribbing_xzs(oubf[0], oubf[2], oubr[0], oubr[2], rj_oub)
            xinbs, zinbs = self._ribbing_xzs(inbf[0], inbf[2], inbr[0], inbr[2], rj_inb)
        elif skin:
            xoubs, _ = self._ribbing_xzs(oubf[0], 0., oubr[0], 0., rj_oub)
            xinbs, _ = self._ribbing_xzs(inbf[0], 0., inbr[0], 0., rj_inb)
            zoubs = np.array([self.wing.upper_skin_z(self.wing.xperc_reduced_from_x(x, yvals[-1]), yvals[-1]) for x in xoubs])
            zinbs = np.array([self.wing.upper_skin_z(self.wing.xperc_reduced_from_x(x, yvals[0]), yvals[0]) for x in xinbs])
        else:
            xoubs, _ = self._ribbing_xzs(oubf[0], 0., oubr[0], 0., rj_oub)
            xinbs, _ = self._ribbing_xzs(inbf[0], 0., inbr[0], 0., rj_inb)
            zoubs = np.array([self.wing.lower_skin_z(self.wing.xperc_reduced_from_x(x, yvals[-1]), yvals[-1]) for x in xoubs])
            zinbs = np.array([self.wing.lower_skin_z(self.wing.xperc_reduced_from_x(x, yvals[0]), yvals[0]) for x in xinbs]) 

        X = xinbs[None, :] + eta * (xoubs - xinbs)[None, :]
        Y = np.repeat(yvals, self.n).reshape((ny, self.n))
        Z = zinbs[None, :] + eta * (zoubs - zinbs)[None, :]
        assert X.shape == Y.shape == Z.shape == (ny, self.n)

        for x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4 in zip(X[:-1, :-1].flatten(), Y[:-1, :-1].flatten(), Z[:-1, :-1].flatten(),
                                                                  X[:-1, 1:].flatten(), Y[:-1, 1:].flatten(), Z[:-1, 1:].flatten(),
                                                                  X[1:, 1:].flatten(), Y[1:, 1:].flatten(), Z[1:, 1:].flatten(), 
                                                                  X[1:, :-1].flatten(), Y[1:, :-1].flatten(), Z[1:, :-1].flatten()):
            c = (self.wing.c_at_y(y1)+self.wing.c_at_y(y4))/2
            H = Hpc*c
            self.eleTypes.append(eleType)
            self.eleArgs.append([H])
            self.eleNodes.append([(x1, y1, z1), (x2, y2, z2), (x3, y3, z3), (x4, y4, z4)])


    def _rib_creation(self, istation:int):
        y = self.wing.yribs[istation]
        c = self.wing.c_at_y(y)
        code = self.wing.ribcodes[istation]
        Hsq = self.HYPERPARAMS["(H/c)_sq"]*c
        Hpq = self.HYPERPARAMS["(H/c)_pq"]*c
        Haq = self.HYPERPARAMS["(H/c)_aq"]*c
        rj = self.HYPERPARAMS["rj/c"]*c

        #1) the part of rib on the fore spar
        _, zpqf = self._ribbing_xzs(0., self.wing.scaffold[istation, 0, 2], 0., self.wing.scaffold[istation, 1, 2], rj)
        xf = self.wing.scaffold[istation, 0, 0]
        for z1, z2 in zip(zpqf[:-1], zpqf[1:]):
            self.eleTypes.append(code)
            self.eleArgs.append([Hpq])
            self.eleNodes.append([(xf, y, z1), (xf, y, z2)])

        #2) the part of rib on the rear spar
        _, zpqr = self._ribbing_xzs(0., self.wing.scaffold[istation, -1, 2], 0., self.wing.scaffold[istation, -2, 2], rj)
        xr = self.wing.scaffold[istation, -1, 0]
        for z1, z2 in zip(zpqr[:-1], zpqr[1:]):
            self.eleTypes.append(code)
            self.eleArgs.append([Hpq])
            self.eleNodes.append([(xr, y, z1), (xr, y, z2)])

        #3) the angled spars
        for xb1, zb1, xb2, zb2 in zip(self.wing.scaffold[istation, 1:-2, 0], self.wing.scaffold[istation, 1:-2, 2], self.wing.scaffold[istation, 2:-1, 0], self.wing.scaffold[istation, 2:-1, 2]):
            xs, zs = self._ribbing_xzs(xb1, zb1, xb2, zb2, rj)
            for x1, x2, z1, z2 in zip(xs[:-1], xs[1:], zs[:-1], zs[1:]):
                self.eleTypes.append(code)
                self.eleArgs.append([Haq])
                self.eleNodes.append([(x1, y, z1), (x2, y, z2)])

        #4) the upper skin
        for xb1, xb2 in zip(self.wing.scaffold[istation, :-2:2, 0], self.wing.scaffold[istation, 2::2, 0]):
            xs, _ = self._ribbing_xzs(xb1, 0., xb2, 0., rj)
            xperc = [self.wing.xperc_reduced_from_x(x, y) for x in xs]
            zs = [self.wing.upper_skin_z(xpc, y) for xpc in xperc]
            for x1, x2, z1, z2 in zip(xs[:-1], xs[1:], zs[:-1], zs[1:]):
                self.eleTypes.append(code)
                self.eleArgs.append([Hsq])
                self.eleNodes.append([(x1, y, z1), (x2, y, z2)])

        #5) the lower skin
        for xb1, xb2 in zip(self.wing.scaffold[istation, 1:-2:2, 0], self.wing.scaffold[istation, 3::2, 0]):
            xs, _ = self._ribbing_xzs(xb1, 0., xb2, 0., rj)
            xperc = [self.wing.xperc_reduced_from_x(x, y) for x in xs]
            zs = [self.wing.lower_skin_z(xpc, y) for xpc in xperc]
            for x1, x2, z1, z2 in zip(xs[:-1], xs[1:], zs[:-1], zs[1:]):
                self.eleTypes.append(code)
                self.eleArgs.append([Hsq])
                self.eleNodes.append([(x1, y, z1), (x2, y, z2)])


    def _ribbing_xzs(self, x1:float, z1:float, x2:float, z2:float, rj:float):
        delta_xz = np.sqrt((x2-x1)**2+(z2-z1)**2)
        nj_n = 2*rj/delta_xz
        nj_signle_side = int(np.round(nj_n/2*(self.n-4)))
        rx = (x2-x1)*rj/delta_xz
        rz = (z2-z1)*rj/delta_xz
        assert np.isclose(rx**2+rz**2, rj**2)

        xs = np.array(np.linspace(x1, x1+rx, 2+nj_signle_side)[:-1].tolist()+np.linspace(x1+rx, x2-rx, self.n-2*nj_signle_side-2).tolist()+np.linspace(x2-rx, x2, 2+nj_signle_side)[1:].tolist())
        zs = np.array(np.linspace(z1, z1+rz, 2+nj_signle_side)[:-1].tolist()+np.linspace(z1+rz, z2-rz, self.n-2*nj_signle_side-2).tolist()+np.linspace(z2-rz, z2, 2+nj_signle_side)[1:].tolist())

        return xs, zs