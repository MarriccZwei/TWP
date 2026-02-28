from .ElysianWing import ElysianWing

import aerosandbox.numpy as np
import numpy.typing as nt

class StructuralSubmesh():
    def __init__(self, wing:ElysianWing, HYPERPARAMS:dict[str, float], n:int):
        self.eleTypes:list[str] = list()
        self.eleArgs:list[list[float]] = list()
        self.eleNodes:list[list[tuple[float]]] = list()
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

        n_x_tot = n * wing.scaffold[:, ::2, 0].shape[1]

        ys_per_bay = np.ceil(n_x_tot * y_difference / x_difference_avg).astype(np.int8)

        #2)sheet creation
        for i, ny in enumerate(ys_per_bay):
            self._sheet_creation(wing.scaffold[i, 0, :], wing.scaffold[i, 1, :], wing.scaffold[i+1, 0, :], wing.scaffold[i+1, 1, :], ny, "sq") #fore spar
            self._sheet_creation(wing.scaffold[i, -1, :], wing.scaffold[i, -2, :], wing.scaffold[i+1, -1, :], wing.scaffold[i+1, -2, :], ny, "sq") #rear spar
            self._rib_creation(i)


    def _sheet_creation(self, inbf:nt.NDArray[np.float64], inbr:nt.NDArray[np.float64], oubf:nt.NDArray[np.float64], oubr:nt.NDArray[np.float64], ny:int, eleType:str, skin:bool=None):
        """
        zproj - None for no projection; True for upper skin, false for lower skin
        """
        Hpc = self.HYPERPARAMS[f"(H/c)_{eleType}"]

        yvals = np.linspace(inbf[1], oubf[1], ny)
        xoubs = np.linspace(oubf[0], oubr[0], self.n)
        xinbs = np.linspace(inbf[0], inbr[0], self.n)
        zoubs = np.linspace(oubf[2], oubr[2], self.n)
        zinbs = np.linspace(inbf[2], inbr[2], self.n)

        # Spanwise interpolation parameter (0=inboard, 1=outboard)
        eta = np.linspace(0.0, 1.0, ny)[:, None]   # shape (ny, 1)
        X = xinbs[None, :] + eta * (xoubs - xinbs)[None, :]
        Y = np.repeat(yvals, self.n).reshape((ny, self.n))
        Z = np.zeros((ny, self.n))

        #Z is a projection for skins, but for other element it's a clear sheet
        if skin is None:
            Z = zinbs[None, :] + eta * (zoubs - zinbs)[None, :]
        else: 
            for i in range(ny):
                for j in range(self.n):
                    x = X[i, j]
                    y = Y[i, j]
                    xpc = self.wing.xperc_reduced_from_x(x, y)
                    Z[i, j] = self.wing.upper_skin_z(xpc, y) if skin else self.wing.lower_skin_z(xpc, y)

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

        #1) the part of rib on the fore spar
        zpqf = np.linspace(self.wing.scaffold[istation, 0, 2], self.wing.scaffold[istation, 1, 2], self.n)
        xf = self.wing.scaffold[istation, 0, 0]
        for z1, z2 in zip(zpqf[:-1], zpqf[1:]):
            self.eleTypes.append(code)
            self.eleArgs.append([Hpq])
            self.eleNodes.append([(xf, y, z1), (xf, y, z2)])

        #2) the part of rib on the rear spar
        zpqr = np.linspace(self.wing.scaffold[istation, -1, 2], self.wing.scaffold[istation, -2, 2], self.n)
        xr = self.wing.scaffold[istation, -1, 0]
        for z1, z2 in zip(zpqr[:-1], zpqr[1:]):
            self.eleTypes.append(code)
            self.eleArgs.append([Hpq])
            self.eleNodes.append([(xr, y, z1), (xr, y, z2)])