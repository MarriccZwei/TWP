import numpy as np
import aerosandbox as asb
import typing as ty

class Config:
    def __init__(
        self,
        # Physical constants
        G0: float,              # N/kg
        YIELD_MARGIN: float,    # [-]

        # Material properties
        E_ALU: float,           # Pa
        E_STEEL: float,         # Pa
        NU: float,              # [-]
        RHO_ALU: float,         # kg/m^3
        RHO_STEEL: float,       # kg/m^3
        SIGMAY_ALU: float,      # Pa
        SIGMAY_STEEL: float,    # Pa
        TAUY_ALU: float,        # Pa
        TAUY_STEEL: float,      # Pa

        # Geometry assumptions
        TRUSS_RC: float,        # [-] ratio of truss radius to chord 
        ORTHG_H: float,         # m, height of the orthogrid stiffeners
        DIN: float,             # m, inner diameter of the rail wormgear
        FULLHALFSPAN: float,        # m, full halfspan
        LGR: float,             # m
        LGL: float,             # m
        ML: float,              # m
        MR: float,              # m
        FOIL_YS: ty.List[float],   # m, ys at which airfoil chords are sampled
        FOILS: ty.List[asb.Airfoil],

        # Masses
        BAT_MASS_1WING: float,  # kg
        MTOM: float,            # kg
        M_MOTOR: float,         # kg, entire engine group of a single engine
        M_LG: float,            # kg, mass of entire landing gear group
        M_LE: float,            # kg, mass of all LE equipment
        M_TE: float,            # kg, mass of all TE equipment
        M_HINGE: float,         # kg, mass of the hinge

        # Mesh settings
        N_COEFF: int,           # [-]
        NA: int,                # [-]

        # CFD settings
        BRES: int,              # [-]
        CRES: int               # [-]
    ) -> None:

        # --- Assign attributes ---
        # Physical constants
        self.G0 = G0
        self.YIELD_MARGIN = YIELD_MARGIN

        # Material properties
        self.E_ALU = E_ALU
        self.E_STEEL = E_STEEL
        self.NU = NU
        self.RHO_ALU = RHO_ALU
        self.RHO_STEEL = RHO_STEEL
        self.SIGMAY_ALU = SIGMAY_ALU
        self.SIGMAY_STEEL = SIGMAY_STEEL
        self.TAUY_ALU = TAUY_ALU
        self.TAUY_STEEL = TAUY_STEEL

        # Geometry assumptions
        self.TRUSS_RC = TRUSS_RC 
        self.ORTHG_H = ORTHG_H
        self.DIN = DIN
        self.FULLHALFSPAN = FULLHALFSPAN
        self.LGR = LGR
        self.LGL = LGL
        self.ML = ML
        self.MR = MR
        self.FOIL_YS = FOIL_YS
        self.FOILS = FOILS

        # Masses - all in kg
        self.BAT_MASS_1WING = BAT_MASS_1WING
        self.MTOM = MTOM
        self.M_MOTOR = M_MOTOR
        self.M_LG = M_LG
        self.M_LE = M_LE
        self.M_TE = M_TE
        self.M_HINGE = M_HINGE

        # Mesh settings
        self.N_COEFF = N_COEFF
        self.NA = NA

        # CFD settings
        self.BRES = BRES
        self.CRES = CRES

    def __repr__(self) -> str:
        return f"<Config: {len(self.__dict__)} parameters>"



if __name__ =="__main__":
    cfg = Config(
    G0=9.81,
    YIELD_MARGIN=1.2,
    E_ALU=72e9,
    E_STEEL=200e9,
    NU=0.33,
    RHO_ALU=2700,
    RHO_STEEL=7850,
    SIGMAY_ALU=435e6,
    SIGMAY_STEEL=2000e6,
    TAUY_ALU=207e6,
    TAUY_STEEL=0.6 * 2000e6,
    TRUSS_RC=0.05,
    ORTHG_H=0.02,
    DIN=0.015,
    FULLHALFSPAN=21,
    LGR=0.5,
    LGL=5,
    ML=3.5,
    MR=0.4,
    FOIL_YS=[0, 1.6, 6, 10, 14, 18, 21],
    FOILS=[asb.Airfoil("naca2412")] * 7,
    BAT_MASS_1WING=17480,
    MTOM=76000,
    M_MOTOR=0, #classified
    M_LG=0, #classified
    M_LE=0, #classified
    M_TE=0, #classified
    M_HINGE=0, #classified
    N_COEFF=1,
    NA=5,
    BRES=20,
    CRES=10
)