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
        N_COEFF: int,           # [-], refinement over the number of nodes per feature
        NA: int,                # [-], number of nodes per feature necessary for buckling analysis, should be 5

        # CFD settings
        BRES: int,              # [-]
        CRES: int,              # [-]

        #CAD import
        CAD_DATA:str,

        #Initial values of design variables
        TSKIN_INITIAL: float,  # [m]
        CSP_INITIAL: float,    # spacing [m]
        BSP_INITIAL: float,    # spacing [m]

        # Truss
        FSPAR_INITIAL: float,  # [-], spar thickness/radius fraction
        FRIB_INITIAL: float,   # [-], rib thickness/radius fraction
        FLETE_INITIAL: float,  # [-], LETE thickness/radius fraction
        FLG_INITIAL: float,    # [-], LG truss thickness/radius fraction

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

        #CAD import
        self.CAD_DATA = CAD_DATA

        #Initial values of design variables
        TSKIN_INITIAL = TSKIN_INITIAL
        CSP_INITIAL = CSP_INITIAL
        BSP_INITIAL = BSP_INITIAL
        FSPAR_INITIAL = FSPAR_INITIAL
        FRIB_INITIAL = FRIB_INITIAL
        FLETE_INITIAL = FLETE_INITIAL
        FLG_INITIAL = FLG_INITIAL
        

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
    CRES=10,
    CAD_DATA="5000;0;-6.5|4750;0;-24|4500;0;-41|4000;0;-75|3500;0;-107|3000;0;-138|2500;0;-167|2000;0;-190|1500;0;-206|1250;0;-211|1000;0;-211.5|750;0;-205|500;0;-187.5|375;0;-173|250;0;-150.5|125;0;-113.5|62.5;0;-82.5|0;0;0|62.5;0;107.5|125;0;149.5|250;0;206.5|375;0;248|500;0;281.5|750;0;330.5|1000;0;363|1250;0;383.5|1500;0;394|2000;0;390|2500;0;362|3000;0;318|3500;0;259|4000;0;187.5|4500;0;104|4750;0;57|5000;0;6.5&4250;18000;2206.872|4125;18000;2198.122|4000;18000;2189.622|3750;18000;2172.622|3500;18000;2156.622|3250;18000;2141.122|3000;18000;2126.622|2750;18000;2115.122|2500;18000;2107.122|2375;18000;2104.622|2250;18000;2104.372|2125;18000;2107.622|2000;18000;2116.372|1937.5;18000;2123.622|1875;18000;2134.872|1812.5;18000;2153.372|1781.25;18000;2168.872|1750;18000;2210.122|1781.25;18000;2263.872|1812.5;18000;2284.872|1875;18000;2313.372|1937.5;18000;2334.122|2000;18000;2350.872|2125;18000;2375.372|2250;18000;2391.622|2375;18000;2401.872|2500;18000;2407.122|2750;18000;2405.122|3000;18000;2391.122|3250;18000;2369.122|3500;18000;2339.622|3750;18000;2303.872|4000;18000;2262.122|4125;18000;2238.622|4250;18000;2213.372&-471.576;3673.46;441.249|126.713;7770.33;945.522|725.002;11867.2;1449.796|1323.292;15964.069;1954.07&3012.266;18950.167;2324.854|4861.588;5721.895;702.56&500;0;0|1999.77;18000;2214.157&3750;0;0|3624.77;18000;2214.157&871.849;1600;0.583|871.849;1600;512.208|1171.071;1600;-6.06|1503.866;1600;570.358|1829.363;1600;6.581|2152.161;1600;565.683|2460.005;1600;32.482|2745.201;1600;526.456|3012.213;1600;63.976|3249.622;1600;475.18|3470.631;1600;92.38|3470.631;1600;447.784&2124.805;18000;2107.631|2124.805;18000;2375.342|2281.374;18000;2104.155|2455.511;18000;2405.769|2625.829;18000;2110.769|2794.735;18000;2403.323|2955.816;18000;2124.322|3105.047;18000;2382.797|3244.763;18000;2140.802|3368.988;18000;2355.966|3484.632;18000;2155.664|3484.632;18000;2341.631&0;0;0|5000;0;0|1749.77;18000;2214.157|4250;18000;2210.122",
    TSKIN_INITIAL = 0.005,
    CSP_INITIAL = 0.125,
    BSP_INITIAL = 0.125,
    FSPAR_INITIAL = 0.05,
    FRIB_INITIAL = 0.02,
    FLETE_INITIAL = 0.03,
    FLG_INITIAL = 0.075,
)