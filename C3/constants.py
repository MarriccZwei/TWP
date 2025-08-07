import geometricClasses as gcl
import numpy as np
import aerosandbox as asb
'''ALL PARAMETERS NOT CHANGED BY THE SIZER'''
CONSTS={
    'G0':9.81, #N/kg

    #material properties
    'E_ALU':72e9, #Pa
    'E_STEEL':200e9, #Pa #TODO: verify
    'NU':.33, #[-]
    'RHO_ALU':2700, #kg/m^3
    'RHO_STEEL':7850, #kg/m^3 #TODO: verify

    #geometry assumptions
    'RIB_FLANGE':0.0125, #m
    'DIN':0.015, #inner diameter of the rail wormgear [m]
    'NTRIG':2, #number of batteries supported on the lower panel [-]
    'FULLSPAN':21, #m, full halfspan
    'LGR':.5, #m
    'LGL':5, #m
    'ML': 3, #m
    'MR': .4, #m
    'FOIL_YS': [0, 1.6, 10, 18, 21], #m, ys at which airfoilchords should be sampled
    'FOILS':[asb.Airfoil("naca2412")]*5,

    #masses
    'BAT_MASS_1WING':17480, #kg
    'MTOM':76000, #kg
    'M_MOTOR':1000, #Entire engine group of a single engine [kg]
    'M_LG':5000, #Mass of the entire landing gear group [kg]
    'M_LE':1000, #Mass of all other LE eqpt: thermal management, avionics, 1/2 cable/pipe mass etc. [kg]
    'M_TE':3000, #Mass of all TE eqpt: hlds, 1/2 cable/pipe mass etc. [kg]
    'M_HINGE':500, #Mass of the hinge
    
    #Mesh settings
    'NB_COEFF':1,
    'NIP_COEFF':1,
    'NA':5,
    'NF2':3,

    #CFD settings
    'BRES':20,
    'CRES':10,
    'VELOCITIES':np.linspace(50, 250, 3)
    }

LOAD_C=[
    {'n':2.5, 'nult':2.5*1.5, 'nlg': 0, 'ndir':gcl.Direction3D(0, 0, 1), "LD":20, "FT":5000,
     'op':asb.OperatingPoint(asb.Atmosphere(7000), .6*310, 3)}, #symmetric coordinated turn
    {'n':1, 'nult':1.5, 'nlg': 0, 'ndir':gcl.Direction3D(0, 0, -1), "LD":16, "FT":2500,
     'op':asb.OperatingPoint(asb.Atmosphere(7000), .6*310, -3)}, #negative load factor
    {'n':1.5, 'nult':1.5*1.5, 'nlg': 2, 'ndir':gcl.Direction3D(0, 0, 1), "LD":14, "FT":0,
     'op':asb.OperatingPoint(asb.Atmosphere(7000), .6*310, 5)}, #landing loads
    ]

#internal ids of element types
CODES={
    'spar':'sp',
    'skin':'sk',
    'panelPlate':'pp',
    'panelRib':'pr',
    'panelFlange':'pf',
    'rail':'rl',
    'varpt':'vp', #special code - this code corresponds to mass point with mass in protocol
    'motor':'en', #typical would be mt, so none shall get it
    'mount':'mm', #typical would be mt, so none shall get it
    'hinge':'hn',
    'lg':'lg',
}

#styled as a sizer protocol
INITIAL = {
'tspar':0.005, #m
'tskin':0.003, #m
'tpan':0.002, #m
'trib':0.0025, #m
'csp':.3 #chordwise panel rib spacing (maximum) [m]
}