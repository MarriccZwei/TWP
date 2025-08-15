import geometricClasses as gcl
import numpy as np
import aerosandbox as asb
'''ALL PARAMETERS NOT CHANGED BY THE SIZER'''
CONSTS={
    'G0':9.81, #N/kg
    'YIELD_MARGIN':1.2, #[-]

    #material properties
    'E_ALU':72e9, #Pa
    'E_STEEL':200e9, #Pa #TODO: verify
    'NU':.33, #[-]
    'RHO_ALU':2700, #kg/m^3
    'RHO_STEEL':7850, #kg/m^3 #TODO: verify
    'SIGMAY_ALU':435e6, #Pa #TODO: verify
    'SIGMAY_STEEL':2000e6, #Pa #TODO:verify
    'TAUY_ALU':207e6, #Pa #TODO: verify
    'TAUY_STEEL':.6*2000e6, #Pa #TODO: verify

    #geometry assumptions
    'TRUSS_R':0.05, #m, radius of the trusses
    'ORTHG_H':0.03, #m, height of the orthogrid stiffeners
    'DIN':0.015, #inner diameter of the rail wormgear [m]
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
    'N_COEFF':1,
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
    'rib':'rb',
    'LETE':'lt',
    'rail':'rl',
    'lg':'lg',
}

#styled as a sizer protocol
INITIAL = {
#skin
'tskin':0.005, #[m]
'csp': 0.15, #spacing [m]
'bsp': 0.3, #spacing [m]

#truss
'fspar':0.05, #[-], spar thickness/radius fraction
'frib':0.02, #[-], rib thickness/radius fraction
'fLETE':0.03, #[-], LETE thickness/radius fraction
'flg':0.075, #[-], LG truss thickness/radius fraction
}

CAD_DATA="5000;0;-6.5|4750;0;-24|4500;0;-41|4000;0;-75|3500;0;-107|3000;0;-138|2500;0;-167|2000;0;-190|1500;0;-206|1250;0;-211|1000;0;-211.5|750;0;-205|500;0;-187.5|375;0;-173|250;0;-150.5|125;0;-113.5|62.5;0;-82.5|0;0;0|62.5;0;107.5|125;0;149.5|250;0;206.5|375;0;248|500;0;281.5|750;0;330.5|1000;0;363|1250;0;383.5|1500;0;394|2000;0;390|2500;0;362|3000;0;318|3500;0;259|4000;0;187.5|4500;0;104|4750;0;57|5000;0;6.5&4250;18000;2206.872|4125;18000;2198.122|4000;18000;2189.622|3750;18000;2172.622|3500;18000;2156.622|3250;18000;2141.122|3000;18000;2126.622|2750;18000;2115.122|2500;18000;2107.122|2375;18000;2104.622|2250;18000;2104.372|2125;18000;2107.622|2000;18000;2116.372|1937.5;18000;2123.622|1875;18000;2134.872|1812.5;18000;2153.372|1781.25;18000;2168.872|1750;18000;2210.122|1781.25;18000;2263.872|1812.5;18000;2284.872|1875;18000;2313.372|1937.5;18000;2334.122|2000;18000;2350.872|2125;18000;2375.372|2250;18000;2391.622|2375;18000;2401.872|2500;18000;2407.122|2750;18000;2405.122|3000;18000;2391.122|3250;18000;2369.122|3500;18000;2339.622|3750;18000;2303.872|4000;18000;2262.122|4125;18000;2238.622|4250;18000;2213.372&-471.576;3673.46;441.249|126.713;7770.33;945.522|725.002;11867.2;1449.796|1323.292;15964.069;1954.07&3012.266;18950.167;2324.854|4861.588;5721.895;702.56&500;0;0|1999.77;18000;2214.157&3750;0;0|3624.77;18000;2214.157&871.849;1600;0.583|871.849;1600;512.208|1171.071;1600;-6.06|1503.866;1600;570.358|1829.363;1600;6.581|2152.161;1600;565.683|2460.005;1600;32.482|2745.201;1600;526.456|3012.213;1600;63.976|3249.622;1600;475.18|3470.631;1600;92.38|3470.631;1600;447.784&2124.805;18000;2107.631|2124.805;18000;2375.342|2281.374;18000;2104.155|2455.511;18000;2405.769|2625.829;18000;2110.769|2794.735;18000;2403.323|2955.816;18000;2124.322|3105.047;18000;2382.797|3244.763;18000;2140.802|3368.988;18000;2355.966|3484.632;18000;2155.664|3484.632;18000;2341.631&0;0;0|5000;0;0|1749.77;18000;2214.157|4250;18000;2210.122" 
    