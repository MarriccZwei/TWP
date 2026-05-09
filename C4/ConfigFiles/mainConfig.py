import aerosandbox.numpy as np
import aerosandbox as asb

from .classified import MASSES

DESVARS_INITIAL = {
        '(2t/H)_Sq':0.3,
        '(2t/H)_Pq':0.15,
        '(2t/H)_Aq':0.1,
        'W_bb':0.004,
        'W_mb':0.004,
        'W_lb':0.03,
        'ds':.02,
        'de':.01,
        '(2t/H)_sq':0.06,
        '(2t/H)_pq':0.03,
        '(2t/H)_aq':0.03
    }

MATERIALS = {
    'E_ALU':72.4e9,
    'NU_ALU':.33,
    'RHO_ALU':2780.,
    'SF_ALU':323.33e6,
    'E_FOAM':0.58644e9,
    'NU_FOAM':.275,
    'RHO_FOAM':250.2,
    'SF_FOAM':3.645e6
}

LC_INFO = [
    {
        'n':2.5,
        'nlg':0.,
        'Ttot':112800., # [N]
        'op':asb.OperatingPoint(asb.Atmosphere(0.), velocity=90., alpha=10.), #[h]=m, [v]=m/s, [alpha]=deg
        'bank':0.,
        'aeroelastic':False
    },
    {
        'n':-1.,
        'nlg':0.,
        'Ttot':32400., # [N]
        'op':asb.OperatingPoint(asb.Atmosphere(7000.), velocity=187., alpha=-4.5), #[h]=m, [v]=m/s, [alpha]=deg
        'bank':0.,
        'aeroelastic':False
    },
    {
        'n':1.,
        'nlg':0.,
        'Ttot':37800., # [N]
        'op':asb.OperatingPoint(asb.Atmosphere(0.), velocity=215., alpha=-.75), #[h]=m, [v]=m/s, [alpha]=deg
        'bank':0.,
        'aeroelastic':True
    },
    {
        'n':1.,
        'nlg':1.5,
        'Ttot':0., # [N]
        'op':asb.OperatingPoint(asb.Atmosphere(0.), velocity=74., alpha=15.), #[h]=m, [v]=m/s, [alpha]=deg
        'bank':6.,
        'aeroelastic':False
    },
]

RES = {
    'bres':20,
    'cres':10,
    'nneighs':10,
    'kfl':30,
    'klb':4,
    'sks':(1e10, 0., 0., 1e10, 0., 0., 0., 1., 0.)
}

G0 = 9.81 # [N/kg]
MTOM = 76000. # [kg]
LBUCKLSF = 1.5

N = 11

BOUNDS = (
    {
        '(2t/H)_Sq':0.03,
        '(2t/H)_Pq':0.03,
        '(2t/H)_Aq':0.03,
        'W_bb':0.003,
        'W_mb':0.003,
        'W_lb':0.003,
        'ds':.004,
        'de':.004,
        '(2t/H)_sq':0.03,
        '(2t/H)_pq':0.03,
        '(2t/H)_aq':0.03,
    },
    {
        '(2t/H)_Sq':0.3,
        '(2t/H)_Pq':0.3,
        '(2t/H)_Aq':0.3,
        'W_bb':0.03,
        'W_mb':0.03,
        'W_lb':0.03,
        'ds':.02,
        'de':.02,
        '(2t/H)_sq':0.3,
        '(2t/H)_pq':0.3,
        '(2t/H)_aq':0.3,
    }
)

NAIRFS = 10

HYPERPARAMS ={
        'delta':.005,
        'D':.3,
        'd':.03,
        'Delta b':.1,
        '(H/c)_sq':.009,
        '(H/c)_aq':.003,
        '(H/c)_pq':.006,
    }

GEOM_SOURCE ={
        #NOTE: all coordinates in m
        "yfus":1.602374,
        "yhn":18.,
        "ytip":21.,
        "deltazhn":1.362017,
        "deltaxhn":1.548961,
        "(x/c)_fore":.15,
        "(x/c)_rear":.7,
        "cr":5.,
        "ct":2.,
        "ylg":5.768546,
        "deltaxlg":.4039,
        "rlg":.801187,
        "ym1":4.005935,
        "ym2":8.091988,
        "ym3":12.178042,
        "ym4":16.290801,
        "deltaxm1":-.801187,
        "deltaxm2":-.267062,
        "deltaxm3":.267062,
        "deltaxm4":.801187,
        "rootfoil":asb.Airfoil("naca2418"),
        "tipfoil":asb.Airfoil("naca2410")
    }