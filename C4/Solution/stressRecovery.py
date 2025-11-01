import pyfe3d as pf3
import pyfe3d.shellprop as psp
import numpy as np
import numpy.typing as nt
import numpy.linalg as nl

def strains_resultants_quad(probe:pf3.Quad4Probe, shellprop:psp.ShellProp):
    '''
    The probe has to already be updated with the xe, ue and KC0ve of the quad!!!
    '''

    exx = list()
    eyy = list()
    gxy = list()
    kxx = list()
    kyy = list()
    kxy = list()
    gxz = list()
    gyz = list()

    Nxx = list()
    Nyy = list()
    Nxy = list()
    Mxx = list()
    Myy = list()
    Mxy = list()
    Qxz = list()
    Qyz = list()

    A11 = shellprop.A11
    A12 = shellprop.A12
    A22 = shellprop.A22
    A16 = shellprop.A16
    A26 = shellprop.A26
    A66 = shellprop.A66

    B11 = shellprop.B11
    B12 = shellprop.B12
    B22 = shellprop.B22
    B16 = shellprop.B16
    B26 = shellprop.B26
    B66 = shellprop.B66

    D11 = shellprop.D11
    D12 = shellprop.D12
    D22 = shellprop.D22
    D16 = shellprop.D16
    D26 = shellprop.D26
    D66 = shellprop.D66

    E44 = shellprop.E44
    E45 = shellprop.E45
    E55 = shellprop.E55

    for xi, eta in [(0,0), (1,0), (0,1), (1,1)]:
        probe.update_BL(xi, eta)
        exx.append(np.dot(probe.BLexx, probe.ue))
        eyy.append(np.dot(probe.BLeyy, probe.ue))
        gxy.append(np.dot(probe.BLgxy, probe.ue))
        kxx.append(np.dot(probe.BLkxx, probe.ue))
        kyy.append(np.dot(probe.BLkyy, probe.ue))
        kxy.append(np.dot(probe.BLkxy, probe.ue))
        gxz_rot = np.dot(probe.BLgxz_rot, probe.ue)
        gxz_grad = np.dot(probe.BLgxz_grad, probe.ue)
        gxz.append(gxz_rot+gxz_grad)
        gyz_rot = np.dot(probe.BLgyz_rot, probe.ue)
        gyz_grad = np.dot(probe.BLgyz_grad, probe.ue)
        gyz.append(gyz_rot+gyz_grad)

        Nxx.append(A11*exx[-1]+A12*eyy[-1]+A16*gxy[-1]+B11*kxx[-1]+B12*kyy[-1]+B16*kxy[-1])
        Nyy.append(A12*exx[-1]+A22*eyy[-1]+A26*gxy[-1]+B12*kxx[-1]+B22*kyy[-1]+B26*kxy[-1])
        Nxy.append(A16*exx[-1]+A26*eyy[-1]+A66*gxy[-1]+B16*kxx[-1]+B26*kyy[-1]+B66*kxy[-1])
        Mxx.append(D11*kxx[-1]+D12*kyy[-1]+D16*kxy[-1]+B11*exx[-1]+B12*eyy[-1]+B16*gxy[-1])
        Myy.append(D12*kxx[-1]+D22*kyy[-1]+D26*kxy[-1]+B12*exx[-1]+B22*eyy[-1]+B26*gxy[-1])
        Mxy.append(D16*kxx[-1]+D26*kyy[-1]+D66*kxy[-1]+B16*exx[-1]+B26*eyy[-1]+B66*gxy[-1])
        Qxz.append(E44*gxz[-1]+E45*gyz[-1])
        Qyz.append(E45*gxz[-1]+E55*gyz[-1])

    # linear interpolation coefficients

    Cexx = exx[0]
    Aexx = exx[1]-Cexx
    Bexx = exx[2]-Cexx

    Ceyy = eyy[0]
    Aeyy = eyy[1]-Ceyy
    Beyy = eyy[2]-Ceyy

    Cgxy = gxy[0]
    Agxy = gxy[1]-Cgxy
    Bgxy = gxy[2]-Cgxy

    Ckxx = kxx[0]
    Akxx = kxx[1]-Ckxx
    Bkxx = kxx[2]-Ckxx

    Ckyy = kyy[0]
    Akyy = kyy[1]-Ckyy
    Bkyy = kyy[2]-Ckyy

    Ckxy = kxy[0]
    Akxy = kxy[1]-Ckxy
    Bkxy = kxy[2]-Ckxy

    Cgxz = gxz[0]
    Agxz = gxz[1]-Cgxz
    Bgxz = gxz[2]-Cgxz

    Cgyz = gyz[0]
    Agyz = gyz[1]-Cgyz
    Bgyz = gyz[2]-Cgyz

    CNxx = Nxx[0]
    ANxx = Nxx[1]-CNxx
    BNxx = Nxx[2]-CNxx

    CNyy = Nyy[0]
    ANyy = Nyy[1]-CNyy
    BNyy = Nyy[2]-CNyy

    CNxy = Nxy[0]
    ANxy = Nxy[1]-CNxy
    BNxy = Nxy[2]-CNxy

    CMxx = Mxx[0]
    AMxx = Mxx[1]-CMxx
    BMxx = Mxx[2]-CMxx

    CMyy = Myy[0]
    AMyy = Myy[1]-CMyy
    BMyy = Myy[2]-CMyy

    CMxy = Mxy[0]
    AMxy = Mxy[1]-CMxy
    BMxy = Mxy[2]-CMxy

    CQxz = Qxz[0]
    AQxz = Qxz[1]-CQxz
    BQxz = Qxz[2]-CQxz

    CQyz = Qyz[0]
    AQyz = Qyz[1]-CQyz
    BQyz = Qyz[2]-CQyz

    return {
        "exx":lambda xi, eta: Aexx*xi+Bexx*eta+Cexx,
        "eyy":lambda xi, eta: Aeyy*xi+Beyy*eta+Ceyy,
        "gxy":lambda xi, eta: Agxy*xi+Bgxy*eta+Cgxy,
        "kxx":lambda xi, eta: Akxx*xi+Bkxx*eta+Ckxx,
        "kyy":lambda xi, eta: Akyy*xi+Bkyy*eta+Ckyy,
        "kxy":lambda xi, eta: Akxy*xi+Bkxy*eta+Ckxy,
        "gxz":lambda xi, eta: Agxz*xi+Bgxz*eta+Cgxz,
        "gyz":lambda xi, eta: Agyz*xi+Bgyz*eta+Cgyz,

        "Nxx":lambda xi, eta: ANxx*xi+BNxx*eta+CNxx,
        "Nyy":lambda xi, eta: ANyy*xi+BNyy*eta+CNyy,
        "Nxy":lambda xi, eta: ANxy*xi+BNxy*eta+CNxy,
        "Mxx":lambda xi, eta: AMxx*xi+BMxx*eta+CMxx,
        "Myy":lambda xi, eta: AMyy*xi+BMyy*eta+CMyy,
        "Mxy":lambda xi, eta: AMxy*xi+BMxy*eta+CMxy,
        "Qxz":lambda xi, eta: AQxz*xi+BQxz*eta+CQxz,
        "Qyz":lambda xi, eta: AQyz*xi+BQyz*eta+CQyz,
    }


    

        
        