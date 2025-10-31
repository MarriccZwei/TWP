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
    exy = list()
    kxx = list()
    kyy = list()
    kxy = list()
    exz = list()
    eyz = list()

    Nxx = list()
    Nyy = list()
    Nxy = list()
    Mxx = list()
    Myy = list()
    Mxy = list()
    Qxz = list()
    Qyz = list()


    for xi, eta in [(0,0), (1,0), (0,1)]:
        probe.update_BL(xi, eta)
        exx.append(np.dot(probe.BLexx, probe.ue))
        eyy.append(np.dot(probe.BLeyy, probe.ue))
        #TODO: rest

        Nxx.append(shellprop.A11*exx[-1]+shellprop.A12*eyy[-1]) #TODO: add b terms
        Nyy.append(shellprop.A12*exx[-1]+shellprop.A22*eyy[-1])

    #linear interpolation coefficients
    Cexx = exx[0]
    Aexx = exx[1]-Cexx
    Bexx = exx[2]-Cexx

    CNxx = Nxx[0]
    ANxx = Nxx[1]-CNxx
    BNxx = Nxx[2]-CNxx

    return {
        "exx":lambda xi, eta: Aexx*xi+Bexx*eta+Cexx,
        "Nxx":lambda xi, eta: ANxx*xi+BNxx*eta+CNxx,
    }

    

        
        