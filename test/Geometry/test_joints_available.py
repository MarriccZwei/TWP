from ...C4.Geometry.joints_available import JointsAvailable, Joint

import numpy as np
import matplotlib.pyplot as plt

def test_Joint():
    j = Joint(1/4, 1.25, 4080, 7360/2, bearing_ratio=.5, rho_bolt=7750.) #the smallest bolt from the dataset
    H = .03
    lbf_to_N = 4.44822

    #case 1: shear load = 3*allowable in bearing
    n = j.get_joint_n(0., 7360*.75*lbf_to_N/2)
    assert n == 4, n
    rj = j.get_joint_dims(n)
    assert np.isclose(rj, .04445), rj
    mj = j.get_joint_mass(n, H, rj)
    assert np.isclose(mj, .065381), mj

    #case 2: normal load = 2*allowable in tension + .2 allowable in fastener shear
    n = j.get_joint_n(4080*2.*lbf_to_N/2, 7360*.1*lbf_to_N/2)
    assert n == 4, n
    rj = j.get_joint_dims(n)
    assert np.isclose(rj, .04445), rj
    mj = j.get_joint_mass(n, H, rj)
    assert np.isclose(mj, .065381), mj

    #case 3: no loads (should be 2 bolts, smallest possible)
    n = j.get_joint_n(0, 0)
    assert n == 2, n
    rj = j.get_joint_dims(n)
    assert np.isclose(rj, .04445), rj
    mj = j.get_joint_mass(n, H, rj)
    assert np.isclose(mj, .0338374581), mj

    #case 4: 4 bolts in a row
    j.nrows = 4
    n = j.get_joint_n(0., 7360*.75*lbf_to_N/2)
    assert n == 4, n
    rj = j.get_joint_dims(n)
    assert np.isclose(rj, .08255), rj
    mj = j.get_joint_mass(n, H, rj)
    assert np.isclose(mj, .0663645), mj


def test_AvailableJoints():
    #case 1: increasing the shear load fraction with load magn staying same. expected: actually nothing happens because of the way in which the bearing load is defined
    loadfrac = np.linspace(0, 1., 11)*np.pi/2
    magn = 1e4
    N = magn*np.cos(loadfrac)
    V = magn*np.sin(loadfrac)
    H = .05
    joints = [JointsAvailable.size_joint(Ni, Vi, H, True) for Ni, Vi in zip(N, V)]
    ns = [joint[1] for joint in joints]
    rj = [joint[2] for joint in joints]
    sheet_t = [joint[0].t_sheet for joint in joints]

    bolt_d = [joint[0].d_bolt for joint in joints]

    plt.plot(loadfrac, np.array(ns)/1000, label="n/1000")
    plt.plot(loadfrac, rj, label="rj")
    plt.plot(loadfrac, sheet_t, label="sheet_t")
    plt.plot(loadfrac, bolt_d, label="bolt_d")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    test_Joint()
    test_AvailableJoints()