import numpy as np
import Analytical as an
import pyfe3d as fem
import pyfe3d.shellprop_utils as spu
from scipy.sparse import coo_matrix

"bending stresses obtained with pfe3D to be compared with the analytical case"
class Sheet():
    def __init__(self, wfix:float, wfree:float, length:float, h:float, stiffener:an.RealStiffener, nstiff:int):
        self.wfix = wfix
        self.wfree = wfree
        self.length = length
        self.h = h #sheet thickness, abbreviated due to being used many times
        self.stiffener = stiffener
        self.nstiff = nstiff

    def default_mesh(self, nw, nl): #numbers of elements in width-wise and chord-wise dirs
        ncoordslist = list()
        for y in np.linspace(0, self.length, nl+1):
            zOverLen = y/self.length
            w = self.wfix*(1-zOverLen)+zOverLen*self.wfree #linear interpol between the edge widths
            for x in np.linspace(-w/2, w/2, nw+1):
                ncoordslist.append(np.array([x, y, 0]))
        #applying pyfe3d
        ncoords = np.array(ncoordslist)
        x = ncoords[:, 0]
        y = ncoords[:, 1]
        z = ncoords[:, 2]
        ncoords_flatten = ncoords.flatten()
        return x, y, z, ncoords, ncoords_flatten
    
if __name__ == "__main__":
    E=1
    nu = .333

    length = 10
    stiff = an.RealLStiffener(0.05, 0.005)
    sheet = Sheet(1, .7, length, 0.01, stiff, 5)

    nw = 20
    nl = 35
    x, y, z, ncoords, ncoords_flatten = sheet.default_mesh(20, 35)

    #getting corners and node ids
    nids = 1 + np.arange(ncoords.shape[0])
    nid_pos = dict(zip(nids, np.arange(len(nids))))
    nids_mesh = nids.reshape(nl, nw) #what is the inner what is the outer dim?
    #in the original is x const for changing y we have the other way around, so indeed the first shape is nl
    n1s = nids_mesh[:-1, :-1].flatten()
    n2s = nids_mesh[1:, :-1].flatten()
    n3s = nids_mesh[1:, 1:].flatten()
    n4s = nids_mesh[:-1, 1:].flatten()
    num_elements = len(n1s)

    #preparing the matrices
    data = fem.Quad4Data()
    KC0r = np.zeros(data.KC0_SPARSE_SIZE*num_elements, dtype=fem.INT)
    KC0c = np.zeros(data.KC0_SPARSE_SIZE*num_elements, dtype=fem.INT)
    KC0v = np.zeros(data.KC0_SPARSE_SIZE*num_elements, dtype=fem.DOUBLE)
    N = fem.DOF*nw*nl

    prop = spu.isotropic_plate(thickness=sheet.h, E=E, nu=nu, calc_scf=True) #TODO: change this line for smeard
    probe = fem.Quad4Probe()

    quads = []
    init_k_KC0 = 0
    for n1, n2, n3, n4 in zip(n1s, n2s, n3s, n4s):
        pos1 = nid_pos[n1]
        pos2 = nid_pos[n2]
        pos3 = nid_pos[n3]
        pos4 = nid_pos[n4]
        r1 = ncoords[pos1]
        r2 = ncoords[pos2]
        r3 = ncoords[pos3]
        normal = np.cross(r2 - r1, r3 - r2)[2]
        assert normal > 0
        quad = fem.Quad4(probe)
        quad.n1 = n1 #@# connectring it to the global stiffness matrix
        quad.n2 = n2
        quad.n3 = n3
        quad.n4 = n4
        quad.c1 = DOF*nid_pos[n1]
        quad.c2 = DOF*nid_pos[n2]
        quad.c3 = DOF*nid_pos[n3]
        quad.c4 = DOF*nid_pos[n4]
        quad.init_k_KC0 = init_k_KC0
        quad.update_rotation_matrix(ncoords_flatten)
        quad.update_probe_xe(ncoords_flatten)
        quad.update_KC0(KC0r, KC0c, KC0v, prop) #matrix contribution, changing the matrices sent
        quads.append(quad)
        init_k_KC0 += data.KC0_SPARSE_SIZE

    KC0 = coo_matrix((KC0v, (KC0r, KC0c)), shape=(N, N)).tocsc() #@# stiffness matrix in sparse format

    print('elements created')