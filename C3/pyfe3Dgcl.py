import numpy as np
import pyfe3d as pf3
import scipy.sparse as ss
import typing as ty
import geometricClasses as gcl

"""The file containing an interpreter of the gcl connection matrix into pyfe3D elements"""

def eles_from_gcl(mesh:gcl.Mesh3D, eleDict:ty.Dict[str, ty.Dict[str, object]]):
    #currently only for the elements we have used, same goes to gcl connections
    probes = {"quad": pf3.Quad4Probe(), "spring": pf3.SpringProbe(), "beam":pf3.BeamCProbe(), "mass":None}
    data = {"quad": pf3.Quad4Data(), "spring": pf3.SpringData(), "beam":pf3.BeamCData(), "mass":None}
    created_eles = {"quad":[], "spring":[], "beam":[], "mass":[]}

    #obtaining nodes geometry
    ncoords, x, y, z, ncoords_flatten = mesh.pyfe3D()

    #converting to pyfe3D id system
    nids = 1 + np.arange(ncoords.shape[0])
    nid_pos = dict(zip(nids, np.arange(len(nids))))

    #calculating matrix size based on element data sizes
    sparse_size = 0
    mass_size = 0
    for key, val in mesh.connections.items():
        if key!="mass": #mass does not contribute here, it's assigned to the mass matrix
            sparse_size += data[key].KC0_SPARSE_SIZE*len(val)
            mass_size += data[key].M_SPARSE_SIZE*len(val)

    #initialising system matrices
    KC0r = np.zeros(sparse_size, dtype=pf3.INT)
    KC0c = np.zeros(sparse_size, dtype=pf3.INT)
    KC0v = np.zeros(sparse_size, dtype=pf3.DOUBLE)
    Mr = np.zeros(mass_size, dtype=pf3.INT)
    Mc = np.zeros(mass_size, dtype=pf3.INT)
    Mv = np.zeros(mass_size, dtype=pf3.DOUBLE)
    N = pf3.DOF*len(mesh.nodes)

    #Element creation based on the connection matrix
    init_k_KC0 = 0
    init_k_M = 0
    #Quad elements
    for conn in mesh.connections["quad"]:
        shellprop = eleDict["quad"][conn.eleid]
        quad = pf3.Quad4(probes["quad"])
        quad.n1 = nids[conn.ids[0]]
        quad.n2 = nids[conn.ids[1]]
        quad.n3 = nids[conn.ids[2]]
        quad.n4 = nids[conn.ids[3]]
        quad.c1 = pf3.DOF*nid_pos[quad.n1]
        quad.c2 = pf3.DOF*nid_pos[quad.n2]
        quad.c3 = pf3.DOF*nid_pos[quad.n3]
        quad.c4 = pf3.DOF*nid_pos[quad.n4]
        quad.init_k_KC0 = init_k_KC0
        quad.init_k_M = init_k_M
        quad.update_rotation_matrix(ncoords_flatten)
        quad.update_probe_xe(ncoords_flatten)
        quad.update_KC0(KC0r, KC0c, KC0v, shellprop) #matrix contribution, changing the matrices sent
        quad.update_M(Mr, Mc, Mv, shellprop)
        created_eles["quad"].append(quad)
        init_k_KC0 += data["quad"].KC0_SPARSE_SIZE
        init_k_M += data["quad"].M_SPARSE_SIZE
    #Spring elements
    for conn in mesh.connections["spring"]:
        spring = pf3.Spring(probes["spring"])
        spring.n1 = nids[conn.ids[0]]
        spring.n2 = nids[conn.ids[1]]
        spring.c1 = pf3.DOF*nid_pos[spring.n1]
        spring.c2 = pf3.DOF*nid_pos[spring.n2]
        spring.kxe, spring.kye, spring.kze, spring.krxe, spring.krye, spring.krze = eleDict["spring"][conn.eleid]
        spring.init_k_KC0 = init_k_KC0
        spring.init_k_M = init_k_M
        spring.update_rotation_matrix(0, 1, 0, 1, 1, 0) #TODO: make this make sense
        spring.update_KC0(KC0r, KC0c, KC0v)
        #spring.update_M(Mr, Mc, Mv, shellprop)
        created_eles["spring"].append(spring)
        init_k_KC0 += data["spring"].KC0_SPARSE_SIZE
        init_k_M += data["spring"].M_SPARSE_SIZE
    #Beam elements
    for conn in mesh.connections["beam"]:
        beam = pf3.BeamC(probes["beam"])


    #TODO: Other eles

    KC0 = ss.coo_matrix((KC0v, (KC0r, KC0c)), shape=(N, N)).tocsc() #@# stiffness matrix in sparse format

    return KC0, N, x, y, z, {'probes': probes, 'data':data, 'elements':created_eles, 'nids':nids, 
                             'nid_pos':nid_pos, 'ncoords_flatten':ncoords_flatten, 'ncoords':ncoords}

if __name__ == "__main__":
    pts1 = [gcl.Point3D(-5, 0, 0), gcl.Point3D(-2.5, 10, 0)]
    ptsMid = [gcl.Point3D(0,0,0), gcl.Point3D(0, 10, 0)]
    pts2 = [gcl.Point3D(5, 0, 0), gcl.Point3D(2.5, 10, 0)]
    coords1 = gcl.multi_section_sheet3D(pts1, ptsMid, 5, [21])
    coords2 = gcl.multi_section_sheet3D(ptsMid, pts2, 6, [21])

    mesh = gcl.Mesh3D()
    #might be worth automating
    idspace1 = mesh.register(list(coords1.flatten()))
    idgrid1 = idspace1.reshape(coords1.shape)
    mesh.quad_interconnect(idgrid1, "quad1")
    idspace2 = mesh.register(list(coords2.flatten()))
    idgrid2 = idspace2.reshape(coords2.shape)
    mesh.quad_interconnect(idgrid2, "quad1")

    #springs at the boundary
    mesh.spring_connect(idgrid1[-1, :], idgrid2[0, :], "spring1")

    #checking the mesh
    import matplotlib.pyplot as plt
    plt.subplot(121)
    for conn in mesh.connections["quad"]:
        pts = [mesh.nodes[i] for i in conn.ids]
        x_, y_, _ = gcl.pts2coords3D(pts)
        plt.plot(x_, y_)
    for conn in mesh.connections["spring"]:
        pts = [mesh.nodes[i] for i in conn.ids]
        x_, y_, _ = gcl.pts2coords3D(pts)
        plt.scatter(x_, y_)

    #standard test data for the elements
    h = 0.005 # m
    E = 200e9
    nu = 0.3

    #dictionaries for the element types
    import pyfe3d.shellprop_utils as psu
    ele_prop = {'quad':{'quad1':psu.isotropic_plate(thickness=h, E=E, nu=nu, calc_scf=True)},
                'spring':{'spring1':(1e2, 1e2, 1e2, 1e2, 1e2, 1e2)}}

    #exporting mesh to pyfe3D
    KC0, N, x, y, z, _ = eles_from_gcl(mesh, ele_prop)

    '''#copypast of applying loads etc'''
    print('elements created')

    #@# boundary conditions
    #@# separating interior and boundary elements
    bk = np.zeros(N, dtype=bool)
    check = np.isclose(y, -4*x+20) | np.isclose(y, 4*x+20) | np.isclose(y, 0) | np.isclose(y, 10)
    bk[2::pf3.DOF] = check

    #constraining the displacements k-known DOF ~k <=> u - unknown DOF
    bk[0::pf3.DOF] = True
    bk[1::pf3.DOF] = True

    bu = ~bk

    # point load at center node @# this is how to apply loads
    f = np.zeros(N)
    fmid = 1.
    check = np.isclose(x, 0) & np.isclose(y, 5)
    f[2::pf3.DOF][check] = fmid #applying the point loads on the node close to the middle

    
    KC0uu = KC0[bu, :][:, bu]
    fu = f[bu]
    assert fu.sum() == 2*fmid

    #@# actually solving the fem model
    from scipy.sparse.linalg import spsolve
    uu = spsolve(KC0uu, fu)

    u = np.zeros(N)
    u[bu] = uu

    w = u[2::pf3.DOF].reshape(11, 21)

    #plotting
    plt.subplot(122)
    levels = np.linspace(w.min(), w.max(), 10)
    plt.contourf(x.reshape(11, 21), y.reshape(11, 21), w, levels=levels)
    plt.colorbar()

    plt.show()


