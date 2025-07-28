import numpy as np
import pyfe3d as pf3
import scipy.sparse as ss
import typing as ty
import geometricClasses as gcl
import pyfe3d.beamprop as pbp
import pyfe3d.shellprop_utils as psp
import copy

"""The file containing an interpreter of the gcl connection matrix into pyfe3D elements"""
PM_SPARSE_SIZE = 3 #point mass sparse size
IN_SPARSE_SIZE = 6 #inertia mass sparse size

class OrientedBeamProp(pbp.BeamProp):
    def __init__(self, xyex, xyey, xyez):
        super().__init__()
        self.xyex = xyex
        self.xyey = xyey
        self.xyez = xyez

class BeamCWithProp(pf3.BeamC):
    def __init__(self, probe:OrientedBeamProp):
        super().__init__()
        self.probe=probe
        self.beamprop = None

class QuadWithProp(pf3.Quad4):
    def __init__(self, probe:pf3.Quad4Probe):
        super().__init__()
        self.probe = probe
        self.shellprop = None

class SpringData(): #spring data with reserved mass matrix for mass at nodes
    def __init__(self):
        self.KC0_SPARSE_SIZE = 72
        self.KG_SPARSE_SIZE = 0
        self.M_SPARSE_SIZE = 6 #2*3 masses for translational DOF

class SpringProp(): #a spring prop class missing from pyfe3d with added rotation matrix arguments and mass to be spread between nodes
    def __init__(self, kxe, kye=0., kze=0., krxe=0., krye=0., krze=0., xex=1., xey=0., xez=0., xyex=1., xyey=1., xyez=0., m=0.):
        self.kxe = kxe
        self.kye = kye
        self.kze = kze
        self.krxe = krxe
        self.krye = krye
        self.krze = krze
        self.xex = xex
        self.xey = xey
        self.xez = xez
        self.xyex = xyex
        self.xyex = xyex
        self.xyey = xyey
        self.xyez = xyez
        self.nodem = m/2 #half of the spring mass shall go to each node

    @property
    def stiffs(self): #to make assingning stiffnesses less ugly
        return self.kxe, self.kye, self.kze, self.krxe, self.krye, self.krze
    
    @property
    def rots(self): #to make passing args to update_rotation_matrix less ugly
        return self.xex, self.xey, self.xez, self.xyex, self.xyey, self.xyez
    
class Spring(pf3.Spring):
    def __init__(self, probe:pf3.SpringProbe):
        super(Spring, self).__init__()
        self.probe=probe

    def update_M(self, Mr, Mc, Mv, prop:SpringProp, init_M):
        k = init_M
        Mr[k] = 0+self.c1
        Mc[k] = 0+self.c1
        Mv[k] = prop.nodem
        k+=1
        Mr[k] = 1+self.c1
        Mc[k] = 1+self.c1
        Mv[k] = prop.nodem
        k+=1
        Mr[k] = 2+self.c1
        Mc[k] = 2+self.c1
        Mv[k] = prop.nodem
        k+=1
        Mr[k] = 0+self.c2
        Mc[k] = 0+self.c2
        Mv[k] = prop.nodem
        k+=1
        Mr[k] = 1+self.c2
        Mc[k] = 1+self.c2
        Mv[k] = prop.nodem
        k+=1
        Mr[k] = 2+self.c2
        Mc[k] = 2+self.c2
        Mv[k] = prop.nodem

def eles_from_gcl(mesh:gcl.Mesh3D, eleDict:ty.Dict[str, ty.Dict[str, object]]):
    #currently only for the elements we have used, same goes to gcl connections
    #probes = {"quad": pf3.Quad4Probe(), "spring": pf3.SpringProbe(), "beam":pf3.BeamCProbe(), "mass":None}
    data = {"quad": pf3.Quad4Data(), "spring":SpringData(), "beam":pf3.BeamCData(), "mass":None}
    created_eles = {"quad":[], "spring":[], "beam":[], "mass":[]}

    #obtaining nodes geometry
    ncoords, x, y, z, ncoords_flatten = mesh.pyfe3D()

    #converting to pyfe3D id system
    nids = 1 + np.arange(ncoords.shape[0])
    nid_pos = dict(zip(nids, np.arange(len(nids))))

    #calculating matrix size based on element data sizes
    sparse_size = 0
    mass_size = 0
    kg_size = 0
    for key, val in mesh.connections.items():
        if key!="mass": #mass does not contribute here, it's assigned to the mass matrix
            sparse_size += data[key].KC0_SPARSE_SIZE*len(val)
            mass_size += data[key].M_SPARSE_SIZE*len(val)
            kg_size += data[key].KG_SPARSE_SIZE*len(val)
        else:
            mass_size += PM_SPARSE_SIZE*len(val) #a mass in the translational DOF on the diagonal
    mass_size += IN_SPARSE_SIZE*len(mesh.inertia)

    #initialising system matrices
    KC0r = np.zeros(sparse_size, dtype=pf3.INT)
    KC0c = np.zeros(sparse_size, dtype=pf3.INT)
    KC0v = np.zeros(sparse_size, dtype=pf3.DOUBLE)
    Mr = np.zeros(mass_size, dtype=pf3.INT)
    Mc = np.zeros(mass_size, dtype=pf3.INT)
    Mv = np.zeros(mass_size, dtype=pf3.DOUBLE)
    KGr = np.zeros(kg_size, dtype=pf3.INT)
    KGc = np.zeros(kg_size, dtype=pf3.INT)
    KGv = np.zeros(kg_size, dtype=pf3.DOUBLE) 
    N = pf3.DOF*len(mesh.nodes)

    #Element creation based on the connection matrix
    init_k_KC0 = 0
    init_k_M = 0
    init_k_KG = 0
    #Quad elements
    for conn in mesh.connections["quad"]:
        if conn.protocol=="":
            shellprop = eleDict["quad"][conn.eleid]
        else:
            shellprop = eleDict["quad"][conn.eleid](conn.protocol)
        quad = QuadWithProp(pf3.Quad4Probe())
        quad.shellprop = shellprop
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
        quad.update_KG(KGr, KGc, KGv, shellprop)
        created_eles["quad"].append(quad)
        init_k_KC0 += data["quad"].KC0_SPARSE_SIZE
        init_k_M += data["quad"].M_SPARSE_SIZE
        init_k_KG += data["quad"].KG_SPARSE_SIZE
    #Spring elements
    for conn in mesh.connections["spring"]:
        spring = Spring(pf3.SpringProbe())
        spring.n1 = nids[conn.ids[0]]
        spring.n2 = nids[conn.ids[1]]
        spring.c1 = pf3.DOF*nid_pos[spring.n1]
        spring.c2 = pf3.DOF*nid_pos[spring.n2]
        if conn.protocol=="":
            sprop = eleDict["spring"][conn.eleid] #has to be in a SpringProp format
            spring.kxe, spring.kye, spring.kze, spring.krxe, spring.krye, spring.krze = sprop.stiffs
        else:
            sprop = eleDict["spring"][conn.eleid](conn.protocol)
            spring.kxe, spring.kye, spring.kze, spring.krxe, spring.krye, spring.krze = sprop.stiffs
        spring.init_k_KC0 = init_k_KC0
        spring.init_k_M = init_k_M
        spring.update_rotation_matrix(*sprop.rots)
        spring.update_KC0(KC0r, KC0c, KC0v)
        spring.update_M(Mr, Mc, Mv, sprop, init_k_M)
        created_eles["spring"].append(spring)
        init_k_KC0 += data["spring"].KC0_SPARSE_SIZE
        init_k_M += data["spring"].M_SPARSE_SIZE
        init_k_KG += data["spring"].KG_SPARSE_SIZE
    #Beam elements
    for conn in mesh.connections["beam"]:
        if conn.protocol=="":
            prop = eleDict["beam"][conn.eleid] #prop is an OrientedBeamProp
        else:
            prop = eleDict["beam"][conn.eleid](conn.protocol)
        beam = BeamCWithProp(pf3.BeamCProbe())
        beam.beamprop = prop
        beam.n1 = nids[conn.ids[0]]
        beam.n2 = nids[conn.ids[1]]
        pos1 = nid_pos[beam.n1]
        pos2 = nid_pos[beam.n2]
        beam.init_k_KC0 = init_k_KC0
        beam.init_k_M = init_k_M
        beam.c1 = pf3.DOF*pos1
        beam.c2 = pf3.DOF*pos2
        beam.update_rotation_matrix(prop.xyex, prop.xyey, prop.xyez, ncoords_flatten) #OrientedBeamProp features used here
        beam.update_probe_xe(ncoords_flatten)
        beam.update_KC0(KC0r, KC0c, KC0v, prop)
        beam.update_M(Mr, Mc, Mv, prop) #TODO: mtype?
        beam.update_KG(KGr, KGc, KGv, prop, 1)
        created_eles["beam"].append(beam)
        init_k_KC0 += data["beam"].KC0_SPARSE_SIZE
        init_k_M += data["beam"].M_SPARSE_SIZE
        init_k_KG += data["beam"].KG_SPARSE_SIZE
    #Point masses
    for conn in mesh.connections["mass"]:
        if conn.protocol=="":
            mass = eleDict["mass"][conn.eleid]
        else:
            mass = eleDict["mass"][conn.eleid](conn.protocol)
        #accessing the mass matrix directly
        k = init_k_M
        id_ = conn.ids[0]
        pos = pf3.DOF*id_ #position based on the id of the element
        Mr[k] = 0+pos
        Mc[k] = 0+pos
        Mv[k] += mass
        k+=1
        Mr[k] = 1+pos
        Mc[k] = 1+pos
        Mv[k] += mass
        k+=1
        Mr[k] =2+pos
        Mc[k] = 2+pos
        Mv[k] += mass
        init_k_M += PM_SPARSE_SIZE
    #Inertia
    for inco in mesh.inertia:
        k = init_k_M
        pos = pf3.DOF*inco.id_
        Mr[k] = 0+pos
        Mc[k] = 0+pos
        Mv[k] += inco.mn
        k+=1
        pos = pf3.DOF*inco.id_
        Mr[k] = 1+pos
        Mc[k] = 1+pos
        Mv[k] += inco.mn
        k+=1
        pos = pf3.DOF*inco.id_
        Mr[k] = 2+pos
        Mc[k] = 2+pos
        Mv[k] += inco.mn
        k+=1
        pos = pf3.DOF*inco.id_
        Mr[k] = 3+pos
        Mc[k] = 3+pos
        Mv[k] += inco.Jxx
        k+=1
        pos = pf3.DOF*inco.id_
        Mr[k] = 4+pos
        Mc[k] = 4+pos
        Mv[k] += inco.Jyy
        k+=1
        pos = pf3.DOF*inco.id_
        Mr[k] = 5+pos
        Mc[k] = 5+pos
        Mv[k] += inco.Jzz
        init_k_M += IN_SPARSE_SIZE

    #TODO: Other eles

    KC0 = ss.coo_matrix((KC0v, (KC0r, KC0c)), shape=(N, N)).tocsc() #@# stiffness matrix in sparse format
    M = ss.coo_matrix((Mv, (Mr, Mc)), shape=(N,N)).tocsc() #mass_matrix
    '''!!! One initialises KG only in post-processsing!!!'''
    #KG = ss.coo_matrix((KGv, (KGr, KGc)), shape=(N,N)).tocsc()

    return KC0, M, N, x, y, z, {'data':data, 'elements':created_eles, 'nids':nids, 
                             'nid_pos':nid_pos, 'ncoords_flatten':ncoords_flatten, 'ncoords':ncoords,
                             'KGv':KGv, 'KGr':KGr, 'KGc':KGc}

def weight(M:ss.coo_matrix, g:float, N:int, DOF:int, wd:gcl.Direction3D):
    diag = M.diagonal()
    Wvec = np.zeros(N)
    Wvec[0::DOF] = g*wd.x*diag[0::DOF]
    Wvec[1::DOF] = g*wd.y*diag[1::DOF]
    Wvec[2::DOF] = g*wd.z*diag[2::DOF]
    return Wvec

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

    #beam through the middle of the y coord
    beamnodeids = list(idgrid1[:, 10])
    mesh.beam_interconnect(beamnodeids, "beam1")
    beamnodeids = list(idgrid2[:, 10])
    mesh.beam_interconnect(beamnodeids, "beam1")
    mesh.pointmass_attach(27, "m1")
    mesh.inertia_smear(31, gcl.Point3D(-5, 5, 1), gcl.Point3D(-5, 4, -1), gcl.Point3D(0, 6, 1))

    #checking the mesh
    import matplotlib.pyplot as plt 
    ax = plt.subplot(121, projection='3d')
    # for conn in mesh.connections["quad"]:
    #     pts = [mesh.nodes[i] for i in conn.ids]
    #     x_, y_, _ = gcl.pts2coords3D(pts)
    #     plt.plot(x_, y_)
    # for conn in mesh.connections["spring"]:
    #     pts = [mesh.nodes[i] for i in conn.ids]
    #     x_, y_, _ = gcl.pts2coords3D(pts)
    #     plt.scatter(x_, y_)
    mesh.visualise(ax, True)

    #standard test data for the elements
    h = 0.005 # m
    E = 200e9
    nu = 0.3
    rho = 7.83e3 # kg/m3

    #copypasted beam prop
    prop = OrientedBeamProp(1, 1,0)
    b = 0.05 # m
    hb = 0.05
    A = hb*b
    Izz = b*hb**3/12
    Iyy = b**3*hb/12
    prop.A = A
    prop.E = E
    scf = 5/6.
    prop.G = scf*E/2/(1+0.3)
    prop.Izz = Izz
    prop.Iyy = Iyy
    prop.intrho = rho*A
    prop.intrhoy2 = rho*Izz
    prop.intrhoz2 = rho*Iyy
    prop.J = Izz + Iyy

    #dictionaries for the element types
    import pyfe3d.shellprop_utils as psu
    ele_prop = {'quad':{'quad1':psu.isotropic_plate(thickness=h, E=E, nu=nu, rho=rho, calc_scf=True)},
                'spring':{'spring1':SpringProp(1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 0,0,1, 0,1,1)}, 'beam':{'beam1':prop}, 'mass':{'m1':20}}

    #exporting mesh to pyfe3D
    KC0, M, N, x, y, z, _ = eles_from_gcl(mesh, ele_prop)

    '''#copypast of applying loads etc'''
    print('elements created')
    print(max(M.diagonal()), (M.diagonal()).argmax())

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
    fmid = 5.
    check = np.isclose(x, 0) & np.isclose(y, 5)
    f[2::pf3.DOF][check] = fmid/np.count_nonzero(check) #applying the point loads on the node close to the middle
    assert np.isclose(f.sum(), fmid)

    #applying weight
    f+=weight(M, 9.81, N, pf3.DOF, gcl.Direction3D(0,0,1))
    
    KC0uu = KC0[bu, :][:, bu]
    # KC0uk = KC0[bu, :][:, bk]
    fu = f[bu]#-KC0uk@uk #uk would be with precribed displacements


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


