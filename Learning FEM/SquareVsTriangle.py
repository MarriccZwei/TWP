import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import coo_matrix
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

from pyfe3d.shellprop_utils import isotropic_plate
from pyfe3d import Quad4, Quad4Data, Quad4Probe, INT, DOUBLE, DOF

'''Comparing the braizer effect on two beams of length L and side length of a, adjusted for same Ixx. One with a square cross ection, and the other with equilateral triangle cross section. 
The braizer deformation shall be measured as change of height at the vertical symmetry line of the beam'''
L = 6
a = 0.6
E = 200e9
h = 0.01
nu = 0.3
na = 19 #number of nodes along a. Corners included. Should be odd. Quadrangle will get a 0.75 n
nL = int(1*na) #number of elements along the length
M = 3000 #bending moment applied
#standard beam coordinate system! y down, z towards free end, x sideways for positively oriented coord!
plot = True

f'''{"Triangle"}==============================================='''
"1) geometry"
xtmp = np.array(np.linspace(-a/2, a/2, na).tolist()+np.linspace(a/2, 0, na).tolist()[1:]+np.linspace(0, -a/2, na)[1:-1].tolist())
ytmp = np.array(np.zeros(na).tolist()+np.linspace(0, a*np.sqrt(3)/2, na)[1:].tolist()+np.linspace(a*np.sqrt(3)/2, 0, na).tolist()[1:-1])

#==CHECKS== 
assert len(xtmp) == len(ytmp) == 3*na-3 #check if we don't double count element
# plt.plot(xtmp, ytmp)
# plt.show()

ztmp = np.linspace(0, L, nL)
#now we have all three dimension that vary, we can just do two meshgrids
xmesh, zmesh = np.meshgrid(xtmp, ztmp)
ymesh, _ = np.meshgrid(ytmp, ztmp)
ncoords = np.vstack((xmesh.T.flatten(), ymesh.T.flatten(), zmesh.T.flatten())).T
x = ncoords[:, 0]
y = ncoords[:, 1]
z = ncoords[:, 2]
ncoords_flatten = ncoords.flatten()

#==CHECKS== 
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# plt.plot(x,y,z)
# plt.show()
f'''{"RESULT"} it completes the mesh by going all the way in z for each of the nodes along the triangle'''

"2) element creation"
nids = 1+np.arange(ncoords.shape[0]) #ids of nodes
nid_pos = dict(zip(nids, np.arange(len(nids)))) #positions of nodes on the id list
nids_mesh = nids.reshape(3*na-3, nL)
#now we have to apply the periodic BCE
shifted_nids_mesh = np.vstack((nids_mesh[1:, :], nids_mesh[0:1, :]))
n1s = nids_mesh[:, :-1].flatten()
n2s = shifted_nids_mesh[:, :-1].flatten()
n3s = shifted_nids_mesh[:, 1:].flatten()
n4s = nids_mesh[:, 1:].flatten()

num_elements = (3*na-3)*(nL-1) #defining and verifying the number of elements
assert num_elements == len(n1s) == len(n2s) ==len(n3s) == len(n4s)

#COPYPASTA from the example file==============================================================
data = Quad4Data()
probe = Quad4Probe()

KC0r = np.zeros(data.KC0_SPARSE_SIZE*num_elements, dtype=INT)
KC0c = np.zeros(data.KC0_SPARSE_SIZE*num_elements, dtype=INT)
KC0v = np.zeros(data.KC0_SPARSE_SIZE*num_elements, dtype=DOUBLE)
N = DOF*(3*na-3)*nL

prop = isotropic_plate(thickness=h, E=E, nu=nu, calc_scf=True) #not needed for smeard

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
    #now the eles will be oriented in diff directions, idk if we need to change some of them
    #TODO: Ask Saullo abt the proper way to do it
    #assert normal > 0
    quad = Quad4(probe)
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
#END COPYPASTA===================================================================================

"3) boundary conditions other than peridical - fixed at z=0"
bk = np.zeros(N, dtype=bool)
check = np.isclose(z, 0.) #z is z of all points, that's why the dimension is correct
#fixing all 6 DOF at the boundary
for i in range(6):
    bk[i::DOF] = check
bu = ~bk #the boundary to switched off

"4) loads - assume a bending moment at each vertex"
f = np.zeros(N)
fmid = M/3
check = (np.isclose(x, a/2) | np.isclose(x, -a/2) | np.isclose(y, a*np.sqrt(3)/2)) & np.isclose(z, L) 
f[1::DOF][check] = fmid

"5) applying loads and BCEs"
KC0uu = KC0[bu, :][:, bu] #constrained stiffness matrix
fu = f[bu]
assert np.isclose(fu.sum(), M)


"6) solution"
uu = spsolve(KC0uu, fu)
u = np.zeros(N)
u[bu] = uu
w = u[1::DOF].reshape(3*(na-1), nL) #@# reformatting the result

"7) taking middle deflections"
wy0 = w[int((na-1)/2), :] #we want all points in z for the w in the middle at y=0, this will happen at half of first a segment
wym = w[2*na-2, :] #we want all points in z for the w in the middle at y=ymax, this after the first two segments, we substract 1 for a common pt and another 1 for indexing
if plot:
    plt.subplot(121)
    plt.plot(ztmp, wy0, label="flat side")
    plt.plot(ztmp, wym, label = "vertex side")
    plt.title(f"Deformation of the upper and lower side of the triangular beam.\n Max cross section deformation: {max(abs(wym-wy0))}")
    plt.legend()



f'''{"Square"}==============================================='''
"1) geometry"
na = int(.75*na)
na = na+1-na%2 #make even if odd
a = (3/8)**(1/3)*a #adjusting a for same Ixx
xtmp = np.array(np.linspace(-a/2, a/2, na).tolist()+(a/2*np.ones(na-2)).tolist()+np.linspace(a/2, -a/2, na).tolist()+(-a/2*np.ones(na-2)).tolist())
ytmp = (-a/2*np.ones(na-2)).tolist()+np.linspace(-a/2, a/2, na).tolist()+(a/2*np.ones(na-2)).tolist()+np.linspace(a/2, -a/2, na).tolist()
ytmp = np.array([ytmp[-1]]+ytmp[:-1]) #shifting ytmp so that we actually get a square, the above formulation is one node off

#==CHECKS== 
assert len(xtmp) == len(ytmp) == 4*na-4 #check if we don't double count element
# plt.figure()
# plt.plot(xtmp, ytmp)
# plt.show()
# print(np.linspace(-a/2, a/2, na).tolist())

ztmp = np.linspace(0, L, nL)
#now we have all three dimension that vary, we can just do two meshgrids
xmesh, zmesh = np.meshgrid(xtmp, ztmp)
ymesh, _ = np.meshgrid(ytmp, ztmp)
ncoords = np.vstack((xmesh.T.flatten(), ymesh.T.flatten(), zmesh.T.flatten())).T
x = ncoords[:, 0]
y = ncoords[:, 1]
z = ncoords[:, 2]
ncoords_flatten = ncoords.flatten()

#==CHECKS== 
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# plt.plot(x,y,z)
# plt.show()
f'''{"RESULT"} it completes the mesh by going all the way in z for each of the nodes along the triangle'''

"2) element creation"
nids = 1+np.arange(ncoords.shape[0]) #ids of nodes
nid_pos = dict(zip(nids, np.arange(len(nids)))) #positions of nodes on the id list
nids_mesh = nids.reshape(4*na-4, nL)
#now we have to apply the periodic BCE
shifted_nids_mesh = np.vstack((nids_mesh[1:, :], nids_mesh[0:1, :]))
n1s = nids_mesh[:, :-1].flatten()
n2s = shifted_nids_mesh[:, :-1].flatten()
n3s = shifted_nids_mesh[:, 1:].flatten()
n4s = nids_mesh[:, 1:].flatten()

num_elements = (4*na-4)*(nL-1) #defining and verifying the number of elements
assert num_elements == len(n1s) == len(n2s) ==len(n3s) == len(n4s)

#COPYPASTA from the example file==============================================================
data = Quad4Data()
probe = Quad4Probe()

KC0r = np.zeros(data.KC0_SPARSE_SIZE*num_elements, dtype=INT)
KC0c = np.zeros(data.KC0_SPARSE_SIZE*num_elements, dtype=INT)
KC0v = np.zeros(data.KC0_SPARSE_SIZE*num_elements, dtype=DOUBLE)
N = DOF*(4*na-4)*nL

prop = isotropic_plate(thickness=h, E=E, nu=nu, calc_scf=True) #not needed for smeard

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
    #now the eles will be oriented in diff directions, idk if we need to change some of them
    #TODO: Ask Saullo abt the proper way to do it
    #assert normal > 0
    quad = Quad4(probe)
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
#END COPYPASTA===================================================================================

"3) boundary conditions other than peridical - fixed at z=0"
bk = np.zeros(N, dtype=bool)
check = np.isclose(z, 0.) #z is z of all points, that's why the dimension is correct
#fixing all 6 DOF at the boundary
for i in range(6):
    bk[i::DOF] = check
bu = ~bk #the boundary to switched off

"4) loads - assume a bending moment at each vertex"
f = np.zeros(N)
fmid = M/4 #so the same bending moment is applied
check = (np.isclose(x, a/2) | np.isclose(x, -a/2)) & np.isclose(z, L) & (np.isclose(y, a/2) | np.isclose(y, -a/2)) 
f[1::DOF][check] = fmid

"5) applying loads and BCEs"
KC0uu = KC0[bu, :][:, bu] #constrained stiffness matrix
fu = f[bu]
assert np.isclose(fu.sum(), M)


"6) solution"
uu = spsolve(KC0uu, fu)
u = np.zeros(N)
u[bu] = uu
w = u[1::DOF].reshape(4*(na-1), nL) #@# reformatting the result

"7) taking middle deflections"
wy0 = w[int((na-1)/2), :] #we want all points in z for the w in the middle at y=0, this will happen at half of first a segment
wym = w[2*na-2+int((na-1)/2), :] #the top of the rectangle is 2 nas -2 away (half of the indices)
if plot:
    plt.subplot(122)
    plt.plot(ztmp, wy0, label="y- side")
    plt.plot(ztmp, wym, label = "y+ side")
    plt.title(f"Deformation of the upper and lower side of the tquadrangular beam.\n Max cross section deformation: {max(abs(wym-wy0))}")
    plt.legend()


"""Showing the plot once both subplots are done"""
if plot:
    plt.show()