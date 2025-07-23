import meshingComponents as mc
import geometricClasses as gcl
import scipy.sparse.linalg as ssl
import ptsFromCAD as pfc

import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import pyfe3d.shellprop_utils as psp
import pyfe3Dgcl as p3g
import pyfe3d as pf3

meshg = gcl.Mesh3D()
data = "5000;0;-6.5|4750;0;-24|4500;0;-41|4000;0;-75|3500;0;-107|3000;0;-138|2500;0;-167|2000;0;-190|1500;0;-206|1250;0;-211|1000;0;-211.5|750;0;-205|500;0;-187.5|375;0;-173|250;0;-150.5|125;0;-113.5|62.5;0;-82.5|0;0;0|62.5;0;107.5|125;0;149.5|250;0;206.5|375;0;248|500;0;281.5|750;0;330.5|1000;0;363|1250;0;383.5|1500;0;394|2000;0;390|2500;0;362|3000;0;318|3500;0;259|4000;0;187.5|4500;0;104|4750;0;57|5000;0;6.5&4250;18000;2206.872|4125;18000;2198.122|4000;18000;2189.622|3750;18000;2172.622|3500;18000;2156.622|3250;18000;2141.122|3000;18000;2126.622|2750;18000;2115.122|2500;18000;2107.122|2375;18000;2104.622|2250;18000;2104.372|2125;18000;2107.622|2000;18000;2116.372|1937.5;18000;2123.622|1875;18000;2134.872|1812.5;18000;2153.372|1781.25;18000;2168.872|1750;18000;2210.122|1781.25;18000;2263.872|1812.5;18000;2284.872|1875;18000;2313.372|1937.5;18000;2334.122|2000;18000;2350.872|2125;18000;2375.372|2250;18000;2391.622|2375;18000;2401.872|2500;18000;2407.122|2750;18000;2405.122|3000;18000;2391.122|3250;18000;2369.122|3500;18000;2339.622|3750;18000;2303.872|4000;18000;2262.122|4125;18000;2238.622|4250;18000;2213.372&-471.576;3673.46;441.249|126.713;7770.33;945.522|725.002;11867.2;1449.796|1323.292;15964.069;1954.07&3012.266;18950.167;2324.854|4861.588;5721.895;702.56&500;0;0|1999.77;18000;2214.157&3500;0;0|3499.77;18000;2214.157&787.633;1600;66.192|2883.375;1600;66.192|2617.418;1600;526.843|1053.59;1600;526.843&2024.321;18000;2152.592|3273.646;18000;2152.592|3148.758;18000;2368.903|2149.209;18000;2368.903" 
up = pfc.UnpackedPoints(data)
sparsh, sparigrd, sparSecIdxs, a1, a2, f2 = mc.trigspars(meshg, 20, 5, 3, 1, 'q', up.ffb, up.frb, up.frt, up.fft, up.tfb, up.trb, up.trt, up.tft)
sparBotConns = sparSecIdxs[0::6]
sbb = [idx-1 for idx in sparBotConns]
sbbci = [sparigrd[:,idx] for idx in sbb]
sparTopConns = sparSecIdxs[3::6]
sbci = [sparigrd[:,idx] for idx in sparBotConns] #spar bottom connection ids
stci = [sparigrd[:,idx] for idx in sparTopConns] #spar top connection ids
beamids, batids = mc.bat_rail(meshg, 1, a1, a2, f2, .015, -.03, stci[0], "rl", "bt", "rm", "bm", 14000)
beamids1, batids1 = mc.bat_rail(meshg, 1, a1, a2, f2, .015, -.03, stci[1], "rl", "bt", "rm", "bm", 14000)
beamids2, batids2 = mc.bat_rail(meshg, 1, a1, a2, f2, .015, .03, sbbci[1], "rl", "bt", "rm", "bm", 14000)
botflsh, botsksh, botflids, botskids = mc.panel(meshg, up.ffb, up.frb, up.tfb, up.trb, 20, 5, 3, 
                                    'q', 'q', 'q', 'rb', up.surfb, sbci, 5, 6)

ax=plt.subplot(121, projection='3d')
meshg.visualise(ax)


#template for the flange prop
prop_flange = p3g.OrientedBeamProp(0, 0, 0)
prop_flange.A = .005*.005
prop_flange.Izz = .005*.005**3/12
prop_flange.Iyy = .005*.005**3/12
scf = 5/6 #ASSUMPTION, TODO: verify
prop_flange.G = scf*72e9/2/(1+0.3)
prop_flange.intrho = 2.7e3*prop_flange.A
prop_flange.intrhoy2 = 2.7e3*prop_flange.Izz
prop_flange.intrhoz2 = 2.7e3*prop_flange.Iyy
import copy
def prop_flange_or(dir): #oriented flange prop
    pfor = copy.deepcopy(prop_flange)
    pfor.xyex = dir[0]
    pfor.xyey = dir[1]
    pfor.xyez = dir[2]
    return pfor

INFTY_STIFF = 1e20
prop_rail = p3g.OrientedBeamProp(1, 0, 0)
prop_rail.A = np.pi/4*.02**2
prop_rail.Iyy = np.pi/64*.02**4
prop_rail.Izz = prop_rail.Iyy
prop_rail.J = prop_rail.Iyy+prop_rail.Izz
scf = 5/6 #ASSUMPTION, TODO: verify
prop_rail.G = scf*200e9/2/(1+0.3)
prop_rail.intrho = 7e3*prop_rail.A
prop_rail.intrhoy2 = 7e3*prop_rail.Izz
prop_rail.intrhoz2 = 7e3*prop_rail.Iyy
eleDict = {"quad":{"q":psp.isotropic_plate(thickness=.005, E=100e9, nu=.3, rho=3e3)}, "mass":{"bt":lambda m:m}, "beam":{'rl':prop_rail, 'rb':prop_flange_or},
           "spring":{"bm":p3g.SpringProp(INFTY_STIFF, INFTY_STIFF, INFTY_STIFF, INFTY_STIFF, INFTY_STIFF, INFTY_STIFF) ,
           "rm":p3g.SpringProp(INFTY_STIFF, INFTY_STIFF, INFTY_STIFF, INFTY_STIFF, INFTY_STIFF, INFTY_STIFF)}}
KC0, M, N, x, y, z, outdict  = p3g.eles_from_gcl(meshg, eleDict)

#bce
bk = np.zeros(N, dtype=bool)
check = np.isclose(y, 0)
for i in range(pf3.DOF):
    bk[i::pf3.DOF][check] = True
bu = ~bk

f = np.zeros(N)
f+=p3g.weight(M, 1, N, pf3.DOF, gcl.Direction3D(0,0,1))

KC0uu = KC0[bu, :][:, bu]
fu = f[bu]
uu = ssl.spsolve(KC0uu, fu)
u = np.zeros(N)
u[bu] = uu

w = u[2::pf3.DOF]
plt.subplot(122)
plt.plot(y, w)

import pyvista as pv
#nodes is our ncoords
elements = outdict["elements"]
nid_pos = outdict["nid_pos"]

'''handling each element separately at each elemnt type gets a diff colour'''
#springs
edges_s = list()
for s in elements["spring"]:
    edges_s.append([nid_pos[s.n1], nid_pos[s.n2]])
colors_s = [1]*len(edges_s)
#beams
edges_b = list()
for s in elements["beam"]:
    edges_b.append([nid_pos[s.n1], nid_pos[s.n2]])
colors_b = [2]*len(edges_b)
#quads
edges_q = list()
for s in elements["quad"]:
    edges_q.append([nid_pos[s.n1], nid_pos[s.n2]])
    edges_q.append([nid_pos[s.n2], nid_pos[s.n3]])
    edges_q.append([nid_pos[s.n3], nid_pos[s.n4]])
    edges_q.append([nid_pos[s.n4], nid_pos[s.n1]])
colors_q = [3]*len(edges_q)

edges = np.array(edges_s+edges_b+edges_q)
colors = np.array(colors_s+colors_b+colors_q)

# We must "pad" the edges to indicate to vtk how many points per edge
padding = np.empty(edges.shape[0], int) * 2
padding[:] = 2
edges_w_padding = np.vstack((padding, edges.T)).T

mesh_ = pv.PolyData(outdict["ncoords"], edges_w_padding)

mesh_.plot(
    scalars=colors,
    render_lines_as_tubes=True,
    style='wireframe',
    line_width=1,
    cmap='jet',
    show_scalar_bar=False,
    background='w',
)
print(len(colors))

import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix

def plot_coo_matrix(m):
    if not isinstance(m, coo_matrix):
        m = coo_matrix(m)
    fig = plt.figure()
    ax = fig.add_subplot(111, facecolor='black')
    ax.plot(m.col, m.row, 's', color='white', ms=1)
    ax.set_xlim(0, m.shape[1])
    ax.set_ylim(0, m.shape[0])
    ax.set_aspect('equal')
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    return ax
plot_coo_matrix(KC0uu)

#nodes is our ncoords
elements = outdict["elements"]
nid_pos = outdict["nid_pos"]
nodex = x+u[0::pf3.DOF]
nodey = y+u[1::pf3.DOF]
nodez = z+w
nodes = list(zip(nodex, nodey, nodez))

'''handling each element separately at each elemnt type gets a diff colour'''
#springs
edges_s = list()
for s in elements["spring"]:
    #if y[nid_pos[s.n1]]>miny and y[nid_pos[s.n1]]<maxy:
        edges_s.append([nid_pos[s.n1], nid_pos[s.n2]])
colors_s = [7]*len(edges_s)
#beams
edges_b = list()
for s in elements["beam"]:
    #if y[nid_pos[s.n1]]>miny and y[nid_pos[s.n1]]<maxy:
        edges_b.append([nid_pos[s.n1], nid_pos[s.n2]])
colors_b = [5]*len(edges_b)
#quads
edges_q = list()
for s in elements["quad"]:
    #if y[nid_pos[s.n1]]>miny and y[nid_pos[s.n1]]<maxy:
        edges_q.append([nid_pos[s.n1], nid_pos[s.n2]])
        edges_q.append([nid_pos[s.n2], nid_pos[s.n3]])
        edges_q.append([nid_pos[s.n3], nid_pos[s.n4]])
        edges_q.append([nid_pos[s.n4], nid_pos[s.n1]])
colors_q = [0]*len(edges_q)

edges = np.array(edges_s+edges_b+edges_q)
colors = np.array(colors_s+colors_b+colors_q)

# We must "pad" the edges to indicate to vtk how many points per edge
padding = np.empty(edges.shape[0], int) * 2
padding[:] = 2
edges_w_padding = np.vstack((padding, edges.T)).T

mesh_ = pv.PolyData(nodes, edges_w_padding)

mesh_.plot(
    scalars=colors,
    render_lines_as_tubes=True,
    style='wireframe',
    line_width=1,
    cmap='jet',
    show_scalar_bar=False,
    background='w',
)
print(len(colors))

plt.show()