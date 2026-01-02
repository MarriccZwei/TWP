from ..Pyfe3DModel import Pyfe3DModel
from ..LoadCase import LoadCase
from .eleProps import quad_stress_recovery, beam_stress_recovery

import numpy.typing as nt
import numpy as np
import typing as ty
from pypardiso import spsolve
import pyfe3d as pf3
import scipy.sparse as ss
import scipy.sparse.linalg as ssl
import pyvista as pv

def process_load_case(model:Pyfe3DModel, lc:LoadCase, materials:ty.Dict[str, float], desvars:ty.Dict[str, float],
                      beamTypes:ty.List[str], quadTypes:ty.List[str], flutter_re_digits:int=2,
                      plot:bool=False, savePath:str=None)->nt.NDArray[np.float64]:
    '''
    Docstring for process_load_case
    
    :param model: an alerady updated Pyfe3DModel
    :type model: Pyfe3DModel
    :param materials: the dictionary of material properties with keys in form "<PROPERTY>_<MATERIAL>"
    :param lc: the analysed load case, with all unchanging loads already applied
    :type lc: LoadCase
    :param plot: whether or not to plot to results
    :type plot: bool
    :param savePath: where to save results, if set to None, but plot==True, will show the plots instead
    :type savePath: str
    :returns: An array with maximum quad stress margin, beam stress margin, buckling load multiplier, flutter bool as float, 0.=>stable
    :type return: NDArray[float64]
    '''
    #1) weight update & static solution
    lc.update_weight(model.M)
    f = lc.loadstack()
    fu = f[model.bu]
    uu = spsolve(model.KC0uu, fu)
    u = np.zeros_like(f)
    u[model.bu] = uu

    #2) post-processing
    quad_failure_margins = list()
    beam_failure_margins = list()
    KGr = np.zeros(model.sizeKG, dtype=pf3.INT)
    KGc = np.zeros(model.sizeKG, dtype=pf3.INT)
    KGv = np.zeros(model.sizeKG, dtype=pf3.DOUBLE)
    
    #2.1) quad postprocessing
    for quad, matdir, shellprop, quadType in zip(model.quads, model.matdirs, model.shellprops, quadTypes):
        quad.update_probe_ue(u)
        quad.update_probe_xe(model.ncoords_flatten)
        quad.update_KG(KGr, KGc, KGv, shellprop)
        quad_failure_margins.append(quad_stress_recovery(desvars, materials, quad, shellprop, matdir, quadType, model.quadprobe))

    #2.2) beam postprocessing
    for beam, beamorient, beamprop, beamType in zip(model.beams, model.beamorients, model.beamprops, beamTypes):
        beam.update_probe_ue(u)
        beam.update_probe_xe(model.ncoords_flatten)
        beam.update_KG(KGr, KGc, KGv, beamprop)
        beam_failure_margins.append(beam_stress_recovery(desvars, materials, beam, beamprop, beamorient, beamType, model.beamprobe))

    #2.3) buckling
    KG = ss.coo_matrix((KGv, (KGr, KGc)), shape=(model.N, model.N)).tocsc()
    KGuu = model.uu_matrix(KG)
    num_eig_lb = 4 #as per the example in documentation, we don't need excess modes
    eigvecs = np.zeros((model.N, num_eig_lb))
    PREC = np.max(1/model.KC0uu.diagonal())
    eigvals, eigvecsu = ssl.eigsh(A=PREC*KGuu, k=num_eig_lb, which='SM', M=PREC*model.KC0uu, sigma=1., mode='cayley')
    eigvals = -1./eigvals
    eigvecs[model.bu] = eigvecsu
    load_mult = eigvals[0]

    #2.4) flutter
    if lc.aeroelastic:
        KAuu = model.uu_matrix(lc.KA)
        k=7
        eigvals, peigvecs = ssl.eigs(A=model.KC0uu - KAuu, M=model.Muu, k=k, which='LM', sigma=-1.)
        omegan = np.sqrt(eigvals)
        re_omegan = np.real(omegan)

        #checking if the real parts are close enough
        hash_omegan = np.int32(np.round(re_omegan*10**flutter_re_digits))
        om_comp_dict = dict()
        for hash in hash_omegan:
            om_comp_dict[hash]=0
        for hash in hash_omegan:
            om_comp_dict[hash]+=1

        #will be 0 if no repetitions more if there are reps
        score = float(np.count_nonzero(np.array(list(om_comp_dict.values()), dtype=np.int16)-1))
        print(omegan)
    else:
        score = 0.

    if plot:
        displ = u.reshape((model.N//pf3.DOF, pf3.DOF))[:,:3] #3d displacements
        coords = model.ncoords+5*displ#for scaling
        cells = list()
        for quad in model.quads:
            cells.append([4, model.nid_pos[quad.n1], model.nid_pos[quad.n2], model.nid_pos[quad.n3], model.nid_pos[quad.n4]])
        cells = np.array(cells).flatten()

        cell_types = np.full(len(model.quads), pv.CellType.QUAD)
        mesh = pv.UnstructuredGrid(cells, cell_types, coords)

        pts = list()
        for ine in model.inertia_poses:
            pts.append(coords[ine,:])
        pts = np.array(pts)
        cloud = pv.PolyData(pts)

        bcells = list()
        for beam in model.beams:
            bcells.append([2, model.nid_pos[beam.n1], model.nid_pos[beam.n2]])
        bcells = np.array(bcells)
        bcell_types = np.full(len(model.beams), pv.CellType.LINE)
        bmesh = pv.UnstructuredGrid(bcells, bcell_types, coords)

        plotter = pv.Plotter()
        
        mesh.cell_data["stress"] = np.array(quad_failure_margins)
        plotter.add_mesh(
            mesh,
            show_edges=True,
            cmap="coolwarm",
            scalars="stress",
            edge_color="black"
        )

        bmesh.cell_data["stress"] = np.array(beam_failure_margins)
        plotter.add_mesh(
            bmesh,
            cmap="coolwarm",
            scalars="stress",
            line_width=8
        )

        plotter.add_mesh(
            cloud,
            point_size=6,
            render_points_as_spheres=True,
            cmap="viridis"
        )

        if savePath is None:
            plotter.show()
        else:
            raise NotImplementedError

    return np.array([
        max(quad_failure_margins),
        max(beam_failure_margins),
        load_mult,
        score
    ])
