from ..Pyfe3DModel import Pyfe3DModel
from ..LoadCase import LoadCase
from .eleProps import quad_stress_recovery, beam_stress_recovery

import numpy.typing as nt
import aerosandbox.numpy as np
import typing as ty
from pypardiso import spsolve
import pyfe3d as pf3
import scipy.sparse as ss
import scipy.sparse.linalg as ssl
import pyvista as pv

def process_load_case(model:Pyfe3DModel, lc:LoadCase, materials:ty.Dict[str, float], desvars:ty.Dict[str, float],
                      beamTypes:ty.List[str], quadTypes:ty.List[str],
                      plot:bool=False, savePath:str=None, 
                      num_eig_lb:float=4)->nt.NDArray[np.float64]:
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
    eigvecs = np.zeros((model.N, num_eig_lb))
    PREC = np.max(1/model.KC0uu.diagonal())
    eigvals, eigvecsu = ssl.eigsh(A=PREC*KGuu, k=num_eig_lb, which='SM', M=PREC*model.KC0uu, sigma=1., mode='cayley')
    eigvals = -1./eigvals
    eigvecs[model.bu] = eigvecsu
    load_mult = eigvals[0]

    failure_margins = np.array([
        max(quad_failure_margins),
        max(beam_failure_margins),
        load_mult, 
    ])

    #3) Plotting
    if plot:
        coords = prep_displacements(u, model, 5.)
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
            print("Generating report...")
            #giving save path will create the full plotting report
            plotter.save_graphic(savePath+"Stresses.pdf", "Stresses")
            plot_nodal_quantity(model.ncoords, lc.A[2::pf3.DOF], model, savePath, "AerodynamicLoadZ")
            plot_nodal_quantity(model.ncoords, lc.L[2::pf3.DOF], model, savePath, "LandingLoadZ")
            plot_nodal_quantity(model.ncoords, lc.L[0::pf3.DOF], model, savePath, "LandingLoadX")
            plot_nodal_quantity(model.ncoords, lc.T[0::pf3.DOF], model, savePath, "ThrustX")
            plot_nodal_quantity(model.ncoords, lc.W[2::pf3.DOF], model, savePath, "WeightZ")
            eigvec_scaling = 25.
            for i in range(eigvecs.shape[1]):
                plot_nodal_quantity(prep_displacements(eigvecs[:,i], model, eigvec_scaling/max(eigvecs[:,i])), eigvecs[:,i][2::pf3.DOF],
                                    model, savePath, f"BucklingMode{i}")
            with open(savePath+"failure_margins.txt", "w") as file:
                file.write(f"fail_margs: {failure_margins}")
            print(f"Report saved at the path below.\n{savePath}")        

    return failure_margins

def process_aeroelastic_load_case(model:Pyfe3DModel, lc:LoadCase, plot:bool=False, savePath:str=None, k:int=7, returnOmegan=False):
    """
    
    """
    KAuu = model.uu_matrix(lc.KA)
    peigvecs = np.zeros((model.N, k))
    eigvalsFlutter, peigvecsu = ssl.eigs(A=model.KC0uu - KAuu, M=model.Muu, k=k, which='LM', sigma=-1.)
    omegan = np.sqrt(eigvalsFlutter)
    peigvecs[model.bu, :] = peigvecsu

    if plot: #TODO: Add savePath handling
        eigvec_scaling = 1.
        for i in range(peigvecs.shape[1]):
                plot_nodal_quantity(prep_displacements(peigvecs[:,i], model, eigvec_scaling/max(abs(peigvecs[:,i][2::pf3.DOF]))), peigvecs[:,i][2::pf3.DOF],
                                    model, savePath, f"NatfreqMode{i}")

    if returnOmegan:
        return omegan
    else:            
        return np.count_nonzero(np.imag(omegan))


def prep_displacements(u:nt.NDArray[np.float64], model:Pyfe3DModel, scaling:float=1.):
    '''
    Interprets displacements into format consistent with ncoords
    
    :param u: displacements as obtained from modal or static analysis
    :type u: nt.NDArray[np.float64]
    :param model: the model on which the analysis yielding the displacements was conducted
    :type model: Pyfe3DModel
    :param scaling: scaling to apply to the displacements for visibility
    :type scaling: float
    :return: model ncoords updated by the displacements
    :rtype: NDArray[float64]
    '''
    displ = u.reshape((model.N//pf3.DOF, pf3.DOF))[:,:3] #3d displacements
    return model.ncoords+scaling*displ#for scaling


def plot_nodal_quantity(ncoords:nt.NDArray[np.float64], qty:nt.NDArray[np.float64], 
                        model:Pyfe3DModel, savePath:str, plotName:str, extension:str='.pdf', show=False):
    '''
    Docstring for plot_nodal_quantity
    
    :param ncoords: nodal coordinates on which the plot is based, shape (N/DOF, 3)
    :type ncoords: nt.NDArray[np.float64]
    :param qty: quantity evaluated @ each node, shape (N/DOF)
    :type qty: nt.NDArray[np.float64]
    :param model: the model object to take elements from
    :type model: Pyfe3DModel
    :param savePath: the folder in which the plot is to be saved, if None, plots will not be saved
    :type savePath: str
    :param plotName: the file name of the plot in that folder
    :type plotName: str
    :param extension: file extensions to save the plot as
    :type extension: str
    '''
    quad_qty = list()
    beam_qty = list()

    cells = list()
    for quad in model.quads:
        np1 = model.nid_pos[quad.n1]
        np2 = model.nid_pos[quad.n2]
        np3 = model.nid_pos[quad.n3]
        np4 = model.nid_pos[quad.n4]
        cells.append([4, np1, np2, np3, np4])
        quad_qty.append((qty[np1]+qty[np2]+qty[np3]+qty[np4])/4)
    cells = np.array(cells).flatten()

    cell_types = np.full(len(model.quads), pv.CellType.QUAD)
    mesh = pv.UnstructuredGrid(cells, cell_types, ncoords)

    pts = list()
    for ine in model.inertia_poses:
        pts.append(ncoords[ine,:])
    pts = np.array(pts)
    cloud = pv.PolyData(pts)

    bcells = list()
    for beam in model.beams:
        np1 = model.nid_pos[beam.n1]
        np2 = model.nid_pos[beam.n2]
        bcells.append([2, np1, np2])
        beam_qty.append((qty[np1]+qty[np2])/2)
    bcells = np.array(bcells)
    bcell_types = np.full(len(model.beams), pv.CellType.LINE)
    bmesh = pv.UnstructuredGrid(bcells, bcell_types, ncoords)

    plotter = pv.Plotter()
    
    mesh.cell_data[plotName] = np.array(quad_qty)
    plotter.add_mesh(
        mesh,
        show_edges=True,
        cmap="coolwarm",
        scalars=plotName,
        edge_color="black"
    )

    bmesh.cell_data[plotName] = np.array(beam_qty)
    plotter.add_mesh(
        bmesh,
        cmap="coolwarm",
        scalars=plotName,
        line_width=8
    )

    plotter.add_mesh(
        cloud,
        point_size=6,
        render_points_as_spheres=True,
        cmap="viridis"
    )

    if not(savePath is None):
        plotter.save_graphic(savePath+plotName+extension, plotName)
    if show:
        plotter.show()