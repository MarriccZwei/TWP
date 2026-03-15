from .Exclusion import Exclusion
from .Mesher import Mesher
from .ElysianWing import ElysianWing
from .InertiaSumesh import InertiaSubmesh
from .StructuralSubmesh import UniformStructuralSubmesh
from ..Pyfe3DModel import Pyfe3DModel

import typing as ty
import aerosandbox.numpy as np
import pyfe3d as pf3

def geometry_init(GEOM_SOURCE:dict[str, float], HYPERPARAMS:dict[str, float], MASSES:dict[str, float], N:int, collisionDecimalPlaces:int=8, 
                  springargs:ty.Tuple[float]=(1e10, 0., 0., 1e10, 0., 0., 0., 1., 0.)) -> ty.Tuple[Pyfe3DModel, Mesher]:
    """
    converts the elements imported from CAD into a pyfe3d model
    
    :param catiaout: catia export string, formatted as elementTypeCode;arg1,arg2,etc.;node1x$node1y$node1z,node2x$node2y$node2z,etc.|
    :type catiaout: str
    :param collisionDecimalPlaces: number of decimal places for which node coordinates in meters have to be matching to be considered a separate node
    :type collisionDecimalPlaces: int
    :param springargs: parameters for initialising spring elements (in this model used as stiff massless axial connectors)
    :type springargs: Tuple[float]
    """
    #1) resolving the mesh from submesh outputs
    wing = ElysianWing(GEOM_SOURCE, HYPERPARAMS["(H/c)_sq"])
    excl = Exclusion(wing.scaffold, HYPERPARAMS["rj/c"], wing.c_at_y)
    ism = InertiaSubmesh(wing.scaffold, HYPERPARAMS, MASSES, wing.c_at_y, wing.large_equipment_summary())
    ssm = UniformStructuralSubmesh(wing, HYPERPARAMS, N)
    mesher = Mesher(collisionDecimalPlaces)
    for eleType, eleArgs, eleNodes in zip(ism.eleTypes+ssm.eleTypes, ism.eleArgs+ssm.eleArgs, ism.eleNodes+ssm.eleNodes):
        mesher.load_ele(eleNodes, eleType, eleArgs)

    #2) initialising the model
    ncoords = np.array(mesher.nodes)
    yfus = ncoords[:,1].min()
    boundary = lambda x,y,z:tuple([np.isclose(y, yfus, atol=.1**collisionDecimalPlaces)]*pf3.DOF)
    model = Pyfe3DModel(np.array(mesher.nodes), boundary)

    #3) loading the elements into the model
    for eleType, eleNodePoses in zip(mesher.eleTypes, mesher.eleNodePoses):
        if eleType[1]=='q':
            model.load_quad(*eleNodePoses)
        elif eleType[1]=='b':
            model.load_beam(*eleNodePoses)
        elif eleType[1]=='s':
            model.load_spring(*eleNodePoses, *springargs)
        elif eleType[1]=='i':
            model.load_inertia(eleNodePoses[0])
        else:
            raise ValueError("Invalid element type. Only 'q'->Quad4, 'b'->BeamC, 's'->spring, 'i'->Inertia elements are supported!!!")
        
    #4) checking for the mesh being properly resolved
    expected_node_count = ism.expected_node_count+ssm.expected_node_count
    assert ncoords.shape[0] == expected_node_count, f"Nodes resolved: {ncoords.shape[0]}, nodes expected: {expected_node_count}"

    # from scipy.spatial import cKDTree
    # epsilon = 1e-4  # distance threshold

    # # build KD-tree
    # tree = cKDTree(ncoords)

    # # find all pairs within epsilon
    # pairs = tree.query_pairs(r=epsilon)

    # # extract indices of points that appear in any pair
    # close_indices = np.unique(np.array(list(pairs)).flatten())

    # # points that are close to at least one other
    # close_points = ncoords[close_indices]

    # import pyvista as pv

    # plotter = pv.Plotter()
    # cloud = pv.PolyData(close_points)
    # print(len(close_indices))

    # plotter = pv.Plotter()
    # plotter.add_mesh(cloud, color="red", point_size=12, render_points_as_spheres=True)
    # plotter.show()

    return model, mesher, excl, wing

