from .CatiaParser import CatiaParser
from .Mesher import Mesher
from ..Pyfe3DModel import Pyfe3DModel

import typing as ty
import aerosandbox.numpy as np
import pyfe3d as pf3

def geometry_init(catiaout:str, collisionDecimalPlaces:int=5, 
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
    #1) resolving the mesh from catia outputs
    parsed = CatiaParser(catiaout)
    mesher = Mesher(collisionDecimalPlaces)
    for eleType, eleArgs, eleNodes in parsed.get_mesh_data():
        if not eleType[1] in ['s', 'i']:
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
            pass#model.load_spring(*eleNodePoses, *springargs)
        elif eleType[1]=='i':
            pass#model.load_inertia(eleNodePoses[0])
        else:
            raise ValueError("Invalid element type. Only 'q'->Quad4, 'b'->BeamC, 's'->spring, 'i'->Inertia elements are supported!!!")

    return model, mesher

