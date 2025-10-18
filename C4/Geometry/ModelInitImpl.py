import numpy as np
import typing as ty

from ..Pyfe3DModel import Pyfe3DModel
from ..Config.Config import Config
from ..Config.Codes import CODES
from .ModelInitInterface import ModelInitInterface, AdditionalMeshOutputs
from .ptsFromCAD import UnpackedPoints
from .meshingComponents import all_components
from ..geometricClasses import Mesh3D

class ModelInitImpl(ModelInitInterface):
    def modelInit(self, config:Config) -> ty.Tuple[Pyfe3DModel, AdditionalMeshOutputs]:
        #1) geometry generation
        up = UnpackedPoints(config.CAD_DATA)
        mesh = Mesh3D()
        meshOutputs = all_components(mesh, up,
                              totmassBat = config.BAT_MASS_1WING,
                              motorR = config.MR,
                              motorL = config.ML,
                              motormass = config.M_MOTOR,
                              lgM = config.M_LG,
                              lgR = config.LGR,
                              lgL = config.LGL,
                              mhn = config.M_HINGE,
                              totmassLE = config.M_LE,
                              totmassTE = config.M_TE,
                              skinCode = CODES.SKIN,
                              lgCode = CODES.LG,
                              sparCode = CODES.SPAR,
                              LETECode = CODES.LETE,
                              railCode = CODES.RAIL,
                              ribCode = CODES.RIB,
                              ncCoeff = config.N_COEFF,
                              nbuckl = config.NA
                              )
        
        #2) output definitions
        additionalMeshOutputs = AdditionalMeshOutputs(
            skinTopCoords=meshOutputs[0]["skinTop"],
            skinBotCoords=meshOutputs[0]["skinBot"],
            skinTopIds=meshOutputs[1]["skinTop"],
            skinBotIds=meshOutputs[1]["skinBot"],
            lgIds=meshOutputs[1]["lg"],
            motorIds=meshOutputs[1]["motor"]
        )

        ncoords, x, y, z, ncoords_flatten = mesh.pyfe3D()
        #we assume the wing beam to be clamped at the fuselage beginning
        model = Pyfe3DModel(ncoords, lambda x, y, z: tuple([np.isclose(y, up.ffb.y)]*6))

        #3) initializing output elements
        for quad in mesh.connections["quad"]:
            model.load_quad(*quad.ids)
            additionalMeshOutputs.load_quad(quad.eleid, quad.protocol)
        for beam in mesh.connections["beam"]:
            model.load_beam(*beam.ids)
            additionalMeshOutputs.load_beam(beam.eleid, beam.protocol)
        for inertia in mesh.inertia:
            model.load_inertia(inertia.id_)
            additionalMeshOutputs.load_inertia(inertia.mn, inertia.Jxx, inertia.Jyy, inertia.Jzz)

        return model, additionalMeshOutputs
        
