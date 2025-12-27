import pyfe3d as pf3
import numpy as np
import numpy.typing as nt
import typing as ty
import pyfe3d.shellprop as psp
import pyfe3d.beamprop as pbp
import scipy.sparse as ss

class Pyfe3DModel():
    def __init__(self, ncoords:nt.NDArray[np.float32], boundary:ty.Callable[[float, float, float], ty.Tuple[bool]]):
        """
        ncoords - a number of elements x 3 array describing the coordinates of each node
        boundary - given node coordinates returns six booleans stating whether the node is fixed wrt. a particular degree of freedom
        (true if fixed)
        """

        #1) coordinate handling
        assert ncoords.shape[1]==3
        self.ncoords = ncoords
        self.x = ncoords[:,0]
        self.y = ncoords[:,1]
        self.z = ncoords[:,2]
        self.ncoords_flatten = ncoords.flatten()
        self.nids = 1 + np.arange(ncoords.shape[0])
        self.nid_pos = dict(zip(self.nids, np.arange(len(self.nids))))
        self.pos_nid = dict(zip(np.arange(len(self.nids)), self.nids))
        self.N = pf3.DOF*len(ncoords)

        #2) pyfe3d elements and their properties
        self.beamprobe = pf3.BeamCProbe()
        self.quadprobe = pf3.Quad4Probe()
        self.beamdata = pf3.BeamCData()
        self.quaddata = pf3.Quad4Data()

        self.beams:ty.List[pf3.BeamC] = list()
        self.beamprops:ty.List[pbp.BeamProp] = list()
        self.beamorients:ty.List[ty.Tuple[int]] = list()

        self.quads:ty.List[pf3.Quad4] = list()
        self.shellprops:ty.List[psp.ShellProp] = list()
        self.matdirs:ty.List[ty.Tuple[int]] = list()

        self.inertia_poses:ty.List[int]=list() #list of poses for inertia to be added at
        self.inertia_vals:ty.List[ty.Tuple[float]] = list() #(m, Jxx, Jyy, Jzz) the inertia additions

        #3) sparse matrices - these will be initialised only after the elements are loaded
        self.KC0r = np.zeros(1)
        self.KC0c = np.zeros(1)
        self.KC0v = np.zeros(1)
        self.Mr = np.zeros(1)
        self.Mc = np.zeros(1)
        self.Mv = np.zeros(1)
        self.INERTIA_SPARSE_SIZE = 6 #for loading additional node inertia to the mass matrix
        self.sizeKG = 0 #parameter updated during M and KC0 update, used to notify the Solution of how big the KG coo has to be

        self.KC0 = ss.coo_matrix((self.KC0v, (self.KC0r, self.KC0c)), shape=(self.N, self.N)).tocsc()
        self.M = ss.coo_matrix((self.Mv, (self.Mr, self.Mc)), shape=(self.N, self.N)).tocsc()

        #4) applying the boundary conditions
        self.bk = np.zeros(self.N, dtype=bool)
        for i in range(ncoords.shape[0]):
            boundaryChecks = boundary(self.x[i], self.y[i], self.z[i])
            for j in range(pf3.DOF):
                self.bk[i*pf3.DOF+j] = boundaryChecks[j]
        self.bu = ~self.bk

        self.KC0uu = self.KC0 #before we have any meaningful values here, we cannot apply the boundary
        self.Muu = self.M


    def uu_matrix(self, matrix:nt.ArrayLike) -> nt.ArrayLike:
        """
        trims a coo matrix by removing elements on the boundary
        """
        return matrix[self.bu, :][:, self.bu]


    def load_quad(self, pos1:int, pos2:int, pos3:int, pos4:int):
        '''
        loads a quad to the model, whose external node ids (so starting from zero) are specified by pos1, pos2, etc.
        Following the pyfe3d doc convention for the order of nodes.
        '''
        quad = pf3.Quad4(self.quadprobe)
        quad.n1 = self.pos_nid[pos1]
        quad.n2 = self.pos_nid[pos2]
        quad.n3 = self.pos_nid[pos3]
        quad.n4 = self.pos_nid[pos4]
        quad.c1 = pf3.DOF*pos1
        quad.c2 = pf3.DOF*pos2
        quad.c3 = pf3.DOF*pos3
        quad.c4 = pf3.DOF*pos4
        self.quads.append(quad)


    def load_beam(self, pos1:int, pos2:int):
        '''
        loads a beam into the model, whose external node ids (so starting from zero) are specified by pos1 and pos2.
        Following the pyfe3d doc convention for the order of nodes.
        '''
        beam = pf3.BeamC(self.beamprobe)
        beam.n1 = self.pos_nid[pos1]
        beam.n2 = self.pos_nid[pos2]
        beam.c1 = pf3.DOF*pos1
        beam.c2 = pf3.DOF*pos2
        self.beams.append(beam)


    def load_inertia(self, pos:int):
        '''loads a holder for additional per node inertia, initialised form the node pos'''
        self.inertia_poses.append(pos)


    def KC0_M_update(self, beamprops:ty.List[pbp.BeamProp], beamorients:ty.List[ty.Tuple[float]], 
                   shellprops:ty.List[psp.ShellProp], matdirs:ty.List[ty.Tuple[float]], 
                   inertia_vals:ty.List[ty.Tuple[float]]):
        '''
        Assigns prop and orientation to the elements and updates the KC0 and M matrices, with re-scaling where necessary.
        For security reasons, has to be called AFTER all elements and inertia have been loaded
        '''
        assert len(beamprops)==len(beamorients)==len(self.beams)
        assert len(matdirs)==len(shellprops)==len(self.quads)
        assert len(inertia_vals)==len(self.inertia_poses)
        for borient in beamorients:
            assert len(borient)==3 
        for matvec in matdirs:
            assert len(matvec)==3 
        for inval in inertia_vals:
            assert len(inval)==4 #1 trnslational mass 3 moments of inertia

        nquads = len(self.quads)
        nbeams = len(self.beams)
        ninert = len(self.inertia_poses)

        #1) storing the element properties
        self.beamprops = beamprops
        self.beamorients = beamorients
        self.shellprops = shellprops
        self.matdirs = matdirs
        self.inertia_vals = inertia_vals

        #2) calculating matrix size based on element data sizes
        kc0_size = 0
        mass_size = 0
        kg_size= 0

        kc0_size += self.beamdata.KC0_SPARSE_SIZE*nbeams
        kc0_size += self.quaddata.KC0_SPARSE_SIZE*nquads

        mass_size += self.beamdata.M_SPARSE_SIZE*nbeams
        mass_size += self.quaddata.M_SPARSE_SIZE*nquads
        mass_size += self.INERTIA_SPARSE_SIZE*ninert

        kg_size += self.beamdata.KG_SPARSE_SIZE*nbeams
        kg_size += self.quaddata.KG_SPARSE_SIZE*nquads
        self.sizeKG = kg_size #as stated in the constructor

        #3) checking if the sparse matrix sizes have changed and re-initializing them if necessary
        if kc0_size != len(self.KC0v):
            self.KC0r = np.zeros(kc0_size, dtype=pf3.INT)
            self.KC0c = np.zeros(kc0_size, dtype=pf3.INT)
            self.KC0v = np.zeros(kc0_size, dtype=pf3.DOUBLE)

        if mass_size != len(self.Mv):
            self.Mr = np.zeros(mass_size, dtype=pf3.INT)
            self.Mc = np.zeros(mass_size, dtype=pf3.INT)
            self.Mv = np.zeros(mass_size, dtype=pf3.DOUBLE)

        # KG is a function of both the model and the deformation applied to it, specific per Solution
        # Thus, not initiated here

        #4) setting the init coordinates
        init_k_KC0 = 0
        init_k_M = 0
        init_k_KG = 0

        #5) quad contributions
        for quad, shellprop, matdir in zip(self.quads, shellprops, matdirs):
            quad.init_k_KC0 = init_k_KC0
            quad.init_k_M = init_k_M
            quad.init_k_KG = init_k_KG
            quad.update_rotation_matrix(self.ncoords_flatten, matdir[0], matdir[1], matdir[2])
            quad.update_probe_xe(self.ncoords_flatten)
            quad.update_KC0(self.KC0r, self.KC0c, self.KC0v, shellprop) #matrix contribution, changing the matrices sent
            quad.update_M(self.Mr, self.Mc, self.Mv, shellprop)
            #as mentioned above, KG not updated here
            init_k_KC0 += self.quaddata.KC0_SPARSE_SIZE
            init_k_M += self.quaddata.M_SPARSE_SIZE
            init_k_KG += self.quaddata.KG_SPARSE_SIZE

        #6) beam contributions
        for beam, beamprop, beamorient in zip(self.beams, beamprops, beamorients):
            beam.init_k_KC0 = init_k_KC0
            beam.init_k_M = init_k_M
            beam.init_k_KG = init_k_KG
            beam.update_rotation_matrix(beamorient[0], beamorient[1], beamorient[2], self.ncoords_flatten)
            beam.update_probe_xe(self.ncoords_flatten)
            beam.update_KC0(self.KC0r, self.KC0c, self.KC0v, beamprop)
            beam.update_M(self.Mr, self.Mc, self.Mv, beamprop) #mtype consistent
            #as mentioned above, KG not updated here
            init_k_KC0 += self.beamdata.KC0_SPARSE_SIZE
            init_k_M += self.beamdata.M_SPARSE_SIZE
            init_k_KG += self.beamdata.KG_SPARSE_SIZE

        #7) inertia contributions
        for inertia_pos, inertia_val in zip(self.inertia_poses, inertia_vals):
            k = init_k_M
            pos = pf3.DOF*inertia_pos
            self.Mr[k] = 0+pos
            self.Mc[k] = 0+pos
            self.Mv[k] += inertia_val[0]
            k+=1
            pos = pf3.DOF*inertia_pos
            self.Mr[k] = 1+pos
            self.Mc[k] = 1+pos
            self.Mv[k] += inertia_val[0]
            k+=1
            pos = pf3.DOF*inertia_pos
            self.Mr[k] = 2+pos
            self.Mc[k] = 2+pos
            self.Mv[k] += inertia_val[0]
            k+=1
            pos = pf3.DOF*inertia_pos
            self.Mr[k] = 3+pos
            self.Mc[k] = 3+pos
            self.Mv[k] += inertia_val[1]
            k+=1
            pos = pf3.DOF*inertia_pos
            self.Mr[k] = 4+pos
            self.Mc[k] = 4+pos
            self.Mv[k] += inertia_val[2]
            k+=1
            pos = pf3.DOF*inertia_pos
            self.Mr[k] = 5+pos
            self.Mc[k] = 5+pos
            self.Mv[k] += inertia_val[3]
            init_k_M += self.INERTIA_SPARSE_SIZE

        #8) re-creating the csc matrices
        self.KC0 = ss.coo_matrix((self.KC0v, (self.KC0r, self.KC0c)), shape=(self.N, self.N)).tocsc()
        self.M = ss.coo_matrix((self.Mv, (self.Mr, self.Mc)), shape=(self.N, self.N)).tocsc()
        
        #9) re-aplying the boundary conditions
        self.KC0uu = self.uu_matrix(self.KC0)
        self.Muu = self.uu_matrix(self.M)


    def structural_mass(self, rho:float, thickness:float):
        '''
        The mass of beams and quads, NOT including inertias
        '''
        structmass = 0
        structmass += sum(self._beam_mass(beam, beamprop) for beam, beamprop in zip(self.beams, self.beamprops))
        structmass += sum(self._quad_mass(quad, rho, thickness) for quad in self.quads)
        return structmass


    def _beam_mass(self, beamEle:pf3.BeamC, beamProp:pbp.BeamProp):
        return beamEle.length*beamProp.intrho


    def _quad_mass(self, quadEle:pf3.Quad4, rho:float, thickness:float):
        coords = lambda nx:self.ncoords[self.nid_pos[nx], :]
        coords1 = coords(quadEle.n1)
        coords2 = coords(quadEle.n2)
        coords3 = coords(quadEle.n3)
        coords4 = coords(quadEle.n4)

        #area of the skin
        area = .5*(np.linalg.norm(np.cross(coords4-coords1, coords2-coords1))+ 
                np.linalg.norm(np.cross(coords2-coords3, coords4-coords3)))
        #NOTE: it is assumed the quad is a uniform plate - to use with sandwiches one has to use equivalent density

        return rho*thickness*area