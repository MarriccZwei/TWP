from ...C4.Geometry.Mesher import Mesher
import numpy as np

def test_ine_beam_quad_mesh():
    mesher = Mesher(2)

    #1) adding the basis quad elements with one node seen as clashing one node seen as not
    mesher.load_ele([(0.,0.,0.), (0., 1., 0.), (-1., 1., 0.), (-1.,0.,0.)], 'q', {'1':None})
    mesher.load_ele([(0.01,0.,0.), (0.009, 1., 0.), (1., 1., 0.), (1.,0.,0.)], 'q', {'1':None})
    assert len(mesher.nodes)==7
    assert mesher.eleNodePoses[0]==[0,1,2,3]
    assert mesher.eleNodePoses[1]==[4,1,5,6]
    
    #2) adding the beam elements
    mesher.load_ele([(0.,-0.005,0.), (-1.001,0.,0.)], 'b', {'1':None})
    mesher.load_ele([(0.,-0.01,0.), (0.,1.,0.)], 'c', {'1':None}) #to test new category for same pyfe3d type and new node
    assert len(mesher.nodes)==8, len(mesher.nodes)
    assert mesher.eleNodePoses[0]==[0,1,2,3]
    assert mesher.eleNodePoses[1]==[4,1,5,6]
    assert mesher.eleNodePoses[2]==[0,3]
    assert mesher.eleNodePoses[3]==[7,1]

    #3) adding inertia elements
    mesher.load_ele([(1.,1.,0.)], 'i', {'1':None})
    assert len(mesher.nodes)==8
    assert mesher.eleNodePoses[0]==[0,1,2,3]
    assert mesher.eleNodePoses[1]==[4,1,5,6]
    assert mesher.eleNodePoses[2]==[0,3]
    assert mesher.eleNodePoses[3]==[7,1]
    assert mesher.eleNodePoses[4]==[5]

    #4) checking mesh slices
    quadIdx, quadCoord = mesher.get_submesh('q')
    assert np.allclose(quadIdx, np.array([0,1,2,3,4,5,6]))
    assert np.allclose(quadCoord, np.array([(0.,0.,0.), (0., 1., 0.), (-1., 1., 0.), (-1.,0.,0.), (0.01,0.,0.), (1., 1., 0.), (1.,0.,0.)]), atol=2e-2)

    cIdx, cCoord = mesher.get_submesh('c')
    assert np.allclose(cIdx, np.array([7,1]))
    assert np.allclose(cCoord, np.array([(0.,-0.01,0.), (0.,1.,0.)]), atol=2e-2)

    bIdx, bCoord = mesher.get_submesh('b')
    assert np.allclose(bIdx, np.array([0,3]))
    assert np.allclose(bCoord, np.array([(0.,-0.005,0.), (-1.001,0.,0.)]), atol=2e-2)

    iIdx, iCoord = mesher.get_submesh('i')
    assert np.allclose(iIdx, np.array([5]))
    assert np.allclose(iCoord, np.array([(1.,1.,0.),]), atol=2e-2)

if __name__ == "__main__":
    test_ine_beam_quad_mesh()
    print("test Mesher passed")