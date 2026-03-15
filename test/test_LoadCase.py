from ..C4.LoadCase import LoadCase 

import aerosandbox as asb
import aerosandbox.numpy as np

def test_compressibility_corrections():
    for i in range(17, 20):
        lc_dummy = LoadCase(1., 100., 36, 9.81, 0., asb.OperatingPoint(velocity=270.), np.array([[0., 0., 0.], [0., 1., 0.], [0., 2., 0.]]), 
                            np.array([[0.2, 0., 0.], [0.2, 1., 0.], [0.2, 2., 0.]]), [asb.Airfoil("naca5412")]*3, 20, i, 10, True)
        airplane, vlm, forces, moments = lc_dummy._vlm(debug=True)
        print(forces)
    vlm.draw()


if __name__ == "__main__":
    test_compressibility_corrections()