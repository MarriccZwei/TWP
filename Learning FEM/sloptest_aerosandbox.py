import aerosandbox as asb
import aerosandbox.numpy as np

# Basic geometry parameters
span = 42  # Total span
root_chord = 5
taper_ratio = 0.4
tip_chord = root_chord * taper_ratio
half_span = span / 2
n_propulsors_per_wing = 4

# Define the wing
wing = asb.Wing(
    name="Main Wing",
    symmetric=True,
    xsecs=[
        asb.WingXSec(
            xyz_le=[0,0,0],
            chord=root_chord,
            airfoil=asb.Airfoil("naca2412")
        ),
        asb.WingXSec(
            xyz_le=[0,half_span,0],
            chord=tip_chord,
            airfoil=asb.Airfoil("naca2412")
        )
    ]
)

# Define propulsor positions along the wing span
# Choose evenly spaced positions on each side, avoiding root/tip
y_positions_right = np.linspace(half_span * 0.15, half_span * 0.85, n_propulsors_per_wing)
y_positions = np.concatenate([-y_positions_right[::-1], y_positions_right])  # Left and right

# Define the thrust magnitude for each propulsor
def constant_thrust(thrust_value):
    return lambda op_point: np.array([thrust_value, 0, 0])  # Thrust in +X (forward)

# Create propulsors
thrust_per_engine = 5000  # N (arbitrary example value)
propulsors = []

for i, y in enumerate(y_positions):
    propulsors.append(
        asb.Propulsor(
            name=f"Engine_{i+1}",
            xyz_c=np.array([-2, y, 0]),  # 2 m ahead of wing LE, spanwise y, z = 0
            xyz_normal=[-1,0,0]
        )
    )

# Define the airplane with wing and propulsors
airplane = asb.Airplane(
    name="Custom 8-Engine Aircraft",
    wings=[wing],
    propulsors=propulsors
)

def run_vlm_force_analysis_custom_cp(
    airplane: asb.Airplane,
    op_point: asb.OperatingPoint,
    custom_control_points: np.ndarray
) -> np.ndarray:
    """
    Runs a VLM analysis using predefined control points.

    Args:
        airplane (asb.Airplane): The airplane object.
        op_point (asb.OperatingPoint): Operating point (flow conditions).
        custom_control_points (np.ndarray): Array of shape (n_panels, 3) giving desired control points [x, y, z].

    Returns:
        np.ndarray: Array of aerodynamic force vectors (Fx, Fy, Fz) at each custom control point.
    """
    import copy

    # Create a deep copy of the airplane to avoid modifying the original
    airplane_custom = copy.deepcopy(airplane)

    # Force panelization with control over mesh
    # Assumption: only one wing
    wing = airplane_custom.wings[0]

    # We'll manually define the mesh by slicing chordwise and spanwise
    n_points = len(custom_control_points)
    
    # Create a dummy VLM to match panel count, just to access sol object
    vlm_dummy = asb.VLM(
        airplane=airplane_custom,
        op_point=op_point,
        spanwise_resolution=int(np.sqrt(n_points)),
        chordwise_resolution=1  # Flat panels
    )
    sol_dummy = vlm_dummy.run()

    # Override control point positions with provided ones
    sol_dummy["xyz_cp"] = custom_control_points

    # Recalculate induced velocities and forces at new control points
    sol_dummy.recompute_geometry()
    sol_dummy.recompute_solution()
    forces = sol_dummy["f_aero_panels"].reshape(-1, 3)

    return forces

#a dummy sheet
import aerosandbox.numpy as np

import aerosandbox.numpy as np

import numpy as onp  # standard NumPy

def naca2412_height(x: float, surface: str = "upper") -> float:
    """
    Returns height of a NACA 2412 airfoil at a given x/c location.

    Args:
        x (float): x-position as a fraction of chord (0 <= x <= 1)
        surface (str): "upper", "lower", or "camber"

    Returns:
        float: z-position as fraction of chord
    """
    # Ensure x is in [0, 1]
    if not (0 <= x <= 1):
        raise ValueError("x must be between 0 and 1.")

    # Parameters for NACA 2412
    m = 0.02  # Max camber
    p = 0.4   # Max camber position
    t = 0.12  # Thickness

    # Thickness distribution formula
    yt = 5 * t * (
        0.2969 * onp.sqrt(x)
        - 0.1260 * x
        - 0.3516 * x**2
        + 0.2843 * x**3
        - 0.1015 * x**4
    )

    # Camber line and its slope
    if x < p:
        yc = m / p**2 * (2 * p * x - x**2)
        dyc_dx = 2 * m / p**2 * (p - x)
    else:
        yc = m / (1 - p)**2 * ((1 - 2 * p) + 2 * p * x - x**2)
        dyc_dx = 2 * m / (1 - p)**2 * (p - x)

    theta = onp.arctan(dyc_dx)

    # Final upper/lower surface
    if surface == "upper":
        return yc + yt * onp.cos(theta)
    elif surface == "lower":
        return yc - yt * onp.cos(theta)
    elif surface == "camber":
        return yc
    else:
        raise ValueError("surface must be 'upper', 'lower', or 'camber'.")



airplane.draw(thin_wings=True, set_lims=False)