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

airplane.draw(thin_wings=True, set_lims=False)