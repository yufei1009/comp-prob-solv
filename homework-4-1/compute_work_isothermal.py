# %%
import numpy as np
import scipy

# %%
# Define function compute_work_isothermal
def compute_work_isothermal(Vi, Vf, n, R, T):
    volume = np.linspace(Vi, Vf, 1000)
    pressure = n * R * T / volume # Calculate pressure by ideal gas law
    work_iso = - scipy.integrate.trapezoid(pressure, volume) # Use scipy.integrate.trapezoid to compute the work
    return work_iso

# %%
# Assign values to the parameters
n = 1
R = 8.314
T = 300
Vi = 0.1
Vf = 3 * Vi

# %%
W_iso = compute_work_isothermal(Vi, Vf, n, R, T) # Compute the work done during an isothermal expansion
print(f"Work done in isothermal expansion: {W_iso:.2f} J")


# %%



