# %%
import numpy as np
import scipy

# %%
# Define function compute_work_adiabatic
def compute_work_adiabatic(Vi, Vf, n, R, T, gamma):
    P_initial = n * R * T / Vi # Calculate the initial pressure
    constant = P_initial * (Vi**gamma) # Calculate the constant
    volume = np.linspace(Vi, Vf, 1000)
    pressure = constant / (volume**gamma) # Calculate the pressures
    work_adiabatic =  - scipy.integrate.trapezoid(pressure, volume) #Use scipy.integrate.trapezoid to compute the work
    return work_adiabatic

# %%
# Assign values to the parameters
Vi = 0.1
Vf = 3 * Vi
n = 1
R = 8.314
T = 300
gamma = 1.4


# %%
W_adi = compute_work_adiabatic(Vi, Vf, n, R, T, gamma) # Compute the work done during an adiabatic expansion
print(f"Work done in adiabatic process: {W_adi:.2f} J")

# %%



