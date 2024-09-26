# %%
import numpy as np
from scipy.optimize import minimize
import math

# %%
# Use scipy.optimize.minimize to find the distance between two Ar atoms

def lennard_jones_potential(r, epsilon, sigma):
    
    lennard_jones_potential_value = 4 * epsilon * ((sigma / r)**12 - (sigma / r)**6)
    
    return lennard_jones_potential_value

# %%
# Define the value of epsilon and sigma
epsilon = 0.1
sigma = 3.4

# Define total lennard jones potential
def total_potential(x):

# Set three values that need to be optimized.
    x2 = x[0] 
    x3 = x[1]
    y3 = x[2]

    # Calculate the distance between Argon atoms.    
    r12 = x2
    r13 = np.sqrt(x3**2 + y3**2)
    r23 = np.sqrt((r12 - y3)**2 + x3**2)

# Argon trimer's total lennard jones potential is the sum of  each Argon dimer pair's lennard jones potential.    
    V_total = lennard_jones_potential(r12, epsilon, sigma) + lennard_jones_potential(r13, epsilon, sigma) + lennard_jones_potential(r23, epsilon, sigma)
   
    return V_total

# %%
initial_guess = [4, 3, 3] # Initial guess: x2 = 4, x3 = 3, y3 = 3

result = minimize(total_potential, initial_guess) # Optimize the results.

x2_opt, x3_opt, y3_opt = result.x

print("The coordinate of Argon atom:" )
print("Ar1:(0, 0, 0)")
print(f"Ar2:({x2_opt:.2f}, 0, 0)")   
print(f"Ar3:({x3_opt:.2f}, {y3_opt:.2f}, 0)") 

# %%
# Store the coordinate of three Argons in a dictionary.

Ar2_x2 = x2_opt
Ar3_x3 = x3_opt
Ar3_y3 = y3_opt

Argon_trimer = {
    "Ar1":[0, 0, 0],
    "Ar2":[Ar2_x2, 0, 0],
    "Ar3":[Ar3_x3, Ar3_y3, 0]
}

# %%
# Using the function compute_bond_length, compute_bond_angle from homework-1-2

from Compute_Bond_Length_and_Bond_Angle import compute_bond_length, compute_bond_angle

# %%
Ar1_coord = Argon_trimer["Ar1"]
Ar2_coord = Argon_trimer["Ar2"]
Ar3_coord = Argon_trimer["Ar3"]


# %%
# Calculate the optimized bond length
r12_opt = compute_bond_length(Ar1_coord, Ar2_coord)
r13_opt = compute_bond_length(Ar1_coord, Ar3_coord)
r23_opt = compute_bond_length(Ar2_coord, Ar3_coord)

print(f"optimal distance r12 = {r12_opt:.2f} Å")
print(f"optimal distance r13 = {r13_opt:.2f} Å")
print(f"optimal distance r23 = {r23_opt:.2f} Å")

# Calculate the optimized bond angle.
Argon_trimer_bond_angle = compute_bond_angle(Ar1_coord, Ar2_coord, Ar3_coord)
Argon_trimer_bond_angle = compute_bond_angle(Ar2_coord, Ar3_coord, Ar1_coord)
Argon_trimer_bond_angle = compute_bond_angle(Ar3_coord, Ar1_coord, Ar2_coord)

# %%
# Discuss wheter the three Argon atoms are in an equilateral triangle.
if r12_opt == r13_opt == r23_opt:
    print("Argon atoms are in equilateral triangle.")
else:
    print("Argon atoms are not in equilateral triangle.")


# %%



# %%



