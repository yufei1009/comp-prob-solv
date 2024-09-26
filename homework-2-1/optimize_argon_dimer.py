# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
# Define Lennard-Jones potential according to the formula

def lennard_jones_potential(r, epsilon, sigma):
   
    lennard_jones_potential_value = 4 * epsilon * ((sigma / r)**12 - (sigma / r)**6)
    
    return lennard_jones_potential_value

# %%
# Use scipy.optimize.minimize to find the distance between two Ar atoms

from scipy.optimize import minimize 

epsilon = 0.01
sigma = 3.40
potential_energy = lambda r: lennard_jones_potential(r, epsilon, sigma) # Using the function lambda to fix epsilon = 0.01, sigma = 3.40 and optimize r.

initial_guess = 4
result = minimize(potential_energy, initial_guess, bounds=[(3, 6)]) # Optimize r between between 3˚A ≤ r ≤ 6˚A.

equilibrium_distance = result.x[0] # Optmized r value

if __name__ == "__main__": # Avoird giving the following results in other python files when import the function.
    print(f"Equilibrium distance: {equilibrium_distance:.2f} Å")

# %%
# Plot the figure

r_values = np.linspace(3, 6, 100)
v_values = lennard_jones_potential(r_values, epsilon, sigma)

if __name__ == "__main__":
    plt.plot(r_values, v_values, label="Lennard-Jones Potential")
    plt.axvline(equilibrium_distance, color="red", linestyle="--", label="Equilibrium Distance") # Mark the equilibrium distance
    plt.title("Lennard-Jones Potential for Argon Dimer")
    plt.xlabel("Distance between Ar atoms (Å)")
    plt.ylabel("Lennard-Jones potential (eV)")
    plt.legend()
    plt.show()


# %%



