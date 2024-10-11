# %%
import numpy as np
from scipy.constants import k, eV, h, u, pi # import the constants
import pandas as pd
import scipy


# %%
# Define lennard Jones potential
def lennard_jones_potential(r, epsilon, sigma):
    return 4 * epsilon * ((sigma / r)**12 - (sigma / r)**6)

# %%
# Define a function to compute the distance r for Lennard Jones Potential based on the position of two particles.
def distance (x1, y1, z1, x2, y2, z2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)

# %%
# Define partition function
def compute_partition_function(epsilon, sigma, V, Tmin, Tmax, data_points):
    
    temperatures = np.linspace(Tmin, Tmax, data_points) # Set the temperatures
    
    partition_function = [] # Create an empty list for partition function    

    
    for T in temperatures:
        
        L = V**(1/3) # Length of the cubic box
       
        # Define the positions of two Ar atoms
        coord = np.linspace(-L/2, L/2, data_points)  
        x1, y1, z1, x2, y2, z2 = np.meshgrid(coord, coord, coord, coord, coord, coord, indexing = 'ij')
        
        beta = 1 / (k * T)
        
        r = distance(x1, y1, z1, x2, y2, z2) # Compute the distance between two Ar atoms
        r[r < sigma] = sigma # Set the minimum distance between two Ar atoms as sigma
        
        V_LJ = lennard_jones_potential(r, epsilon, sigma)
        boltzmann_factor = np.exp(-beta * V_LJ) # Intergal part

        # Intergal over x1, y1, z1, x2, y2 and z2
        integral_x1 = scipy.integrate.trapezoid(boltzmann_factor, coord)
        integral_x1_y1 = scipy.integrate.trapezoid(integral_x1, coord)
        integral_x1_y1_z1 = scipy.integrate.trapezoid(integral_x1_y1, coord)
        integral_x1_y1_z1_x2 = scipy.integrate.trapezoid(integral_x1_y1_z1, coord)
        integral_x1_y1_z1_x2_y2 = scipy.integrate.trapezoid(integral_x1_y1_z1_x2, coord)
        integral_x1_y1_z1_x2_y2_z2 = scipy.integrate.trapezoid(integral_x1_y1_z1_x2_y2, coord)
   
        Ar_mass = 39.948 * u
        lambda_Ar = np.sqrt((beta * h**2) / (2 * pi * Ar_mass))

        # Compute the partition function
        Z = (1 / lambda_Ar**6) * integral_x1_y1_z1_x2_y2_z2
        
        partition_function.append(Z) # Save the results of partition function in the list.
        
    return temperatures, partition_function


# %%
# Assign values to the variables
epsilon = 0.0103 * eV # Convert eV to J
sigma = 3.4e-10 # Convert Å to m
V = 1000 * (1e-10)**3 # Convert Å^3 to m^3
Tmin = 10 # K
Tmax = 1000 # K
data_points = 11 # Avoid using too much memory (Error occurs when data points > 30)

temperatures, partition_function = compute_partition_function(epsilon, sigma, V, Tmin, Tmax, data_points)

# %%
# Create a DataFrame and save the results as a .csv file.
df = pd.DataFrame({
    'Temperature (K)': temperatures,
    'Partition Function (Z)': partition_function
})

df.to_csv('Partition_Function_vs_Temperature.csv')

# %%


# %%



