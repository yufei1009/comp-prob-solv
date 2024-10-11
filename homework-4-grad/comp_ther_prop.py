# %%
import numpy as np
from scipy.constants import k # import the constants
import pandas as pd

# %%
# Define a function to compute internal energy and heat capacity
def compute_thermodynamic_properties(temperatures, partition_function):
    
    beta = 1 / (k * temperatures)
    ln_Z = np.log(partition_function)
    U = -np.gradient(ln_Z, beta) # Internal Energy
    
    Cv = np.gradient(U, temperatures) # Heat capacity
    return U, Cv

# %%
# Read the results from the previous step.
data = pd.read_csv('Partition_Function_vs_Temperature.csv') 

temperatures = data['Temperature (K)']
partition_function = data['Partition Function (Z)']

# %%
# Cpmpute internal energy and heat capacity
U, Cv = compute_thermodynamic_properties(partition_function, temperatures)

# %%
# Create a DataFrame and save the results of themaldynamic properties as a .csv file.
thermodynamic_properties_df = pd.DataFrame({
    'Temperature (K)': temperatures,
    'Internal Energy (J)': U,
    'Heat Capacity (J/K)': Cv
})

thermodynamic_properties_df.to_csv('Thermodynamic_Properties_vs_Temperatures.csv')

# %%



