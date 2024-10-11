# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# %%
# Read the thermaldynamic properties data from the csv file created in the previous step.
plot_data = pd.read_csv('Thermodynamic_Properties_vs_Temperatures.csv')

temperatures = plot_data['Temperature (K)']
Cv = plot_data['Heat Capacity (J/K)']

# %%
# Plot heat capacity as a function of temperature
plt.plot(temperatures, Cv, label="Heat Capacity")
plt.xlabel("Temperature (K)")
plt.ylabel("Heat Capacity (J/K)")
plt.title("Heat Capacity vs Temperature")
plt.legend()
plt.savefig("Heat_Capacity_vs_Temperature.png")
plt.show()

# %%



