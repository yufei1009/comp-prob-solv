# %%
def lennard_jones_potential(r, epsilon, sigma):
   
    lennard_jones_potential_value = 4 * epsilon * ((sigma / r)**12 - (sigma / r)**6)
    
    return lennard_jones_potential_value

# %%



