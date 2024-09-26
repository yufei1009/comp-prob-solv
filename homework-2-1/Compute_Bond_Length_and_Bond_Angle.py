# %%
import math
import numpy as np

# %%
def compute_bond_length(coord1, coord2):
    """
    Compute the bond length between two atoms given their Cartesian coordinates.
    
    Parameters:
    coord1 (list): A list of three floats representing the coordinates of the first atom [x1, y1, z1].
    coord2 (list): A list of three floats representing the coordinates of the second atom [x2, y2, z2].
    
    Returns:
    float: The bond length between the two atoms.
    """
    x1, y1, z1 = coord1
    x2, y2, z2 = coord2
     
    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1

# Calculate the bond length by eqation:
   
    distance = math.sqrt(dx**2 + dy**2 + dz**2)
    

    return distance

# %%
def compute_bond_angle(coord1, coord2, coord3):
    """
    Compute the bond angle between three atoms given their Cartesian coordinates.
    
    Parameters:
    coord1, coord2, coord3 (list): Lists of three floats representing the Cartesian coordinates 
                                   of the atoms A, B, and C.
    
    Returns:
    float: The bond angle in degrees.
    """
    
    A = np.array(coord1)
    B = np.array(coord2)
    C = np.array(coord3)
    
    AB = A - B
    BC = C - B

# Calculate the dot product of AB and BC
    AB_dot_BC = np.dot(AB, BC)
    
# Calculate the magnitude of AB and BC
    magnitude_AB = np.linalg.norm(AB)
    magnitude_BC = np.linalg.norm(BC)

# Calculate cos_theta    
    cos_theta = AB_dot_BC / (magnitude_AB * magnitude_BC)

# Calculate the degree    
    theta_radians = np.arccos(np.clip(cos_theta, -1.0, 1.0)) 
    theta_degrees = np.degrees(theta_radians)
    
# Classify the bond angle as acute, right, or obtuse    
    if theta_degrees < 90:
        classification = "acute"
    elif theta_degrees == 90:
        classification = "right"
    else:
        classification = "obtuse"
    
    print(f"The bond angle is {theta_degrees:.2f} degrees and is classified as {classification}.")
    
    return theta_degrees

# %%
def calculate_all_bond_lengths(molecule):
# Store values in lists
    bond_lengths = []
    unique_bond_lengths = []
    
    for atom1 in molecule:
        for atom2 in molecule:
            if atom1 != atom2:
                coord1 = molecule[atom1]
                coord2 = molecule[atom2]
                bond_length = compute_bond_length(coord1, coord2)
                bond_lengths.append((atom1, atom2, bond_length))
    
    for bond in bond_lengths:
        atom1, atom2, bond_length = bond
        
# Output unique bond lengths
        if not any(abs(bond_length - unique_length) < 1e-5 for unique_length in unique_bond_lengths):
            unique_bond_lengths.append(bond_length)  
            print(f"Bond length between {atom1} and {atom2}: {bond_length:.4f} Å")

# Add a warning if the bond is too long :            
            warning = None
            if  bond_length > 2.0:
                warning = "Warning: The bond length is too long for a covalent bond."
                print(warning)
          
    return bond_lengths

# %%
def calculate_all_bond_angles(molecule):
   
    bond_angles = []
    
    atoms = list(molecule.keys())
    
# For Hydrogen, there are only two atoms, bond angle cannot be calculated by function    
    if len(atoms) == 2:
        print("Bond angle is 180 degrees.")
        return bond_angles
        
# Make sure the second and third atoms always come after the first atom.  
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            for k in range(j + 1, len(atoms)):
                atom1 = atoms[i]
                atom2 = atoms[j]
                atom3 = atoms[k]
                
                coord1 = molecule[atom1]
                coord2 = molecule[atom2]
                coord3 = molecule[atom3]
                
                angle = compute_bond_angle(coord1, coord2, coord3)
                bond_angles.append((atom1, atom2, atom3, angle))
                print(f"Bond angle between {atom1}, {atom2}, and {atom3}: {angle:.2f}°")
    
    return bond_angles

# %%



