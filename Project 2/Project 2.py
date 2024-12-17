# %%
# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

# Boltzmann constant and random seed
k_B = 1.380649e-23 # Boltzmann constant in units of (m^2 kg s^-2 K^-1)
np.random.seed(42) # Set a random seed for repeatability

# %%
# Apply periodic boundary conditions
def apply_pbc(position, box_size):
    return position % box_size

# %%
# Initialize positions
def initialize_chain(n_particles, box_size, r0):
    positions = np.zeros((n_particles,3))
    current_position = np.array([box_size/2, box_size/2, box_size/2])
    positions[0] = current_position
    
    for i in range(1, n_particles):
        direction = np.random.randn(3)
        direction /= np.linalg.norm(direction) # Generate a random unit vector

        # Compute the next position
        next_position = current_position + r0 * direction
        positions[i] = apply_pbc(next_position, box_size) # Apply periodic boundary conditions
        # Update the current position
        current_position = positions[i]
    
    return positions 

# %%
# Initialize velocities
def initialize_velocities(n_particles, target_temperature, mass):
    # Standard deviation of velocities for the Maxwell-Boltzmann distribution
    std_dev = np.sqrt(k_B * target_temperature / mass) 
    velocities = np.random.normal(0, std_dev, (n_particles, 3)) # Generate random velocities from a normal distribution
    velocities -=np.mean(velocities, axis=0)
    
    return velocities   

# %%
# Apply the minimum image convention to a displacement vector
def minimum_image(displacement, box_size):
    return displacement - np.round(displacement / box_size) * box_size

# %%
# Compute harmonic forces
def compute_harmonic_forces(positions, k, r0, box_size):
   
    forces = np.zeros_like(positions)
    n_praticles = positions.shape[0]

    for i in range(n_particles - 1):
        # Compute the displacement vector
        displacement = positions[i + 1] - positions[i]
        displacement = minimum_image(displacement, box_size)
        distance = np.linalg.norm(displacement) # Compute the distance between the particles
        
        # Compute the force magnitude
        force_magnitude = -k * (distance - r0)
        # Compute the force vector
        force = force_magnitude * (displacement / distance)
        
        # Update the forces on the particles
        forces[i] -= force
        forces[i+1] += force
    
    return forces

# %%
# Compute harmonic forces
def compute_lennard_jones_forces(positions, epsilon, sigma, box_size, interaction_type):
    
    n_particles = positions.shape[0]
    forces = np.zeros_like(positions)
    cutoff = 2**(1/6) * sigma # Lennard-Jones cutoff radius
     
    for i in range(n_particles):
        for j in range(i + 1, n_particles): # Determine the appropriate epsilon based on interaction type
            if interaction_type == 'repulsive' and abs(i - j) == 2:
                epsilon = epsilon_repulsive
            elif interaction_type == 'attractive' and abs(i -j) > 2:
                epsilon = epsilon_attractive
            else:
                continue

            #Compute the displacement and apply minimum image convention
            displacement = positions[j] - positions[i]
            displacemnet = minimum_image(displacement, box_size)
            distance = np.linalg.norm(displacement)

            if distance >= cutoff: # Skip if outside the cutoff
                continue

            #Compute Lennard-Jones force magnitude and vector
            lj_factor = (sigma / distance)
            force_magnitude = 24 * epsilon * ((lj_factor**12) - 0.5 * (lj_factor**6)) / distance
            force = force_magnitude * (displacement / distance)

             # Update forces
            forces[i] -= force
            forces[j] += force
    
    return forces


# %%
# Compute potential energy
def compute_potential_energy(positions, k, r0, epsilon_repulsive, epsilon_attractive, sigma, box_size):
    
    n_particles = positions.shape[0]
    
    # Harmonic potential energy
    harmonic_energy = 0.0
    for i in range(n_particles - 1):
        displacement = positions[i + 1] - positions[i]
        displacement = minimum_image(displacement, box_size)
        distance = np.linalg.norm(displacement)
        harmonic_energy += 0.5 * k * (distance - r0)**2

    # Lennard-Jones potential energy
    lj_energy = 0.0
    cutoff = 2**(1/6) * sigma
    for i in range(n_particles):
        for j in range(i + 1, n_particles):
            
            displacement = positions[j] - positions[i]
            displacement = minimum_image(displacement, box_size)
            distance = np.linalg.norm(displacement)
            
            if distance >= cutoff:
                continue
            
            # Repulsive or attractive interaction
            if abs(i - j) == 2:  # Repulsive
                epsilon = epsilon_repulsive
            elif abs(i - j) > 2:  # Attractive
                epsilon = epsilon_attractive
            else:
                continue
            
            lj_factor = (sigma / distance)
            lj_energy += 4 * epsilon * ((lj_factor**12) - (lj_factor**6))

    return harmonic_energy + lj_energy # Return total potential energy


# %%
# Velocity Verlet Integration
def velocity_verlet(positions, velocities, forces, dt, mass):
    velocities += 0.5 * forces / mass * dt # Update velocities
    positions += velocities * dt # Update positions
    positions = apply_pbc(positions, box_size) # Apply periodic boundary conditions

    # Compute new foreces
    forces_harmonic = compute_harmonic_forces(positions, k, r0, box_size)
    forces_repulsive = compute_lennard_jones_forces(positions, epsilon_repulsive, sigma, box_size, 'repulsive')
    forces_attractive = compute_lennard_jones_forces(positions, epsilon_attractive, sigma, box_size, 'attractive')
    forces_new = forces_harmonic + forces_repulsive + forces_attractive

    velocities += 0.5 * forces_new / mass *dt
    
    return positions, velocities, forces_new

# %%
# Velocity Rescaling Thermostat
def rescale_velocities(velocities, target_temperature, mass):
    # Compute the kinetic energy
    kinetic_energy = 0.5 * mass * np.sum(np.linalg.norm(velocities, axis = 1)**2)
    # Calculate the current temperature
    n_particles = velocities.shape[0]
    current_temperature = (2/3) * kinetic_energy / (n_particles *k_B)

    # Determine the scaling factor
    scaling_factor = np.sqrt(target_temperature / current_temperature)
    # Rescale the velocities
    velocities *= scaling_factor
   
    return velocities

# %%
# Calculate Radius of Gyration
def calculate_radius_of_gyration(positions):
    center_of_mass = np.mean(positions, axis = 0)
    Rg_squared = np.mean(np.sum((positions - center_of_mass)**2, axis = 1))
    Rg = np.sqrt(Rg_squared)
    return Rg

# %%
# Calculate End-to-End Distance
def calculate_end_to_end_distance(positions):
    Ree = np.linalg.norm(positions[-1]-positions[0])
    return Ree

# %%
# Simulation: k = 1.0, epsilon repulsive = 1.0
# Simulation parameters
dt = 0.01  # Time step
total_steps = 10000  # Number of steps
box_size = 100.0  # Size of the cubic box
mass = 1.0  # Particle mass
r0 = 1.0  # Equilibrium bond length
rescale_interval = 100  # Steps between velocity rescaling
n_particles = 20  # Number of particles
epsilon_attractive = 0.5  # Depth of attractive LJ potential
sigma = 1.0  # LJ potential parameter
temperatures = np.linspace(0.1, 1.0, 10)

# Variable Parameters: identify values of K and epsilon repulsive that prevent folding at low temperatures
k = 1.0  # Spring constant
epsilon_repulsive = 1.0  # Depth of repulsive LJ potential

# Arrays to store properties
Rg_values = []
Ree_values = []
potential_energies = []

for T in temperatures:
    # Set target temperature
    target_temperature = T
    # Re-initialize positions and velocities
    positions = initialize_chain(n_particles, box_size, r0)
    velocities = initialize_velocities(n_particles, T, mass)
    
    # Run simulation
    positions_history = []
    potential_energy_sum = 0
    for step in range(total_steps):
        forces_harmonic = compute_harmonic_forces(positions, k, r0, box_size)
        forces_repulsive = compute_lennard_jones_forces(positions, epsilon_repulsive, sigma, box_size, 'repulsive')
        forces_attractive = compute_lennard_jones_forces(positions, epsilon_attractive, sigma, box_size, 'attractive')
        total_forces = forces_harmonic + forces_repulsive + forces_attractive

        positions, velocities, total_forces = velocity_verlet(positions, velocities, total_forces, dt, mass)

        if step % rescale_interval == 0:
            velocities = rescale_velocities(velocities, T, mass)
            
            positions_history.append(positions.copy())

        # Accumulate potential energy 
        potential_energy_sum += compute_potential_energy(positions, k, r0, epsilon_repulsive, epsilon_attractive, sigma, box_size)
   
    # Compute properties
    Rg = calculate_radius_of_gyration(positions)
    Ree = calculate_end_to_end_distance(positions)
    Rg_values.append(Rg)
    Ree_values.append(Ree)
    potential_energies.append(np.mean(potential_energy_sum))
    
    # Plot the final configurations of the polymer chain
    final_positions = positions_history[-1]
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(final_positions[:, 0], final_positions[:, 1], final_positions[:, 2], '-o', color ='grey', markerfacecolor ='red', markeredgecolor ='red', label='Polymer Chain')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Final Configuration of Polymer Chain (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}, T = {T:.2f} K)')
    ax.legend()
    plt.savefig(f'Final Configuration of Polymer Chain (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}, T = {T:.2f} K).png')
    plt.show()
    
    # Create an animation to show the evolution of polymer chain configurations
    def update(frame):
        ax.clear()
        positions = positions_history[frame]
    
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], '-o',color ='grey', markerfacecolor ='red', markeredgecolor ='red', label=f'Step {frame * rescale_interval}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Evolution of Polymer Chain (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}, T = {T:.2f} K)')
        ax.legend()
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    polymer_chain_conformation_animation = animation.FuncAnimation(fig, update, frames=len(positions_history), interval=100)
    polymer_chain_conformation_animation.save(f'Evolution of Polymer Chain Configuration (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}, T = {T:.2f} K).gif', fps=10, writer = 'pillow')
    plt.show()

# Plot Radius of Gyration vs Temperature
plt.figure()
plt.plot(temperatures, Rg_values, label='Radius of Gyration')
plt.xlabel('Temperature (K)')
plt.ylabel('Radius of Gyration (Å)')
plt.title(f'Radius of Gyration vs Temperature (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f})')
plt.legend()
plt.savefig(f'Radius of Gyration vs Temperature (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}).png')
plt.show()

# Plot End-to-End Distance vs Temperature
plt.figure()
plt.plot(temperatures, Ree_values, label='End-to-End Distance')
plt.xlabel('Temperature (K)')
plt.ylabel('End-to-End Distance (Å)')
plt.title(f'End-to-End Distance vs Temperature (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f})')
plt.legend()
plt.savefig(f'End-to-End Distance vs Temperature (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}).png')
plt.show()

# Plot Potential Energy vs Temperature
plt.figure()
plt.plot(temperatures, potential_energies, label='Potential Energy')
plt.xlabel('Temperature (K)')
plt.ylabel('Potential Energy')
plt.title(f'Potential Energy vs Temperature (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f})')
plt.legend()
plt.savefig(f'Potential Energy vs Temperature (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}).png')
plt.show()

# %%
# Simulation: k = 1.0, epsilon repulsive = 0.5
# Simulation parameters
dt = 0.01  # Time step
total_steps = 10000  # Number of steps
box_size = 100.0  # Size of the cubic box
mass = 1.0  # Particle mass
r0 = 1.0  # Equilibrium bond length
rescale_interval = 100  # Steps between velocity rescaling
n_particles = 20  # Number of particles
epsilon_attractive = 0.5  # Depth of attractive LJ potential
sigma = 1.0  # LJ potential parameter
temperatures = np.linspace(0.1, 1.0, 10)

# Variable Parameters: identify values of K and epsilon repulsive that prevent folding at low temperatures
k = 1.0  # Spring constant
epsilon_repulsive = 0.5  # Depth of repulsive LJ potential

# Arrays to store properties
Rg_values = []
Ree_values = []
potential_energies = []

for T in temperatures:
    # Set target temperature
    target_temperature = T
    # Re-initialize positions and velocities
    positions = initialize_chain(n_particles, box_size, r0)
    velocities = initialize_velocities(n_particles, T, mass)
    
    # Run simulation
    positions_history = []
    potential_energy_sum = 0
    for step in range(total_steps):
        forces_harmonic = compute_harmonic_forces(positions, k, r0, box_size)
        forces_repulsive = compute_lennard_jones_forces(positions, epsilon_repulsive, sigma, box_size, 'repulsive')
        forces_attractive = compute_lennard_jones_forces(positions, epsilon_attractive, sigma, box_size, 'attractive')
        total_forces = forces_harmonic + forces_repulsive + forces_attractive

        positions, velocities, total_forces = velocity_verlet(positions, velocities, total_forces, dt, mass)

        if step % rescale_interval == 0:
            velocities = rescale_velocities(velocities, T, mass)
            
            positions_history.append(positions.copy())

        # Accumulate potential energy 
        potential_energy_sum += compute_potential_energy(positions, k, r0, epsilon_repulsive, epsilon_attractive, sigma, box_size)
   
    # Compute properties
    Rg = calculate_radius_of_gyration(positions)
    Ree = calculate_end_to_end_distance(positions)
    Rg_values.append(Rg)
    Ree_values.append(Ree)
    potential_energies.append(np.mean(potential_energy_sum))
    
    # Plot the final configurations of the polymer chain
    final_positions = positions_history[-1]
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(final_positions[:, 0], final_positions[:, 1], final_positions[:, 2], '-o', color ='grey', markerfacecolor ='red', markeredgecolor ='red', label='Polymer Chain')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Final Configuration of Polymer Chain (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}, T = {T:.2f} K)')
    ax.legend()
    plt.savefig(f'Final Configuration of Polymer Chain (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}, T = {T:.2f} K).png')
    plt.show()
    
    # Create an animation to show the evolution of polymer chain configurations
    def update(frame):
        ax.clear()
        positions = positions_history[frame]
    
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], '-o',color ='grey', markerfacecolor ='red', markeredgecolor ='red', label=f'Step {frame * rescale_interval}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Evolution of Polymer Chain (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}, T = {T:.2f} K)')
        ax.legend()
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    polymer_chain_conformation_animation = animation.FuncAnimation(fig, update, frames=len(positions_history), interval=100)
    polymer_chain_conformation_animation.save(f'Evolution of Polymer Chain Configuration (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}, T = {T:.2f} K).gif', fps=10, writer = 'pillow')
    plt.show()

# Plot Radius of Gyration vs Temperature
plt.figure()
plt.plot(temperatures, Rg_values, label='Radius of Gyration')
plt.xlabel('Temperature (K)')
plt.ylabel('Radius of Gyration (Å)')
plt.title(f'Radius of Gyration vs Temperature (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f})')
plt.legend()
plt.savefig(f'Radius of Gyration vs Temperature (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}).png')
plt.show()

# Plot End-to-End Distance vs Temperature
plt.figure()
plt.plot(temperatures, Ree_values, label='End-to-End Distance')
plt.xlabel('Temperature (K)')
plt.ylabel('End-to-End Distance (Å)')
plt.title(f'End-to-End Distance vs Temperature (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f})')
plt.legend()
plt.savefig(f'End-to-End Distance vs Temperature (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}).png')
plt.show()

# Plot Potential Energy vs Temperature
plt.figure()
plt.plot(temperatures, potential_energies, label='Potential Energy')
plt.xlabel('Temperature (K)')
plt.ylabel('Potential Energy')
plt.title(f'Potential Energy vs Temperature (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f})')
plt.legend()
plt.savefig(f'Potential Energy vs Temperature (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}).png')
plt.show()

# %%
# Simulation: k = 1.0, epsilon repulsive = 0
# Simulation parameters
dt = 0.01  # Time step
total_steps = 10000  # Number of steps
box_size = 100.0  # Size of the cubic box
mass = 1.0  # Particle mass
r0 = 1.0  # Equilibrium bond length
rescale_interval = 100  # Steps between velocity rescaling
n_particles = 20  # Number of particles
epsilon_attractive = 0.5  # Depth of attractive LJ potential
sigma = 1.0  # LJ potential parameter
temperatures = np.linspace(0.1, 1.0, 10)

# Variable Parameters: identify values of K and epsilon repulsive that prevent folding at low temperatures
k = 1.0  # Spring constant
epsilon_repulsive = 0  # Depth of repulsive LJ potential

# Arrays to store properties
Rg_values = []
Ree_values = []
potential_energies = []

for T in temperatures:
    # Set target temperature
    target_temperature = T
    # Re-initialize positions and velocities
    positions = initialize_chain(n_particles, box_size, r0)
    velocities = initialize_velocities(n_particles, T, mass)
    
    # Run simulation
    positions_history = []
    potential_energy_sum = 0
    for step in range(total_steps):
        forces_harmonic = compute_harmonic_forces(positions, k, r0, box_size)
        forces_repulsive = compute_lennard_jones_forces(positions, epsilon_repulsive, sigma, box_size, 'repulsive')
        forces_attractive = compute_lennard_jones_forces(positions, epsilon_attractive, sigma, box_size, 'attractive')
        total_forces = forces_harmonic + forces_repulsive + forces_attractive

        positions, velocities, total_forces = velocity_verlet(positions, velocities, total_forces, dt, mass)

        if step % rescale_interval == 0:
            velocities = rescale_velocities(velocities, T, mass)
            
            positions_history.append(positions.copy())

        # Accumulate potential energy 
        potential_energy_sum += compute_potential_energy(positions, k, r0, epsilon_repulsive, epsilon_attractive, sigma, box_size)
   
    # Compute properties
    Rg = calculate_radius_of_gyration(positions)
    Ree = calculate_end_to_end_distance(positions)
    Rg_values.append(Rg)
    Ree_values.append(Ree)
    potential_energies.append(np.mean(potential_energy_sum))
    
    # Plot the final configurations of the polymer chain
    final_positions = positions_history[-1]
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(final_positions[:, 0], final_positions[:, 1], final_positions[:, 2], '-o', color ='grey', markerfacecolor ='red', markeredgecolor ='red', label='Polymer Chain')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Final Configuration of Polymer Chain (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}, T = {T:.2f} K)')
    ax.legend()
    plt.savefig(f'Final Configuration of Polymer Chain (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}, T = {T:.2f} K).png')
    plt.show()
    
    # Create an animation to show the evolution of polymer chain configurations
    def update(frame):
        ax.clear()
        positions = positions_history[frame]
    
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], '-o',color ='grey', markerfacecolor ='red', markeredgecolor ='red', label=f'Step {frame * rescale_interval}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Evolution of Polymer Chain (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}, T = {T:.2f} K)')
        ax.legend()
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    polymer_chain_conformation_animation = animation.FuncAnimation(fig, update, frames=len(positions_history), interval=100)
    polymer_chain_conformation_animation.save(f'Evolution of Polymer Chain Configuration (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}, T = {T:.2f} K).gif', fps=10, writer = 'pillow')
    plt.show()

# Plot Radius of Gyration vs Temperature
plt.figure()
plt.plot(temperatures, Rg_values, label='Radius of Gyration')
plt.xlabel('Temperature (K)')
plt.ylabel('Radius of Gyration (Å)')
plt.title(f'Radius of Gyration vs Temperature (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f})')
plt.legend()
plt.savefig(f'Radius of Gyration vs Temperature (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}).png')
plt.show()

# Plot End-to-End Distance vs Temperature
plt.figure()
plt.plot(temperatures, Ree_values, label='End-to-End Distance')
plt.xlabel('Temperature (K)')
plt.ylabel('End-to-End Distance (Å)')
plt.title(f'End-to-End Distance vs Temperature (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f})')
plt.legend()
plt.savefig(f'End-to-End Distance vs Temperature (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}).png')
plt.show()

# Plot Potential Energy vs Temperature
plt.figure()
plt.plot(temperatures, potential_energies, label='Potential Energy')
plt.xlabel('Temperature (K)')
plt.ylabel('Potential Energy')
plt.title(f'Potential Energy vs Temperature (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f})')
plt.legend()
plt.savefig(f'Potential Energy vs Temperature (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}).png')
plt.show()

# %%
# Simulation: k = 1.0, epsilon repulsive = 1.5
# Simulation parameters
dt = 0.01  # Time step
total_steps = 10000  # Number of steps
box_size = 100.0  # Size of the cubic box
mass = 1.0  # Particle mass
r0 = 1.0  # Equilibrium bond length
rescale_interval = 100  # Steps between velocity rescaling
n_particles = 20  # Number of particles
epsilon_attractive = 0.5  # Depth of attractive LJ potential
sigma = 1.0  # LJ potential parameter
temperatures = np.linspace(0.1, 1.0, 10)

# Variable Parameters: identify values of K and epsilon repulsive that prevent folding at low temperatures
k = 1.0  # Spring constant
epsilon_repulsive = 1.5  # Depth of repulsive LJ potential

# Arrays to store properties
Rg_values = []
Ree_values = []
potential_energies = []

for T in temperatures:
    # Set target temperature
    target_temperature = T
    # Re-initialize positions and velocities
    positions = initialize_chain(n_particles, box_size, r0)
    velocities = initialize_velocities(n_particles, T, mass)
    
    # Run simulation
    positions_history = []
    potential_energy_sum = 0
    for step in range(total_steps):
        forces_harmonic = compute_harmonic_forces(positions, k, r0, box_size)
        forces_repulsive = compute_lennard_jones_forces(positions, epsilon_repulsive, sigma, box_size, 'repulsive')
        forces_attractive = compute_lennard_jones_forces(positions, epsilon_attractive, sigma, box_size, 'attractive')
        total_forces = forces_harmonic + forces_repulsive + forces_attractive

        positions, velocities, total_forces = velocity_verlet(positions, velocities, total_forces, dt, mass)

        if step % rescale_interval == 0:
            velocities = rescale_velocities(velocities, T, mass)
            
            positions_history.append(positions.copy())

        # Accumulate potential energy 
        potential_energy_sum += compute_potential_energy(positions, k, r0, epsilon_repulsive, epsilon_attractive, sigma, box_size)
   
    # Compute properties
    Rg = calculate_radius_of_gyration(positions)
    Ree = calculate_end_to_end_distance(positions)
    Rg_values.append(Rg)
    Ree_values.append(Ree)
    potential_energies.append(np.mean(potential_energy_sum))
    
    # Plot the final configurations of the polymer chain
    final_positions = positions_history[-1]
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(final_positions[:, 0], final_positions[:, 1], final_positions[:, 2], '-o', color ='grey', markerfacecolor ='red', markeredgecolor ='red', label='Polymer Chain')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Final Configuration of Polymer Chain (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}, T = {T:.2f} K)')
    ax.legend()
    plt.savefig(f'Final Configuration of Polymer Chain (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}, T = {T:.2f} K).png')
    plt.show()
    
    # Create an animation to show the evolution of polymer chain configurations
    def update(frame):
        ax.clear()
        positions = positions_history[frame]
    
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], '-o',color ='grey', markerfacecolor ='red', markeredgecolor ='red', label=f'Step {frame * rescale_interval}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Evolution of Polymer Chain (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}, T = {T:.2f} K)')
        ax.legend()
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    polymer_chain_conformation_animation = animation.FuncAnimation(fig, update, frames=len(positions_history), interval=100)
    polymer_chain_conformation_animation.save(f'Evolution of Polymer Chain Configuration (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}, T = {T:.2f} K).gif', fps=10, writer = 'pillow')
    plt.show()

# Plot Radius of Gyration vs Temperature
plt.figure()
plt.plot(temperatures, Rg_values, label='Radius of Gyration')
plt.xlabel('Temperature (K)')
plt.ylabel('Radius of Gyration (Å)')
plt.title(f'Radius of Gyration vs Temperature (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f})')
plt.legend()
plt.savefig(f'Radius of Gyration vs Temperature (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}).png')
plt.show()

# Plot End-to-End Distance vs Temperature
plt.figure()
plt.plot(temperatures, Ree_values, label='End-to-End Distance')
plt.xlabel('Temperature (K)')
plt.ylabel('End-to-End Distance (Å)')
plt.title(f'End-to-End Distance vs Temperature (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f})')
plt.legend()
plt.savefig(f'End-to-End Distance vs Temperature (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}).png')
plt.show()

# Plot Potential Energy vs Temperature
plt.figure()
plt.plot(temperatures, potential_energies, label='Potential Energy')
plt.xlabel('Temperature (K)')
plt.ylabel('Potential Energy')
plt.title(f'Potential Energy vs Temperature (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f})')
plt.legend()
plt.savefig(f'Potential Energy vs Temperature (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}).png')
plt.show()

# %%
# Simulation: k = 1.0, epsilon repulsive = 2
# Simulation parameters
dt = 0.01  # Time step
total_steps = 10000  # Number of steps
box_size = 100.0  # Size of the cubic box
mass = 1.0  # Particle mass
r0 = 1.0  # Equilibrium bond length
rescale_interval = 100  # Steps between velocity rescaling
n_particles = 20  # Number of particles
epsilon_attractive = 0.5  # Depth of attractive LJ potential
sigma = 1.0  # LJ potential parameter
temperatures = np.linspace(0.1, 1.0, 10)

# Variable Parameters: identify values of K and epsilon repulsive that prevent folding at low temperatures
k = 1.0  # Spring constant
epsilon_repulsive = 2  # Depth of repulsive LJ potential

# Arrays to store properties
Rg_values = []
Ree_values = []
potential_energies = []

for T in temperatures:
    # Set target temperature
    target_temperature = T
    # Re-initialize positions and velocities
    positions = initialize_chain(n_particles, box_size, r0)
    velocities = initialize_velocities(n_particles, T, mass)
    
    # Run simulation
    positions_history = []
    potential_energy_sum = 0
    for step in range(total_steps):
        forces_harmonic = compute_harmonic_forces(positions, k, r0, box_size)
        forces_repulsive = compute_lennard_jones_forces(positions, epsilon_repulsive, sigma, box_size, 'repulsive')
        forces_attractive = compute_lennard_jones_forces(positions, epsilon_attractive, sigma, box_size, 'attractive')
        total_forces = forces_harmonic + forces_repulsive + forces_attractive

        positions, velocities, total_forces = velocity_verlet(positions, velocities, total_forces, dt, mass)

        if step % rescale_interval == 0:
            velocities = rescale_velocities(velocities, T, mass)
            
            positions_history.append(positions.copy())

        # Accumulate potential energy 
        potential_energy_sum += compute_potential_energy(positions, k, r0, epsilon_repulsive, epsilon_attractive, sigma, box_size)
   
    # Compute properties
    Rg = calculate_radius_of_gyration(positions)
    Ree = calculate_end_to_end_distance(positions)
    Rg_values.append(Rg)
    Ree_values.append(Ree)
    potential_energies.append(np.mean(potential_energy_sum))
    
    # Plot the final configurations of the polymer chain
    final_positions = positions_history[-1]
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(final_positions[:, 0], final_positions[:, 1], final_positions[:, 2], '-o', color ='grey', markerfacecolor ='red', markeredgecolor ='red', label='Polymer Chain')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Final Configuration of Polymer Chain (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}, T = {T:.2f} K)')
    ax.legend()
    plt.savefig(f'Final Configuration of Polymer Chain (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}, T = {T:.2f} K).png')
    plt.show()
    
    # Create an animation to show the evolution of polymer chain configurations
    def update(frame):
        ax.clear()
        positions = positions_history[frame]
    
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], '-o',color ='grey', markerfacecolor ='red', markeredgecolor ='red', label=f'Step {frame * rescale_interval}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Evolution of Polymer Chain (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}, T = {T:.2f} K)')
        ax.legend()
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    polymer_chain_conformation_animation = animation.FuncAnimation(fig, update, frames=len(positions_history), interval=100)
    polymer_chain_conformation_animation.save(f'Evolution of Polymer Chain Configuration (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}, T = {T:.2f} K).gif', fps=10, writer = 'pillow')
    plt.show()

# Plot Radius of Gyration vs Temperature
plt.figure()
plt.plot(temperatures, Rg_values, label='Radius of Gyration')
plt.xlabel('Temperature (K)')
plt.ylabel('Radius of Gyration (Å)')
plt.title(f'Radius of Gyration vs Temperature (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f})')
plt.legend()
plt.savefig(f'Radius of Gyration vs Temperature (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}).png')
plt.show()

# Plot End-to-End Distance vs Temperature
plt.figure()
plt.plot(temperatures, Ree_values, label='End-to-End Distance')
plt.xlabel('Temperature (K)')
plt.ylabel('End-to-End Distance (Å)')
plt.title(f'End-to-End Distance vs Temperature (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f})')
plt.legend()
plt.savefig(f'End-to-End Distance vs Temperature (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}).png')
plt.show()

# Plot Potential Energy vs Temperature
plt.figure()
plt.plot(temperatures, potential_energies, label='Potential Energy')
plt.xlabel('Temperature (K)')
plt.ylabel('Potential Energy')
plt.title(f'Potential Energy vs Temperature (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f})')
plt.legend()
plt.savefig(f'Potential Energy vs Temperature (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}).png')
plt.show()

# %%
# Simulation: k = 1.0, epsilon repulsive = 5
# Simulation parameters
dt = 0.01  # Time step
total_steps = 10000  # Number of steps
box_size = 100.0  # Size of the cubic box
mass = 1.0  # Particle mass
r0 = 1.0  # Equilibrium bond length
rescale_interval = 100  # Steps between velocity rescaling
n_particles = 20  # Number of particles
epsilon_attractive = 0.5  # Depth of attractive LJ potential
sigma = 1.0  # LJ potential parameter
temperatures = np.linspace(0.1, 1.0, 10)

# Variable Parameters: identify values of K and epsilon repulsive that prevent folding at low temperatures
k = 1.0  # Spring constant
epsilon_repulsive = 5  # Depth of repulsive LJ potential

# Arrays to store properties
Rg_values = []
Ree_values = []
potential_energies = []

for T in temperatures:
    # Set target temperature
    target_temperature = T
    # Re-initialize positions and velocities
    positions = initialize_chain(n_particles, box_size, r0)
    velocities = initialize_velocities(n_particles, T, mass)
    
    # Run simulation
    positions_history = []
    potential_energy_sum = 0
    for step in range(total_steps):
        forces_harmonic = compute_harmonic_forces(positions, k, r0, box_size)
        forces_repulsive = compute_lennard_jones_forces(positions, epsilon_repulsive, sigma, box_size, 'repulsive')
        forces_attractive = compute_lennard_jones_forces(positions, epsilon_attractive, sigma, box_size, 'attractive')
        total_forces = forces_harmonic + forces_repulsive + forces_attractive

        positions, velocities, total_forces = velocity_verlet(positions, velocities, total_forces, dt, mass)

        if step % rescale_interval == 0:
            velocities = rescale_velocities(velocities, T, mass)
            
            positions_history.append(positions.copy())

        # Accumulate potential energy 
        potential_energy_sum += compute_potential_energy(positions, k, r0, epsilon_repulsive, epsilon_attractive, sigma, box_size)
   
    # Compute properties
    Rg = calculate_radius_of_gyration(positions)
    Ree = calculate_end_to_end_distance(positions)
    Rg_values.append(Rg)
    Ree_values.append(Ree)
    potential_energies.append(np.mean(potential_energy_sum))
    
    # Plot the final configurations of the polymer chain
    final_positions = positions_history[-1]
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(final_positions[:, 0], final_positions[:, 1], final_positions[:, 2], '-o', color ='grey', markerfacecolor ='red', markeredgecolor ='red', label='Polymer Chain')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Final Configuration of Polymer Chain (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}, T = {T:.2f} K)')
    ax.legend()
    plt.savefig(f'Final Configuration of Polymer Chain (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}, T = {T:.2f} K).png')
    plt.show()
    
    # Create an animation to show the evolution of polymer chain configurations
    def update(frame):
        ax.clear()
        positions = positions_history[frame]
    
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], '-o',color ='grey', markerfacecolor ='red', markeredgecolor ='red', label=f'Step {frame * rescale_interval}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Evolution of Polymer Chain (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}, T = {T:.2f} K)')
        ax.legend()
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    polymer_chain_conformation_animation = animation.FuncAnimation(fig, update, frames=len(positions_history), interval=100)
    polymer_chain_conformation_animation.save(f'Evolution of Polymer Chain Configuration (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}, T = {T:.2f} K).gif', fps=10, writer = 'pillow')
    plt.show()

# Plot Radius of Gyration vs Temperature
plt.figure()
plt.plot(temperatures, Rg_values, label='Radius of Gyration')
plt.xlabel('Temperature (K)')
plt.ylabel('Radius of Gyration (Å)')
plt.title(f'Radius of Gyration vs Temperature (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f})')
plt.legend()
plt.savefig(f'Radius of Gyration vs Temperature (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}).png')
plt.show()

# Plot End-to-End Distance vs Temperature
plt.figure()
plt.plot(temperatures, Ree_values, label='End-to-End Distance')
plt.xlabel('Temperature (K)')
plt.ylabel('End-to-End Distance (Å)')
plt.title(f'End-to-End Distance vs Temperature (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f})')
plt.legend()
plt.savefig(f'End-to-End Distance vs Temperature (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}).png')
plt.show()

# Plot Potential Energy vs Temperature
plt.figure()
plt.plot(temperatures, potential_energies, label='Potential Energy')
plt.xlabel('Temperature (K)')
plt.ylabel('Potential Energy')
plt.title(f'Potential Energy vs Temperature (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f})')
plt.legend()
plt.savefig(f'Potential Energy vs Temperature (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}).png')
plt.show()

# %%
# Simulation: k = 0.5, epsilon repulsive = 1
# Simulation parameters
dt = 0.01  # Time step
total_steps = 10000  # Number of steps
box_size = 100.0  # Size of the cubic box
mass = 1.0  # Particle mass
r0 = 1.0  # Equilibrium bond length
rescale_interval = 100  # Steps between velocity rescaling
n_particles = 20  # Number of particles
epsilon_attractive = 0.5  # Depth of attractive LJ potential
sigma = 1.0  # LJ potential parameter
temperatures = np.linspace(0.1, 1.0, 10)

# Variable Parameters: identify values of K and epsilon repulsive that prevent folding at low temperatures
k = 0.5  # Spring constant
epsilon_repulsive = 1  # Depth of repulsive LJ potential

# Arrays to store properties
Rg_values = []
Ree_values = []
potential_energies = []

for T in temperatures:
    # Set target temperature
    target_temperature = T
    # Re-initialize positions and velocities
    positions = initialize_chain(n_particles, box_size, r0)
    velocities = initialize_velocities(n_particles, T, mass)
    
    # Run simulation
    positions_history = []
    potential_energy_sum = 0
    for step in range(total_steps):
        forces_harmonic = compute_harmonic_forces(positions, k, r0, box_size)
        forces_repulsive = compute_lennard_jones_forces(positions, epsilon_repulsive, sigma, box_size, 'repulsive')
        forces_attractive = compute_lennard_jones_forces(positions, epsilon_attractive, sigma, box_size, 'attractive')
        total_forces = forces_harmonic + forces_repulsive + forces_attractive

        positions, velocities, total_forces = velocity_verlet(positions, velocities, total_forces, dt, mass)

        if step % rescale_interval == 0:
            velocities = rescale_velocities(velocities, T, mass)
            
            positions_history.append(positions.copy())

        # Accumulate potential energy 
        potential_energy_sum += compute_potential_energy(positions, k, r0, epsilon_repulsive, epsilon_attractive, sigma, box_size)
   
    # Compute properties
    Rg = calculate_radius_of_gyration(positions)
    Ree = calculate_end_to_end_distance(positions)
    Rg_values.append(Rg)
    Ree_values.append(Ree)
    potential_energies.append(np.mean(potential_energy_sum))
    
    # Plot the final configurations of the polymer chain
    final_positions = positions_history[-1]
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(final_positions[:, 0], final_positions[:, 1], final_positions[:, 2], '-o', color ='grey', markerfacecolor ='red', markeredgecolor ='red', label='Polymer Chain')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Final Configuration of Polymer Chain (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}, T = {T:.2f} K)')
    ax.legend()
    plt.savefig(f'Final Configuration of Polymer Chain (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}, T = {T:.2f} K).png')
    plt.show()
    
    # Create an animation to show the evolution of polymer chain configurations
    def update(frame):
        ax.clear()
        positions = positions_history[frame]
    
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], '-o',color ='grey', markerfacecolor ='red', markeredgecolor ='red', label=f'Step {frame * rescale_interval}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Evolution of Polymer Chain (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}, T = {T:.2f} K)')
        ax.legend()
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    polymer_chain_conformation_animation = animation.FuncAnimation(fig, update, frames=len(positions_history), interval=100)
    polymer_chain_conformation_animation.save(f'Evolution of Polymer Chain Configuration (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}, T = {T:.2f} K).gif', fps=10, writer = 'pillow')
    plt.show()

# Plot Radius of Gyration vs Temperature
plt.figure()
plt.plot(temperatures, Rg_values, label='Radius of Gyration')
plt.xlabel('Temperature (K)')
plt.ylabel('Radius of Gyration (Å)')
plt.title(f'Radius of Gyration vs Temperature (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f})')
plt.legend()
plt.savefig(f'Radius of Gyration vs Temperature (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}).png')
plt.show()

# Plot End-to-End Distance vs Temperature
plt.figure()
plt.plot(temperatures, Ree_values, label='End-to-End Distance')
plt.xlabel('Temperature (K)')
plt.ylabel('End-to-End Distance (Å)')
plt.title(f'End-to-End Distance vs Temperature (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f})')
plt.legend()
plt.savefig(f'End-to-End Distance vs Temperature (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}).png')
plt.show()

# Plot Potential Energy vs Temperature
plt.figure()
plt.plot(temperatures, potential_energies, label='Potential Energy')
plt.xlabel('Temperature (K)')
plt.ylabel('Potential Energy')
plt.title(f'Potential Energy vs Temperature (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f})')
plt.legend()
plt.savefig(f'Potential Energy vs Temperature (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}).png')
plt.show()

# %%
# Simulation: k = 0, epsilon repulsive = 1
# Simulation parameters
dt = 0.01  # Time step
total_steps = 10000  # Number of steps
box_size = 100.0  # Size of the cubic box
mass = 1.0  # Particle mass
r0 = 1.0  # Equilibrium bond length
rescale_interval = 100  # Steps between velocity rescaling
n_particles = 20  # Number of particles
epsilon_attractive = 0.5  # Depth of attractive LJ potential
sigma = 1.0  # LJ potential parameter
temperatures = np.linspace(0.1, 1.0, 10)

# Variable Parameters: identify values of K and epsilon repulsive that prevent folding at low temperatures
k = 0  # Spring constant
epsilon_repulsive = 1  # Depth of repulsive LJ potential

# Arrays to store properties
Rg_values = []
Ree_values = []
potential_energies = []

for T in temperatures:
    # Set target temperature
    target_temperature = T
    # Re-initialize positions and velocities
    positions = initialize_chain(n_particles, box_size, r0)
    velocities = initialize_velocities(n_particles, T, mass)
    
    # Run simulation
    positions_history = []
    potential_energy_sum = 0
    for step in range(total_steps):
        forces_harmonic = compute_harmonic_forces(positions, k, r0, box_size)
        forces_repulsive = compute_lennard_jones_forces(positions, epsilon_repulsive, sigma, box_size, 'repulsive')
        forces_attractive = compute_lennard_jones_forces(positions, epsilon_attractive, sigma, box_size, 'attractive')
        total_forces = forces_harmonic + forces_repulsive + forces_attractive

        positions, velocities, total_forces = velocity_verlet(positions, velocities, total_forces, dt, mass)

        if step % rescale_interval == 0:
            velocities = rescale_velocities(velocities, T, mass)
            
            positions_history.append(positions.copy())

        # Accumulate potential energy 
        potential_energy_sum += compute_potential_energy(positions, k, r0, epsilon_repulsive, epsilon_attractive, sigma, box_size)
   
    # Compute properties
    Rg = calculate_radius_of_gyration(positions)
    Ree = calculate_end_to_end_distance(positions)
    Rg_values.append(Rg)
    Ree_values.append(Ree)
    potential_energies.append(np.mean(potential_energy_sum))
    
    # Plot the final configurations of the polymer chain
    final_positions = positions_history[-1]
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(final_positions[:, 0], final_positions[:, 1], final_positions[:, 2], '-o', color ='grey', markerfacecolor ='red', markeredgecolor ='red', label='Polymer Chain')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Final Configuration of Polymer Chain (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}, T = {T:.2f} K)')
    ax.legend()
    plt.savefig(f'Final Configuration of Polymer Chain (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}, T = {T:.2f} K).png')
    plt.show()
    
    # Create an animation to show the evolution of polymer chain configurations
    def update(frame):
        ax.clear()
        positions = positions_history[frame]
    
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], '-o',color ='grey', markerfacecolor ='red', markeredgecolor ='red', label=f'Step {frame * rescale_interval}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Evolution of Polymer Chain (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}, T = {T:.2f} K)')
        ax.legend()
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    polymer_chain_conformation_animation = animation.FuncAnimation(fig, update, frames=len(positions_history), interval=100)
    polymer_chain_conformation_animation.save(f'Evolution of Polymer Chain Configuration (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}, T = {T:.2f} K).gif', fps=10, writer = 'pillow')
    plt.show()

# Plot Radius of Gyration vs Temperature
plt.figure()
plt.plot(temperatures, Rg_values, label='Radius of Gyration')
plt.xlabel('Temperature (K)')
plt.ylabel('Radius of Gyration (Å)')
plt.title(f'Radius of Gyration vs Temperature (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f})')
plt.legend()
plt.savefig(f'Radius of Gyration vs Temperature (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}).png')
plt.show()

# Plot End-to-End Distance vs Temperature
plt.figure()
plt.plot(temperatures, Ree_values, label='End-to-End Distance')
plt.xlabel('Temperature (K)')
plt.ylabel('End-to-End Distance (Å)')
plt.title(f'End-to-End Distance vs Temperature (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f})')
plt.legend()
plt.savefig(f'End-to-End Distance vs Temperature (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}).png')
plt.show()

# Plot Potential Energy vs Temperature
plt.figure()
plt.plot(temperatures, potential_energies, label='Potential Energy')
plt.xlabel('Temperature (K)')
plt.ylabel('Potential Energy')
plt.title(f'Potential Energy vs Temperature (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f})')
plt.legend()
plt.savefig(f'Potential Energy vs Temperature (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}).png')
plt.show()

# %%
# Simulation: k = 0, epsilon repulsive = 0.5
# Simulation parameters
dt = 0.01  # Time step
total_steps = 10000  # Number of steps
box_size = 100.0  # Size of the cubic box
mass = 1.0  # Particle mass
r0 = 1.0  # Equilibrium bond length
rescale_interval = 100  # Steps between velocity rescaling
n_particles = 20  # Number of particles
epsilon_attractive = 0.5  # Depth of attractive LJ potential
sigma = 1.0  # LJ potential parameter
temperatures = np.linspace(0.1, 1.0, 10)

# Variable Parameters: identify values of K and epsilon repulsive that prevent folding at low temperatures
k = 0  # Spring constant
epsilon_repulsive = 0.5  # Depth of repulsive LJ potential

# Arrays to store properties
Rg_values = []
Ree_values = []
potential_energies = []

for T in temperatures:
    # Set target temperature
    target_temperature = T
    # Re-initialize positions and velocities
    positions = initialize_chain(n_particles, box_size, r0)
    velocities = initialize_velocities(n_particles, T, mass)
    
    # Run simulation
    positions_history = []
    potential_energy_sum = 0
    for step in range(total_steps):
        forces_harmonic = compute_harmonic_forces(positions, k, r0, box_size)
        forces_repulsive = compute_lennard_jones_forces(positions, epsilon_repulsive, sigma, box_size, 'repulsive')
        forces_attractive = compute_lennard_jones_forces(positions, epsilon_attractive, sigma, box_size, 'attractive')
        total_forces = forces_harmonic + forces_repulsive + forces_attractive

        positions, velocities, total_forces = velocity_verlet(positions, velocities, total_forces, dt, mass)

        if step % rescale_interval == 0:
            velocities = rescale_velocities(velocities, T, mass)
            
            positions_history.append(positions.copy())

        # Accumulate potential energy 
        potential_energy_sum += compute_potential_energy(positions, k, r0, epsilon_repulsive, epsilon_attractive, sigma, box_size)
   
    # Compute properties
    Rg = calculate_radius_of_gyration(positions)
    Ree = calculate_end_to_end_distance(positions)
    Rg_values.append(Rg)
    Ree_values.append(Ree)
    potential_energies.append(np.mean(potential_energy_sum))
    
    # Plot the final configurations of the polymer chain
    final_positions = positions_history[-1]
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(final_positions[:, 0], final_positions[:, 1], final_positions[:, 2], '-o', color ='grey', markerfacecolor ='red', markeredgecolor ='red', label='Polymer Chain')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Final Configuration of Polymer Chain (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}, T = {T:.2f} K)')
    ax.legend()
    plt.savefig(f'Final Configuration of Polymer Chain (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}, T = {T:.2f} K).png')
    plt.show()
    
    # Create an animation to show the evolution of polymer chain configurations
    def update(frame):
        ax.clear()
        positions = positions_history[frame]
    
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], '-o',color ='grey', markerfacecolor ='red', markeredgecolor ='red', label=f'Step {frame * rescale_interval}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Evolution of Polymer Chain (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}, T = {T:.2f} K)')
        ax.legend()
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    polymer_chain_conformation_animation = animation.FuncAnimation(fig, update, frames=len(positions_history), interval=100)
    polymer_chain_conformation_animation.save(f'Evolution of Polymer Chain Configuration (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}, T = {T:.2f} K).gif', fps=10, writer = 'pillow')
    plt.show()

# Plot Radius of Gyration vs Temperature
plt.figure()
plt.plot(temperatures, Rg_values, label='Radius of Gyration')
plt.xlabel('Temperature (K)')
plt.ylabel('Radius of Gyration (Å)')
plt.title(f'Radius of Gyration vs Temperature (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f})')
plt.legend()
plt.savefig(f'Radius of Gyration vs Temperature (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}).png')
plt.show()

# Plot End-to-End Distance vs Temperature
plt.figure()
plt.plot(temperatures, Ree_values, label='End-to-End Distance')
plt.xlabel('Temperature (K)')
plt.ylabel('End-to-End Distance (Å)')
plt.title(f'End-to-End Distance vs Temperature (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f})')
plt.legend()
plt.savefig(f'End-to-End Distance vs Temperature (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}).png')
plt.show()

# Plot Potential Energy vs Temperature
plt.figure()
plt.plot(temperatures, potential_energies, label='Potential Energy')
plt.xlabel('Temperature (K)')
plt.ylabel('Potential Energy')
plt.title(f'Potential Energy vs Temperature (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f})')
plt.legend()
plt.savefig(f'Potential Energy vs Temperature (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}).png')
plt.show()

# %%
# Simulation: k = 2, epsilon repulsive = 1
# Simulation parameters
dt = 0.01  # Time step
total_steps = 10000  # Number of steps
box_size = 100.0  # Size of the cubic box
mass = 1.0  # Particle mass
r0 = 1.0  # Equilibrium bond length
rescale_interval = 100  # Steps between velocity rescaling
n_particles = 20  # Number of particles
epsilon_attractive = 0.5  # Depth of attractive LJ potential
sigma = 1.0  # LJ potential parameter
temperatures = np.linspace(0.1, 1.0, 10)

# Variable Parameters: identify values of K and epsilon repulsive that prevent folding at low temperatures
k = 2  # Spring constant
epsilon_repulsive = 1  # Depth of repulsive LJ potential

# Arrays to store properties
Rg_values = []
Ree_values = []
potential_energies = []

for T in temperatures:
    # Set target temperature
    target_temperature = T
    # Re-initialize positions and velocities
    positions = initialize_chain(n_particles, box_size, r0)
    velocities = initialize_velocities(n_particles, T, mass)
    
    # Run simulation
    positions_history = []
    potential_energy_sum = 0
    for step in range(total_steps):
        forces_harmonic = compute_harmonic_forces(positions, k, r0, box_size)
        forces_repulsive = compute_lennard_jones_forces(positions, epsilon_repulsive, sigma, box_size, 'repulsive')
        forces_attractive = compute_lennard_jones_forces(positions, epsilon_attractive, sigma, box_size, 'attractive')
        total_forces = forces_harmonic + forces_repulsive + forces_attractive

        positions, velocities, total_forces = velocity_verlet(positions, velocities, total_forces, dt, mass)

        if step % rescale_interval == 0:
            velocities = rescale_velocities(velocities, T, mass)
            
            positions_history.append(positions.copy())

        # Accumulate potential energy 
        potential_energy_sum += compute_potential_energy(positions, k, r0, epsilon_repulsive, epsilon_attractive, sigma, box_size)
   
    # Compute properties
    Rg = calculate_radius_of_gyration(positions)
    Ree = calculate_end_to_end_distance(positions)
    Rg_values.append(Rg)
    Ree_values.append(Ree)
    potential_energies.append(np.mean(potential_energy_sum))
    
    # Plot the final configurations of the polymer chain
    final_positions = positions_history[-1]
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(final_positions[:, 0], final_positions[:, 1], final_positions[:, 2], '-o', color ='grey', markerfacecolor ='red', markeredgecolor ='red', label='Polymer Chain')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Final Configuration of Polymer Chain (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}, T = {T:.2f} K)')
    ax.legend()
    plt.savefig(f'Final Configuration of Polymer Chain (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}, T = {T:.2f} K).png')
    plt.show()
    
    # Create an animation to show the evolution of polymer chain configurations
    def update(frame):
        ax.clear()
        positions = positions_history[frame]
    
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], '-o',color ='grey', markerfacecolor ='red', markeredgecolor ='red', label=f'Step {frame * rescale_interval}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Evolution of Polymer Chain (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}, T = {T:.2f} K)')
        ax.legend()
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    polymer_chain_conformation_animation = animation.FuncAnimation(fig, update, frames=len(positions_history), interval=100)
    polymer_chain_conformation_animation.save(f'Evolution of Polymer Chain Configuration (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}, T = {T:.2f} K).gif', fps=10, writer = 'pillow')
    plt.show()

# Plot Radius of Gyration vs Temperature
plt.figure()
plt.plot(temperatures, Rg_values, label='Radius of Gyration')
plt.xlabel('Temperature (K)')
plt.ylabel('Radius of Gyration (Å)')
plt.title(f'Radius of Gyration vs Temperature (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f})')
plt.legend()
plt.savefig(f'Radius of Gyration vs Temperature (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}).png')
plt.show()

# Plot End-to-End Distance vs Temperature
plt.figure()
plt.plot(temperatures, Ree_values, label='End-to-End Distance')
plt.xlabel('Temperature (K)')
plt.ylabel('End-to-End Distance (Å)')
plt.title(f'End-to-End Distance vs Temperature (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f})')
plt.legend()
plt.savefig(f'End-to-End Distance vs Temperature (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}).png')
plt.show()

# Plot Potential Energy vs Temperature
plt.figure()
plt.plot(temperatures, potential_energies, label='Potential Energy')
plt.xlabel('Temperature (K)')
plt.ylabel('Potential Energy')
plt.title(f'Potential Energy vs Temperature (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f})')
plt.legend()
plt.savefig(f'Potential Energy vs Temperature (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}).png')
plt.show()

# %%
# Simulation: k = 5, epsilon repulsive = 1
# Simulation parameters
dt = 0.01  # Time step
total_steps = 10000  # Number of steps
box_size = 100.0  # Size of the cubic box
mass = 1.0  # Particle mass
r0 = 1.0  # Equilibrium bond length
rescale_interval = 100  # Steps between velocity rescaling
n_particles = 20  # Number of particles
epsilon_attractive = 0.5  # Depth of attractive LJ potential
sigma = 1.0  # LJ potential parameter
temperatures = np.linspace(0.1, 1.0, 10)

# Variable Parameters: identify values of K and epsilon repulsive that prevent folding at low temperatures
k = 5  # Spring constant
epsilon_repulsive = 1  # Depth of repulsive LJ potential

# Arrays to store properties
Rg_values = []
Ree_values = []
potential_energies = []

for T in temperatures:
    # Set target temperature
    target_temperature = T
    # Re-initialize positions and velocities
    positions = initialize_chain(n_particles, box_size, r0)
    velocities = initialize_velocities(n_particles, T, mass)
    
    # Run simulation
    positions_history = []
    potential_energy_sum = 0
    for step in range(total_steps):
        forces_harmonic = compute_harmonic_forces(positions, k, r0, box_size)
        forces_repulsive = compute_lennard_jones_forces(positions, epsilon_repulsive, sigma, box_size, 'repulsive')
        forces_attractive = compute_lennard_jones_forces(positions, epsilon_attractive, sigma, box_size, 'attractive')
        total_forces = forces_harmonic + forces_repulsive + forces_attractive

        positions, velocities, total_forces = velocity_verlet(positions, velocities, total_forces, dt, mass)

        if step % rescale_interval == 0:
            velocities = rescale_velocities(velocities, T, mass)
            
            positions_history.append(positions.copy())

        # Accumulate potential energy 
        potential_energy_sum += compute_potential_energy(positions, k, r0, epsilon_repulsive, epsilon_attractive, sigma, box_size)
   
    # Compute properties
    Rg = calculate_radius_of_gyration(positions)
    Ree = calculate_end_to_end_distance(positions)
    Rg_values.append(Rg)
    Ree_values.append(Ree)
    potential_energies.append(np.mean(potential_energy_sum))
    
    # Plot the final configurations of the polymer chain
    final_positions = positions_history[-1]
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(final_positions[:, 0], final_positions[:, 1], final_positions[:, 2], '-o', color ='grey', markerfacecolor ='red', markeredgecolor ='red', label='Polymer Chain')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Final Configuration of Polymer Chain (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}, T = {T:.2f} K)')
    ax.legend()
    plt.savefig(f'Final Configuration of Polymer Chain (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}, T = {T:.2f} K).png')
    plt.show()
    
    # Create an animation to show the evolution of polymer chain configurations
    def update(frame):
        ax.clear()
        positions = positions_history[frame]
    
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], '-o',color ='grey', markerfacecolor ='red', markeredgecolor ='red', label=f'Step {frame * rescale_interval}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Evolution of Polymer Chain (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}, T = {T:.2f} K)')
        ax.legend()
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    polymer_chain_conformation_animation = animation.FuncAnimation(fig, update, frames=len(positions_history), interval=100)
    polymer_chain_conformation_animation.save(f'Evolution of Polymer Chain Configuration (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}, T = {T:.2f} K).gif', fps=10, writer = 'pillow')
    plt.show()

# Plot Radius of Gyration vs Temperature
plt.figure()
plt.plot(temperatures, Rg_values, label='Radius of Gyration')
plt.xlabel('Temperature (K)')
plt.ylabel('Radius of Gyration (Å)')
plt.title(f'Radius of Gyration vs Temperature (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f})')
plt.legend()
plt.savefig(f'Radius of Gyration vs Temperature (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}).png')
plt.show()

# Plot End-to-End Distance vs Temperature
plt.figure()
plt.plot(temperatures, Ree_values, label='End-to-End Distance')
plt.xlabel('Temperature (K)')
plt.ylabel('End-to-End Distance (Å)')
plt.title(f'End-to-End Distance vs Temperature (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f})')
plt.legend()
plt.savefig(f'End-to-End Distance vs Temperature (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}).png')
plt.show()

# Plot Potential Energy vs Temperature
plt.figure()
plt.plot(temperatures, potential_energies, label='Potential Energy')
plt.xlabel('Temperature (K)')
plt.ylabel('Potential Energy')
plt.title(f'Potential Energy vs Temperature (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f})')
plt.legend()
plt.savefig(f'Potential Energy vs Temperature (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}).png')
plt.show()

# %%
# Simulation: k = 5, epsilon repulsive = 5
# Simulation parameters
dt = 0.01  # Time step
total_steps = 10000  # Number of steps
box_size = 100.0  # Size of the cubic box
mass = 1.0  # Particle mass
r0 = 1.0  # Equilibrium bond length
rescale_interval = 100  # Steps between velocity rescaling
n_particles = 20  # Number of particles
epsilon_attractive = 0.5  # Depth of attractive LJ potential
sigma = 1.0  # LJ potential parameter
temperatures = np.linspace(0.1, 1.0, 10)

# Variable Parameters: identify values of K and epsilon repulsive that prevent folding at low temperatures
k = 5  # Spring constant
epsilon_repulsive = 5  # Depth of repulsive LJ potential

# Arrays to store properties
Rg_values = []
Ree_values = []
potential_energies = []

for T in temperatures:
    # Set target temperature
    target_temperature = T
    # Re-initialize positions and velocities
    positions = initialize_chain(n_particles, box_size, r0)
    velocities = initialize_velocities(n_particles, T, mass)
    
    # Run simulation
    positions_history = []
    potential_energy_sum = 0
    for step in range(total_steps):
        forces_harmonic = compute_harmonic_forces(positions, k, r0, box_size)
        forces_repulsive = compute_lennard_jones_forces(positions, epsilon_repulsive, sigma, box_size, 'repulsive')
        forces_attractive = compute_lennard_jones_forces(positions, epsilon_attractive, sigma, box_size, 'attractive')
        total_forces = forces_harmonic + forces_repulsive + forces_attractive

        positions, velocities, total_forces = velocity_verlet(positions, velocities, total_forces, dt, mass)

        if step % rescale_interval == 0:
            velocities = rescale_velocities(velocities, T, mass)
            
            positions_history.append(positions.copy())

        # Accumulate potential energy 
        potential_energy_sum += compute_potential_energy(positions, k, r0, epsilon_repulsive, epsilon_attractive, sigma, box_size)
   
    # Compute properties
    Rg = calculate_radius_of_gyration(positions)
    Ree = calculate_end_to_end_distance(positions)
    Rg_values.append(Rg)
    Ree_values.append(Ree)
    potential_energies.append(np.mean(potential_energy_sum))
    
    # Plot the final configurations of the polymer chain
    final_positions = positions_history[-1]
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(final_positions[:, 0], final_positions[:, 1], final_positions[:, 2], '-o', color ='grey', markerfacecolor ='red', markeredgecolor ='red', label='Polymer Chain')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Final Configuration of Polymer Chain (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}, T = {T:.2f} K)')
    ax.legend()
    plt.savefig(f'Final Configuration of Polymer Chain (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}, T = {T:.2f} K).png')
    plt.show()
    
    # Create an animation to show the evolution of polymer chain configurations
    def update(frame):
        ax.clear()
        positions = positions_history[frame]
    
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], '-o',color ='grey', markerfacecolor ='red', markeredgecolor ='red', label=f'Step {frame * rescale_interval}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Evolution of Polymer Chain (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}, T = {T:.2f} K)')
        ax.legend()
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    polymer_chain_conformation_animation = animation.FuncAnimation(fig, update, frames=len(positions_history), interval=100)
    polymer_chain_conformation_animation.save(f'Evolution of Polymer Chain Configuration (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}, T = {T:.2f} K).gif', fps=10, writer = 'pillow')
    plt.show()

# Plot Radius of Gyration vs Temperature
plt.figure()
plt.plot(temperatures, Rg_values, label='Radius of Gyration')
plt.xlabel('Temperature (K)')
plt.ylabel('Radius of Gyration (Å)')
plt.title(f'Radius of Gyration vs Temperature (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f})')
plt.legend()
plt.savefig(f'Radius of Gyration vs Temperature (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}).png')
plt.show()

# Plot End-to-End Distance vs Temperature
plt.figure()
plt.plot(temperatures, Ree_values, label='End-to-End Distance')
plt.xlabel('Temperature (K)')
plt.ylabel('End-to-End Distance (Å)')
plt.title(f'End-to-End Distance vs Temperature (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f})')
plt.legend()
plt.savefig(f'End-to-End Distance vs Temperature (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}).png')
plt.show()

# Plot Potential Energy vs Temperature
plt.figure()
plt.plot(temperatures, potential_energies, label='Potential Energy')
plt.xlabel('Temperature (K)')
plt.ylabel('Potential Energy')
plt.title(f'Potential Energy vs Temperature (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f})')
plt.legend()
plt.savefig(f'Potential Energy vs Temperature (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}).png')
plt.show()

# %%
# Simulation: k = 0, epsilon repulsive = 0
# Simulation parameters
dt = 0.01  # Time step
total_steps = 10000  # Number of steps
box_size = 100.0  # Size of the cubic box
mass = 1.0  # Particle mass
r0 = 1.0  # Equilibrium bond length
rescale_interval = 100  # Steps between velocity rescaling
n_particles = 20  # Number of particles
epsilon_attractive = 0.5  # Depth of attractive LJ potential
sigma = 1.0  # LJ potential parameter
temperatures = np.linspace(0.1, 1.0, 10)

# Variable Parameters: identify values of K and epsilon repulsive that prevent folding at low temperatures
k = 0  # Spring constant
epsilon_repulsive = 0  # Depth of repulsive LJ potential

# Arrays to store properties
Rg_values = []
Ree_values = []
potential_energies = []

for T in temperatures:
    # Set target temperature
    target_temperature = T
    # Re-initialize positions and velocities
    positions = initialize_chain(n_particles, box_size, r0)
    velocities = initialize_velocities(n_particles, T, mass)
    
    # Run simulation
    positions_history = []
    potential_energy_sum = 0
    for step in range(total_steps):
        forces_harmonic = compute_harmonic_forces(positions, k, r0, box_size)
        forces_repulsive = compute_lennard_jones_forces(positions, epsilon_repulsive, sigma, box_size, 'repulsive')
        forces_attractive = compute_lennard_jones_forces(positions, epsilon_attractive, sigma, box_size, 'attractive')
        total_forces = forces_harmonic + forces_repulsive + forces_attractive

        positions, velocities, total_forces = velocity_verlet(positions, velocities, total_forces, dt, mass)

        if step % rescale_interval == 0:
            velocities = rescale_velocities(velocities, T, mass)
            
            positions_history.append(positions.copy())

        # Accumulate potential energy 
        potential_energy_sum += compute_potential_energy(positions, k, r0, epsilon_repulsive, epsilon_attractive, sigma, box_size)
   
    # Compute properties
    Rg = calculate_radius_of_gyration(positions)
    Ree = calculate_end_to_end_distance(positions)
    Rg_values.append(Rg)
    Ree_values.append(Ree)
    potential_energies.append(np.mean(potential_energy_sum))
    
    # Plot the final configurations of the polymer chain
    final_positions = positions_history[-1]
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(final_positions[:, 0], final_positions[:, 1], final_positions[:, 2], '-o', color ='grey', markerfacecolor ='red', markeredgecolor ='red', label='Polymer Chain')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Final Configuration of Polymer Chain (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}, T = {T:.2f} K)')
    ax.legend()
    plt.savefig(f'Final Configuration of Polymer Chain (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}, T = {T:.2f} K).png')
    plt.show()
    
    # Create an animation to show the evolution of polymer chain configurations
    def update(frame):
        ax.clear()
        positions = positions_history[frame]
    
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], '-o',color ='grey', markerfacecolor ='red', markeredgecolor ='red', label=f'Step {frame * rescale_interval}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Evolution of Polymer Chain (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}, T = {T:.2f} K)')
        ax.legend()
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    polymer_chain_conformation_animation = animation.FuncAnimation(fig, update, frames=len(positions_history), interval=100)
    polymer_chain_conformation_animation.save(f'Evolution of Polymer Chain Configuration (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}, T = {T:.2f} K).gif', fps=10, writer = 'pillow')
    plt.show()

# Plot Radius of Gyration vs Temperature
plt.figure()
plt.plot(temperatures, Rg_values, label='Radius of Gyration')
plt.xlabel('Temperature (K)')
plt.ylabel('Radius of Gyration (Å)')
plt.title(f'Radius of Gyration vs Temperature (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f})')
plt.legend()
plt.savefig(f'Radius of Gyration vs Temperature (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}).png')
plt.show()

# Plot End-to-End Distance vs Temperature
plt.figure()
plt.plot(temperatures, Ree_values, label='End-to-End Distance')
plt.xlabel('Temperature (K)')
plt.ylabel('End-to-End Distance (Å)')
plt.title(f'End-to-End Distance vs Temperature (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f})')
plt.legend()
plt.savefig(f'End-to-End Distance vs Temperature (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}).png')
plt.show()

# Plot Potential Energy vs Temperature
plt.figure()
plt.plot(temperatures, potential_energies, label='Potential Energy')
plt.xlabel('Temperature (K)')
plt.ylabel('Potential Energy')
plt.title(f'Potential Energy vs Temperature (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f})')
plt.legend()
plt.savefig(f'Potential Energy vs Temperature (k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}).png')
plt.show()

# %%
# Simulation: k = 1, epsilon repulsive = 1, number of particles = 5
# Simulation parameters
dt = 0.01  # Time step
total_steps = 10000  # Number of steps
box_size = 100.0  # Size of the cubic box
mass = 1.0  # Particle mass
r0 = 1.0  # Equilibrium bond length
rescale_interval = 100  # Steps between velocity rescaling
n_particles = 5  # Number of particles
epsilon_attractive = 0.5  # Depth of attractive LJ potential
sigma = 1.0  # LJ potential parameter
temperatures = np.linspace(0.1, 1.0, 10)

# Variable Parameters: identify values of K and epsilon repulsive that prevent folding at low temperatures
k = 1  # Spring constant
epsilon_repulsive = 1  # Depth of repulsive LJ potential

# Arrays to store properties
Rg_values = []
Ree_values = []
potential_energies = []

for T in temperatures:
    # Set target temperature
    target_temperature = T
    # Re-initialize positions and velocities
    positions = initialize_chain(n_particles, box_size, r0)
    velocities = initialize_velocities(n_particles, T, mass)
    
    # Run simulation
    positions_history = []
    potential_energy_sum = 0
    for step in range(total_steps):
        forces_harmonic = compute_harmonic_forces(positions, k, r0, box_size)
        forces_repulsive = compute_lennard_jones_forces(positions, epsilon_repulsive, sigma, box_size, 'repulsive')
        forces_attractive = compute_lennard_jones_forces(positions, epsilon_attractive, sigma, box_size, 'attractive')
        total_forces = forces_harmonic + forces_repulsive + forces_attractive

        positions, velocities, total_forces = velocity_verlet(positions, velocities, total_forces, dt, mass)

        if step % rescale_interval == 0:
            velocities = rescale_velocities(velocities, T, mass)
            
            positions_history.append(positions.copy())

        # Accumulate potential energy 
        potential_energy_sum += compute_potential_energy(positions, k, r0, epsilon_repulsive, epsilon_attractive, sigma, box_size)
   
    # Compute properties
    Rg = calculate_radius_of_gyration(positions)
    Ree = calculate_end_to_end_distance(positions)
    Rg_values.append(Rg)
    Ree_values.append(Ree)
    potential_energies.append(np.mean(potential_energy_sum))
    
    # Plot the final configurations of the polymer chain
    final_positions = positions_history[-1]
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(final_positions[:, 0], final_positions[:, 1], final_positions[:, 2], '-o', color ='grey', markerfacecolor ='red', markeredgecolor ='red', label='Polymer Chain')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Final Configuration of Polymer Chain (polymer chain length = {n_particles}, k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}, T = {T:.2f} K)')
    ax.legend()
    plt.savefig(f'Final Configuration of Polymer Chain (polymer chain length = {n_particles}, k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}, T = {T:.2f} K).png')
    plt.show()
    
    # Create an animation to show the evolution of polymer chain configurations
    def update(frame):
        ax.clear()
        positions = positions_history[frame]
    
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], '-o',color ='grey', markerfacecolor ='red', markeredgecolor ='red', label=f'Step {frame * rescale_interval}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Evolution of Polymer Chain (polymer chain length = {n_particles}, k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}, T = {T:.2f} K)')
        ax.legend()
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    polymer_chain_conformation_animation = animation.FuncAnimation(fig, update, frames=len(positions_history), interval=100)
    polymer_chain_conformation_animation.save(f'Evolution of Polymer Chain Configuration (polymer chain length = {n_particles}, k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}, T = {T:.2f} K).gif', fps=10, writer = 'pillow')
    plt.show()

# Plot Radius of Gyration vs Temperature
plt.figure()
plt.plot(temperatures, Rg_values, label='Radius of Gyration')
plt.xlabel('Temperature (K)')
plt.ylabel('Radius of Gyration (Å)')
plt.title(f'Radius of Gyration vs Temperature (polymer chain length = {n_particles}, k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f})')
plt.legend()
plt.savefig(f'Radius of Gyration vs Temperature (polymer chain length = {n_particles}, k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}).png')
plt.show()

# Plot End-to-End Distance vs Temperature
plt.figure()
plt.plot(temperatures, Ree_values, label='End-to-End Distance')
plt.xlabel('Temperature (K)')
plt.ylabel('End-to-End Distance (Å)')
plt.title(f'End-to-End Distance vs Temperature polymer chain length = {n_particles}, k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f})')
plt.legend()
plt.savefig(f'End-to-End Distance vs Temperature (polymer chain length = {n_particles}, k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}).png')
plt.show()

# Plot Potential Energy vs Temperature
plt.figure()
plt.plot(temperatures, potential_energies, label='Potential Energy')
plt.xlabel('Temperature (K)')
plt.ylabel('Potential Energy')
plt.title(f'Potential Energy vs Temperature (polymer chain length = {n_particles}, k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f})')
plt.legend()
plt.savefig(f'Potential Energy vs Temperature (polymer chain length = {n_particles}, k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}).png')
plt.show()

# %%
# Simulation: k = 1, epsilon repulsive = 1, number of particles = 10
# Simulation parameters
dt = 0.01  # Time step
total_steps = 10000  # Number of steps
box_size = 100.0  # Size of the cubic box
mass = 1.0  # Particle mass
r0 = 1.0  # Equilibrium bond length
rescale_interval = 100  # Steps between velocity rescaling
n_particles = 10  # Number of particles
epsilon_attractive = 0.5  # Depth of attractive LJ potential
sigma = 1.0  # LJ potential parameter
temperatures = np.linspace(0.1, 1.0, 10)

# Variable Parameters: identify values of K and epsilon repulsive that prevent folding at low temperatures
k = 1  # Spring constant
epsilon_repulsive = 1  # Depth of repulsive LJ potential

# Arrays to store properties
Rg_values = []
Ree_values = []
potential_energies = []

for T in temperatures:
    # Set target temperature
    target_temperature = T
    # Re-initialize positions and velocities
    positions = initialize_chain(n_particles, box_size, r0)
    velocities = initialize_velocities(n_particles, T, mass)
    
    # Run simulation
    positions_history = []
    potential_energy_sum = 0
    for step in range(total_steps):
        forces_harmonic = compute_harmonic_forces(positions, k, r0, box_size)
        forces_repulsive = compute_lennard_jones_forces(positions, epsilon_repulsive, sigma, box_size, 'repulsive')
        forces_attractive = compute_lennard_jones_forces(positions, epsilon_attractive, sigma, box_size, 'attractive')
        total_forces = forces_harmonic + forces_repulsive + forces_attractive

        positions, velocities, total_forces = velocity_verlet(positions, velocities, total_forces, dt, mass)

        if step % rescale_interval == 0:
            velocities = rescale_velocities(velocities, T, mass)
            
            positions_history.append(positions.copy())

        # Accumulate potential energy 
        potential_energy_sum += compute_potential_energy(positions, k, r0, epsilon_repulsive, epsilon_attractive, sigma, box_size)
   
    # Compute properties
    Rg = calculate_radius_of_gyration(positions)
    Ree = calculate_end_to_end_distance(positions)
    Rg_values.append(Rg)
    Ree_values.append(Ree)
    potential_energies.append(np.mean(potential_energy_sum))
    
    # Plot the final configurations of the polymer chain
    final_positions = positions_history[-1]
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(final_positions[:, 0], final_positions[:, 1], final_positions[:, 2], '-o', color ='grey', markerfacecolor ='red', markeredgecolor ='red', label='Polymer Chain')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Final Configuration of Polymer Chain (polymer chain length = {n_particles}, k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}, T = {T:.2f} K)')
    ax.legend()
    plt.savefig(f'Final Configuration of Polymer Chain (polymer chain length = {n_particles}, k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}, T = {T:.2f} K).png')
    plt.show()
    
    # Create an animation to show the evolution of polymer chain configurations
    def update(frame):
        ax.clear()
        positions = positions_history[frame]
    
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], '-o',color ='grey', markerfacecolor ='red', markeredgecolor ='red', label=f'Step {frame * rescale_interval}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Evolution of Polymer Chain (polymer chain length = {n_particles}, k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}, T = {T:.2f} K)')
        ax.legend()
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    polymer_chain_conformation_animation = animation.FuncAnimation(fig, update, frames=len(positions_history), interval=100)
    polymer_chain_conformation_animation.save(f'Evolution of Polymer Chain Configuration (polymer chain length = {n_particles}, k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}, T = {T:.2f} K).gif', fps=10, writer = 'pillow')
    plt.show()

# Plot Radius of Gyration vs Temperature
plt.figure()
plt.plot(temperatures, Rg_values, label='Radius of Gyration')
plt.xlabel('Temperature (K)')
plt.ylabel('Radius of Gyration (Å)')
plt.title(f'Radius of Gyration vs Temperature (polymer chain length = {n_particles}, k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f})')
plt.legend()
plt.savefig(f'Radius of Gyration vs Temperature (polymer chain length = {n_particles}, k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}).png')
plt.show()

# Plot End-to-End Distance vs Temperature
plt.figure()
plt.plot(temperatures, Ree_values, label='End-to-End Distance')
plt.xlabel('Temperature (K)')
plt.ylabel('End-to-End Distance (Å)')
plt.title(f'End-to-End Distance vs Temperature polymer chain length = {n_particles}, k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f})')
plt.legend()
plt.savefig(f'End-to-End Distance vs Temperature (polymer chain length = {n_particles}, k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}).png')
plt.show()

# Plot Potential Energy vs Temperature
plt.figure()
plt.plot(temperatures, potential_energies, label='Potential Energy')
plt.xlabel('Temperature (K)')
plt.ylabel('Potential Energy')
plt.title(f'Potential Energy vs Temperature (polymer chain length = {n_particles}, k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f})')
plt.legend()
plt.savefig(f'Potential Energy vs Temperature (polymer chain length = {n_particles}, k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}).png')
plt.show()

# %%
# Simulation: k = 1, epsilon repulsive = 1, number of particles = 40
# Simulation parameters
dt = 0.01  # Time step
total_steps = 10000  # Number of steps
box_size = 100.0  # Size of the cubic box
mass = 1.0  # Particle mass
r0 = 1.0  # Equilibrium bond length
rescale_interval = 100  # Steps between velocity rescaling
n_particles = 40  # Number of particles
epsilon_attractive = 0.5  # Depth of attractive LJ potential
sigma = 1.0  # LJ potential parameter
temperatures = np.linspace(0.1, 1.0, 10)

# Variable Parameters: identify values of K and epsilon repulsive that prevent folding at low temperatures
k = 1  # Spring constant
epsilon_repulsive = 1  # Depth of repulsive LJ potential

# Arrays to store properties
Rg_values = []
Ree_values = []
potential_energies = []

for T in temperatures:
    # Set target temperature
    target_temperature = T
    # Re-initialize positions and velocities
    positions = initialize_chain(n_particles, box_size, r0)
    velocities = initialize_velocities(n_particles, T, mass)
    
    # Run simulation
    positions_history = []
    potential_energy_sum = 0
    for step in range(total_steps):
        forces_harmonic = compute_harmonic_forces(positions, k, r0, box_size)
        forces_repulsive = compute_lennard_jones_forces(positions, epsilon_repulsive, sigma, box_size, 'repulsive')
        forces_attractive = compute_lennard_jones_forces(positions, epsilon_attractive, sigma, box_size, 'attractive')
        total_forces = forces_harmonic + forces_repulsive + forces_attractive

        positions, velocities, total_forces = velocity_verlet(positions, velocities, total_forces, dt, mass)

        if step % rescale_interval == 0:
            velocities = rescale_velocities(velocities, T, mass)
            
            positions_history.append(positions.copy())

        # Accumulate potential energy 
        potential_energy_sum += compute_potential_energy(positions, k, r0, epsilon_repulsive, epsilon_attractive, sigma, box_size)
   
    # Compute properties
    Rg = calculate_radius_of_gyration(positions)
    Ree = calculate_end_to_end_distance(positions)
    Rg_values.append(Rg)
    Ree_values.append(Ree)
    potential_energies.append(np.mean(potential_energy_sum))
    
    # Plot the final configurations of the polymer chain
    final_positions = positions_history[-1]
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(final_positions[:, 0], final_positions[:, 1], final_positions[:, 2], '-o', color ='grey', markerfacecolor ='red', markeredgecolor ='red', label='Polymer Chain')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Final Configuration of Polymer Chain (polymer chain length = {n_particles}, k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}, T = {T:.2f} K)')
    ax.legend()
    plt.savefig(f'Final Configuration of Polymer Chain (polymer chain length = {n_particles}, k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}, T = {T:.2f} K).png')
    plt.show()
    
    # Create an animation to show the evolution of polymer chain configurations
    def update(frame):
        ax.clear()
        positions = positions_history[frame]
    
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], '-o',color ='grey', markerfacecolor ='red', markeredgecolor ='red', label=f'Step {frame * rescale_interval}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Evolution of Polymer Chain (polymer chain length = {n_particles}, k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}, T = {T:.2f} K)')
        ax.legend()
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    polymer_chain_conformation_animation = animation.FuncAnimation(fig, update, frames=len(positions_history), interval=100)
    polymer_chain_conformation_animation.save(f'Evolution of Polymer Chain Configuration (polymer chain length = {n_particles}, k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}, T = {T:.2f} K).gif', fps=10, writer = 'pillow')
    plt.show()

# Plot Radius of Gyration vs Temperature
plt.figure()
plt.plot(temperatures, Rg_values, label='Radius of Gyration')
plt.xlabel('Temperature (K)')
plt.ylabel('Radius of Gyration (Å)')
plt.title(f'Radius of Gyration vs Temperature (polymer chain length = {n_particles}, k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f})')
plt.legend()
plt.savefig(f'Radius of Gyration vs Temperature (polymer chain length = {n_particles}, k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}).png')
plt.show()

# Plot End-to-End Distance vs Temperature
plt.figure()
plt.plot(temperatures, Ree_values, label='End-to-End Distance')
plt.xlabel('Temperature (K)')
plt.ylabel('End-to-End Distance (Å)')
plt.title(f'End-to-End Distance vs Temperature polymer chain length = {n_particles}, k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f})')
plt.legend()
plt.savefig(f'End-to-End Distance vs Temperature (polymer chain length = {n_particles}, k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}).png')
plt.show()

# Plot Potential Energy vs Temperature
plt.figure()
plt.plot(temperatures, potential_energies, label='Potential Energy')
plt.xlabel('Temperature (K)')
plt.ylabel('Potential Energy')
plt.title(f'Potential Energy vs Temperature (polymer chain length = {n_particles}, k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f})')
plt.legend()
plt.savefig(f'Potential Energy vs Temperature (polymer chain length = {n_particles}, k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}).png')
plt.show()

# %%
# Simulation: k = 1, epsilon repulsive = 1, number of particles = 100
# Simulation parameters
dt = 0.01  # Time step
total_steps = 10000  # Number of steps
box_size = 100.0  # Size of the cubic box
mass = 1.0  # Particle mass
r0 = 1.0  # Equilibrium bond length
rescale_interval = 100  # Steps between velocity rescaling
n_particles = 100  # Number of particles
epsilon_attractive = 0.5  # Depth of attractive LJ potential
sigma = 1.0  # LJ potential parameter
temperatures = np.linspace(0.1, 1.0, 10)

# Variable Parameters: identify values of K and epsilon repulsive that prevent folding at low temperatures
k = 1  # Spring constant
epsilon_repulsive = 1  # Depth of repulsive LJ potential

# Arrays to store properties
Rg_values = []
Ree_values = []
potential_energies = []

for T in temperatures:
    # Set target temperature
    target_temperature = T
    # Re-initialize positions and velocities
    positions = initialize_chain(n_particles, box_size, r0)
    velocities = initialize_velocities(n_particles, T, mass)
    
    # Run simulation
    positions_history = []
    potential_energy_sum = 0
    for step in range(total_steps):
        forces_harmonic = compute_harmonic_forces(positions, k, r0, box_size)
        forces_repulsive = compute_lennard_jones_forces(positions, epsilon_repulsive, sigma, box_size, 'repulsive')
        forces_attractive = compute_lennard_jones_forces(positions, epsilon_attractive, sigma, box_size, 'attractive')
        total_forces = forces_harmonic + forces_repulsive + forces_attractive

        positions, velocities, total_forces = velocity_verlet(positions, velocities, total_forces, dt, mass)

        if step % rescale_interval == 0:
            velocities = rescale_velocities(velocities, T, mass)
            
            positions_history.append(positions.copy())

        # Accumulate potential energy 
        potential_energy_sum += compute_potential_energy(positions, k, r0, epsilon_repulsive, epsilon_attractive, sigma, box_size)
   
    # Compute properties
    Rg = calculate_radius_of_gyration(positions)
    Ree = calculate_end_to_end_distance(positions)
    Rg_values.append(Rg)
    Ree_values.append(Ree)
    potential_energies.append(np.mean(potential_energy_sum))
    
    # Plot the final configurations of the polymer chain
    final_positions = positions_history[-1]
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(final_positions[:, 0], final_positions[:, 1], final_positions[:, 2], '-o', color ='grey', markerfacecolor ='red', markeredgecolor ='red', label='Polymer Chain')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Final Configuration of Polymer Chain (polymer chain length = {n_particles}, k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}, T = {T:.2f} K)')
    ax.legend()
    plt.savefig(f'Final Configuration of Polymer Chain (polymer chain length = {n_particles}, k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}, T = {T:.2f} K).png')
    plt.show()
    
    # Create an animation to show the evolution of polymer chain configurations
    def update(frame):
        ax.clear()
        positions = positions_history[frame]
    
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], '-o',color ='grey', markerfacecolor ='red', markeredgecolor ='red', label=f'Step {frame * rescale_interval}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Evolution of Polymer Chain (polymer chain length = {n_particles}, k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}, T = {T:.2f} K)')
        ax.legend()
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    polymer_chain_conformation_animation = animation.FuncAnimation(fig, update, frames=len(positions_history), interval=100)
    polymer_chain_conformation_animation.save(f'Evolution of Polymer Chain Configuration (polymer chain length = {n_particles}, k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}, T = {T:.2f} K).gif', fps=10, writer = 'pillow')
    plt.show()

# Plot Radius of Gyration vs Temperature
plt.figure()
plt.plot(temperatures, Rg_values, label='Radius of Gyration')
plt.xlabel('Temperature (K)')
plt.ylabel('Radius of Gyration (Å)')
plt.title(f'Radius of Gyration vs Temperature (polymer chain length = {n_particles}, k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f})')
plt.legend()
plt.savefig(f'Radius of Gyration vs Temperature (polymer chain length = {n_particles}, k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}).png')
plt.show()

# Plot End-to-End Distance vs Temperature
plt.figure()
plt.plot(temperatures, Ree_values, label='End-to-End Distance')
plt.xlabel('Temperature (K)')
plt.ylabel('End-to-End Distance (Å)')
plt.title(f'End-to-End Distance vs Temperature polymer chain length = {n_particles}, k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f})')
plt.legend()
plt.savefig(f'End-to-End Distance vs Temperature (polymer chain length = {n_particles}, k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}).png')
plt.show()

# Plot Potential Energy vs Temperature
plt.figure()
plt.plot(temperatures, potential_energies, label='Potential Energy')
plt.xlabel('Temperature (K)')
plt.ylabel('Potential Energy')
plt.title(f'Potential Energy vs Temperature (polymer chain length = {n_particles}, k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f})')
plt.legend()
plt.savefig(f'Potential Energy vs Temperature (polymer chain length = {n_particles}, k = {k:.2f}, epsilon repulsive = {epsilon_repulsive:.2f}).png')
plt.show()

# %%



