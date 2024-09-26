# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
# Assign values to the function's variables
omega = 1
D = 10
beta = np.sqrt(1 / (2 * D))
L = 40
x = np.linspace(-L/2, L/2, 2000)
dx = L / 2000
x0 = 0

# %%
# Define harmonic potential and anharmonic potential
def harmonic_potential(x, omega):
    return 0.5 * omega**2 * x**2

def anharmonic_potential(x, D, beta, x0):
    return D * (1 - np.exp(-beta * (x - x0)))**2

# %%
# Construct the potential matrix for the harmonic and anharmonic potentials by np.diag
harmonic_matrix = np.diag(harmonic_potential(x, omega))
anharmonic_matrix = np.diag(anharmonic_potential(x, D, beta, x0))


# %%
# Construct the Laplacian matrix
Laplacian =((-2 * np.eye(2000) + np.eye(2000, k=1)) + np.eye(2000, k=-1))/ dx**2

# %%
# Construct the Hamiltonian Matrix
def Hamiltonian_Matrix(V, Laplacian):
    return -0.5 * Laplacian + V

Hamilton_harmonic = Hamiltonian_Matrix(harmonic_matrix, Laplacian)
Hamilton_anharmonic = Hamiltonian_Matrix(anharmonic_matrix, Laplacian)


# %%
# Using np.linalg.eig to compute the eigenvalues and eigenfunctions of the Hamiltonian for the harmonic and anharmonic potentials
Eigenvalues_harmonic, Eigenfunctions_harmonic = np.linalg.eig(Hamilton_harmonic)
Eigenvalues_anharmonic, Eigenfunctions_anharmonic = np.linalg.eig(Hamilton_anharmonic)

print("The eigenfunction of the Hamiltonian for the harmonic potential:")
print(Eigenfunctions_harmonic)

print("The eigenfunction of the Hamiltonian for the anharmonic potential:")
print(Eigenfunctions_anharmonic)

# %%
# Sort the eigenvalues in increasing order and extract the first ten energy levels

sort_harmonic = np.argsort(Eigenvalues_harmonic)
Eigenvalues_harmonic_sorted = Eigenvalues_harmonic[sort_harmonic]
Eigenfunctions_harmonic_sorted = Eigenfunctions_harmonic[:, sort_harmonic] # Sort the Eigenfunctions


sort_anharmonic = np.argsort(Eigenvalues_anharmonic)
Eigenvalues_anharmonic_sorted = Eigenvalues_anharmonic[sort_anharmonic]
Eigenfunctions_anharmonic_sorted = Eigenfunctions_anharmonic[:, sort_anharmonic]

print("The first ten eigenvalues of the Hamiltonian for the harmonic potential:")
print(Eigenvalues_harmonic_sorted[:10])

print("The first ten eigenvalues of the Hamiltonian for the anharmonic potential:")
print(Eigenvalues_anharmonic_sorted[:10])
      

# %%
# Plot the first ten wavefunctions for harmonic potential

plt.figure(figsize=(8, 6))
for i in range(10):
    plt.plot(x, (Eigenfunctions_harmonic_sorted[:, i] + Eigenvalues_harmonic_sorted[i]), label=f"n={i+1}, E={Eigenvalues_harmonic_sorted[i]:.4f}") # Show the wavefunctions at their corresponding energy level.
    plt.axhline(Eigenvalues_harmonic_sorted[i], color = "black", linestyle = "dotted" )
    
plt.plot(x, harmonic_potential(x, omega), linestyle = "--", color = "grey", label = "Harmonic Potential")
plt.ylim(0, 10)
plt.xlabel("distance (a.u.)")
plt.ylabel("Energy (a.u.)")
plt.title('The First Ten Harmonic Oscillator Wavefunctions')
plt.legend()
plt.show()

# %%
# Plot the first ten wavefunctions for anharmonic potential
plt.figure(figsize=(8, 6))
for i in range(10):
    plt.plot(x, Eigenfunctions_anharmonic_sorted[:, i] + Eigenvalues_anharmonic_sorted[i], label=f"n={i+1}, E={Eigenvalues_anharmonic_sorted[i]:.4f}")
    plt.axhline(Eigenvalues_anharmonic_sorted[i], color = "black", linestyle = "dotted" )

plt.plot(x, anharmonic_potential(x, D, beta, x0), linestyle = "--", color = "grey", label = "Anharmonic Potential")
plt.ylim(0, 8) # Set a limit on y axis
plt.xlabel("distance (a.u.)")
plt.ylabel("Energy (a.u)")
plt.title('The First Ten Anharmonic Oscillator Wavefunctions')
plt.legend()
plt.show()

# %%



