#%%
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
import time
def is_overlapping(new_position, existing_positions, radius_of_particales, box_width, box_height):
    """
    Check if the particle overlaps with existing particles.
    """
    if len(existing_positions) == 0:
        return False  # No particles yet, no overlap
    box_size = np.array([box_width, box_height])
    raw_distances = existing_positions - new_position
    raw_distances = raw_distances + box_size / 2 # shift the distance
    raw_distances = raw_distances % box_size - box_size / 2 # find shortest distance
    distances = np.linalg.norm(raw_distances, axis=1) # distandces with periodic boundaries
    return np.any(distances < 2 * radius_of_particales)


def initialize_particals_on_surface(number_of_particles, radius_of_particales, box_width, box_height, max_attempts=50000):
    """
    Initialize particles on the surface.
    """
    particels_position_list = []
    attempts = 0
    packing_fraction = number_of_particles * np.pi * radius_of_particales**2 / (box_width * box_height)
   
    while len(particels_position_list) < number_of_particles and attempts < max_attempts:
        
        # Randomly generate a position
        x = np.random.uniform(0, box_width)
        y = np.random.uniform(0, box_height)
        new_position = np.array([x, y])
        # Check if the particle overlaps with existing particles
        if not is_overlapping(new_position, np.array(particels_position_list), radius_of_particales, box_width, box_height):
            particels_position_list.append(new_position)



        attempts += 1
    
    packing_fraction = len(particels_position_list) * np.pi * radius_of_particales**2 / (box_width * box_height)
    particals = np.array(particels_position_list)
    return particals, packing_fraction

def initialize_velocities_of_particals(number_of_particles, T = 1.0, kb = 1., m=1.0):
    """
    Initialize velocities of particles.
    """
    sigma = np.sqrt(kb * T / m)  # Standard deviation for velocity distribution
    xy_velocity = np.random.normal(0, sigma, size=(number_of_particles, 2))
    velocity_cm = np.mean(xy_velocity, axis=0)
    xy_velocity -= velocity_cm  # Center of mass velocity
  

    return xy_velocity


# Simplified Wang-Frenkel potential
def wang_frenkel(delta, rc,  epsilon=1.0, sigma=1.0):
    #print("delta", delta)
    r = np.linalg.norm(delta, axis=-1)
    #print("r", r)
    mask = (r < rc) & (r > 0)  # r > 0 to avoid division by zero
    wf = np.zeros_like(r)
    wf[mask] = epsilon * ((sigma / r[mask])**2 - 1) * (((rc / r[mask])**2 - 1)) ** 2
    return 0.5 * np.sum(wf)  # 0.5 to avoid double counting


def wang_frenkel_force_vector(delta, rc, epsilon=1.0, sigma=1.0):
    r = np.linalg.norm(delta, axis=-1)  # Shape: (N, N)
    forces = np.zeros_like(delta)
    
    mask = (r < rc) & (r > 0)  # Avoid division by zero
    
    r_valid = r[mask]
    
    term1 = (sigma / r_valid)**2 - 1
    term2 = (rc / r_valid)**2 - 1
    
    dU_dr = (-2 * epsilon * sigma**2 / r_valid**3) * (term2**2) + \
            (4 * epsilon * term1 * term2 * rc**2 / r_valid**3)

    # Calculate unit vectors safely
    with np.errstate(divide='ignore', invalid='ignore'):
        unit_vectors = np.nan_to_num(delta[mask] / r_valid[:, None])  # Shape (M, 2)

    # Calculate final force vectors
    forces[mask] = -dU_dr[:, None] * unit_vectors  # Multiply scalar force by unit vector

    total_forces = np.sum(forces, axis=1)  # Shape: (N, 2)

    return total_forces



def create_lists_of_particals_in_subcells(particals_position, box_width, box_height, rc):
    """
    Create lists of particles in subcells.
    """
    num_subcells_x = int(box_width // rc)
    num_subcells_y = int(box_height // rc)

    cell_indices_x = ((particals_position[:, 0] // rc) % num_subcells_x).astype(int)
    cell_indices_y = ((particals_position[:, 1] // rc) % num_subcells_y).astype(int)
    cell_indices = np.vstack((cell_indices_x, cell_indices_y)).T
    #print(f"Cell indices:\n{cell_indices}")
    sorted_cell_indices = np.sort(cell_indices, axis=0)
    #print(f"Sorted Cell indices:{cell_indices[sorted_cell_indices]}")
    cell_ids = cell_indices_y * num_subcells_x + cell_indices_x  # 1D Cell-IDs
    #print(f"Cell IDs: {cell_ids}")
    sorted_indices = np.argsort(cell_ids)
    sorted_cell_ids = cell_ids[sorted_indices]
    #print(f"Sorted Cell IDs: {sorted_cell_ids}")

    # Array der 9 möglichen Nachbar-Offsets
    neighbor_offsets = np.array([
        [-1, -1], [0, -1], [1, -1],
        [-1,  0], [0,  0], [1,  0],
        [-1,  1], [0,  1], [1,  1]
    ])

    # Berechne alle Nachbarzellen mit PB
    target_cell = np.array([0, 0])  # Beispiel Zielzelles
    neighbors = (np.array(target_cell) + neighbor_offsets) % [num_subcells_x, num_subcells_y]

    #print(f"Neighbors: {neighbors}")
    #print("Cell indices:", cell_indices)
    #indices = np.where(cell_indices[:, None] == neighbors).any(axis=2)
    #print(f"Indices: {indices}")


    return cell_indices

def calculate_distance_between_particals(positions, box_w, box_h):
    """
    Calculate the distance between two particles with periodic boundaries.
    """
    box_size = np.array([box_w, box_h])
    delta = positions[:,None,:] - positions[None,:,:]
    delta -= np.round(delta / box_size) * box_size  # Apply periodic boundary conditions
    #particals_distancees = np.linalg.norm(delta, axis=-1)

    #print("Particals distances:", delta)
    return delta

def calculate_forces_and_potential_between_particals(positions, rc):
    """
    Calculate forces between particles based on the Wang-Frenkel potential.
    """
    partical_distances = calculate_distance_between_particals(positions, box_w, box_h)
    #print(f"Partical distances: {partical_distances.shape}")
    N = partical_distances.shape[0]
    i_lower, j_lower = np.tril_indices(N, k=-1)
    r_values = partical_distances[i_lower, j_lower]
    #print(f"R values: {r_values.shape}")
    # Calculate forces using the Wang-Frenkel potential
    wf_force_values = wang_frenkel_force_vector(partical_distances, rc=rc)
    wf_pot_energy_values = wang_frenkel(partical_distances, rc=rc)
    #print("wf_pot_energy_values", wf_pot_energy_values)
    return wf_force_values, wf_pot_energy_values
    # Calculate forces using the Wang-Frenkel potential

def evaluate_computational_cost():
    N_values = np.arange(100, 4100, 500)  # Vary N from 100 to 1000
    times = []

    for N in N_values:
        print(f"Evaluating for N = {N}")
        positions = np.random.rand(N, 2)  # 2D positions
        start_time = time.perf_counter()
        distances = calculate_distance_between_particals(positions, box_w, box_h)
        end_time = time.perf_counter()
        times.append(end_time - start_time)

    times = np.array(times) 
    # Polynomial fit of degree 2 (quadratic)
    coeffs = np.polyfit(N_values, times, 2)
    fit_func = np.poly1d(coeffs)  # Create a callable polynomial function

    # Plot results
    plt.figure(figsize=(8, 5))
    plt.plot(N_values, times, 'o-', label='Measured Time')
    plt.plot(N_values, fit_func(N_values), 'r--', label=f'Quadratic Fit: {coeffs[0]:.2e}·N² + {coeffs[1]:.2e}·N + {coeffs[2]:.2e}')
    plt.xlabel('Number of Particles (N)')
    plt.ylabel('Time (s)')
    plt.title('Computational Cost of Distance Calculation')
    plt.grid(True)
    plt.legend()
    plt.savefig("Cost_Evaluaion")
    #plt.show()

def velocity_verlet(positions, velocities, forces, dt, box_size, m=1.0, kb = 1.0):
    """One step of the velocity-Verlet algorithm."""

    # Half-step velocity update
    velocities_half = velocities.copy() + 0.5 * forces * dt / m

    # Position update
    positions += velocities_half * dt
    positions %= box_size  # Periodic boundary conditions

    # Force update
    forces_new, potential_energy = calculate_forces_and_potential_between_particals(positions, box_size)
    #print("forces_new", forces_new)
    # Full-step velocity update
    velocities = velocities_half + 0.5 * forces_new * dt / m
    #print("velocities", velocities)
    kinetic_energy = 0.5 * m * np.sum(np.linalg.norm(velocities, axis=1)**2)
    
    total_energy = kinetic_energy + potential_energy
    N = positions.shape[0]
    temperature = kinetic_energy / (N * kb)

    #print("kinetic_energy", kinetic_energy)
    #print("total_energy", total_energy)
    #print("temperature", temperature)
    return positions, velocities, forces_new, potential_energy, kinetic_energy, total_energy, temperature

    

# Simulation Parameters
num_particles = 5000
radius = 1.
# Simulation box dimensions
box_w, box_h = 10, 10

m = 1.0
T = 1.0  # Temperature in Kelvin
kb = 1.  # Boltzmann constant
sigma = 1.0 # length scale
epsilon = 1.0  # eneregy scale
delta_t = 0.00001 * np.sqrt(m * sigma ** 2 / epsilon) # time step
rc = 5. * sigma  # Cutoff radius for neighbor search
# OBERVATION: Packingfracion is when random between 0.5 and 0.6

particals_position, packing_fraction = initialize_particals_on_surface(num_particles, radius, box_w, box_h)
particals_velocity = initialize_velocities_of_particals(len(particals_position))
#print("particals_velocity", particals_velocity)
print(f"Packing fraction: {packing_fraction:.2f}")
#print(f" True numbre of particles: {len(particals_position)}")


# Evaluate computational cost
evaluate_computational_cost()

# Plotting
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(0, box_w)
ax.set_ylim(0, box_h)
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_title('Particles on Surface')
ax.set_aspect('equal', adjustable='box')
ax.grid(True)

# Plot velocities as arrows (quiver plot)
ax.quiver(
    particals_position[:, 0],  # X positions
    particals_position[:, 1],  # Y positions
    particals_velocity[:, 0],  # vx components
    particals_velocity[:, 1],  # vy components
    angles='xy', scale_units='xy', scale=1, color='red', width=0.005
)

# Draw Particles and Circles
for pos in particals_position:
    circle = plt.Circle(pos, radius, color='lightblue', fill=False, linestyle='--')
    ax.add_patch(circle)
    ax.plot(pos[0], pos[1], 'o', color='blue', markersize=5)
plt.tight_layout()
plt.savefig("initial_particals")
#plt.show()



num_steps = 100000


time_total = np.arange(num_steps) * delta_t

kin_energies = np.zeros(num_steps)
pot_energies = np.zeros(num_steps)
tot_energies = np.zeros(num_steps)
tempeeratures = np.zeros(num_steps)

positions = particals_position.copy()
velocities = particals_velocity.copy()

for i in range(num_steps):

    print("current step:", i)
    forces, potential = calculate_forces_and_potential_between_particals(positions, rc)
    #print("forces", forces)
    #print("positions", positions)
    #print("velocities", velocities)
    positions, velocities, forces, pot_energies[i], kin_energies[i], tot_energies[i], tempeeratures[i] = velocity_verlet(
        positions, velocities, forces, delta_t, box_w, m
    )
    #print("pot_energies", pot_energies.shape)



plt.figure(figsize=(10, 6))
plt.plot(time_total, kin_energies, label="Kinetic Energy", linestyle='-', marker='o', markersize=3)
plt.plot(time_total, pot_energies, label="Potential Energy", linestyle='-', marker='x', markersize=3)
plt.plot(time_total, tot_energies, label="Total Energy", linestyle='-', marker='s', markersize=3)
plt.xlabel("Time [s]")
plt.ylabel("Energy [J]")
plt.title("Energy vs Time")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("Energy Balance")
#plt.show()


plt.figure(figsize=(8, 5))
plt.plot(time_total, tempeeratures, color='orange', linestyle='-', marker='d', markersize=3)
plt.xlabel("Time [s]")
plt.ylabel("Temperature [K or Reduced Units]")
plt.title("Temperature vs Time")
plt.grid(True)
plt.tight_layout()
plt.savefig("Temperature")
#plt.show()


# %%
