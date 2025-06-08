#%%
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
import time
from numba import njit, prange
from tqdm import tqdm
from matplotlib.cm import get_cmap

##-----------Functions------------------

def plot_diffusion_constant(r_s, v_s, dt, given_temperatures):
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=False)
    # Pick a color map (e.g., "tab10" or "viridis")
    cmap = get_cmap("tab10")
    D = {}
    axs = axs.flatten()
    for i, temp in enumerate(given_temperatures):
        color = cmap(i % 10)  # safe for up to 10 temps; adjust for more
        print(f"\nTemperature: {temp} K")
        r = r_s[temp]
        v = v_s[temp]
        num_time_steps = r.shape[0]
        times = np.arange(num_time_steps) * dt

        # Compute all diffusion constants and time-dependent correlations
        D[temp] = np.zeros(3)
        D[temp][0], msd = diffusion_constant_with_r(r, dt)
        #D[temp][1], vacf_time, clean_timesteps = diffusion_constant_with_v_(v, dt)
        D[temp][2], vacf_fft, freqs, power_spec, clean_timesteps = diffusion_constant_fft(v, dt=dt)

        equilibrium_times = np.arange(clean_timesteps) * dt

        # Print all diffusion constants
        print(f"MSD: {D[temp][0]:.4f}, VACF-time: {D[temp][1]:.4f}, VACF-FFT: {D[temp][2]:.4f}")

        # Plot MSD (Δr²(t))
        axs[0].plot(times, msd, color=color, label=f'T = {temp} K')

        # Plot VACFs
        #axs[1].plot(equilibrium_times, vacf_time, '--', color=color, label=f'VACF time, T={temp}K')
        axs[1].plot(equilibrium_times, vacf_fft, '-', color=color, label=f'VACF FFT, T={temp}K')

        # Power spectrum plot (positive frequencies only)
        positive_freqs = freqs[:num_time_steps//2]
        power_pos = power_spec[:num_time_steps//2]
        axs[2].plot(positive_freqs, power_pos, label=f'T = {temp} K')

        # Plot horizontal lines (Diffusion constants)
        axs[3].axhline(D[temp][0], linestyle='-', color=color, label=f'D_MSD, D={D[temp][0]:.4f}K')
        #axs[3].axhline(D[temp][1], linestyle='--', color=color, label=f'D_VACF time, T={temp}K')
        axs[3].axhline(D[temp][2], linestyle=':', color=color, label=f'D_VACF FFT, D={D[temp][2]:.4f}K')
    
    # Plot styling
    axs[0].set_ylabel(r'MSD  $\langle \Delta r^2(t) \rangle$')
    axs[0].set_title('Mean Squared Displacement')
    axs[0].legend()
    axs[0].grid()

    axs[1].set_xlabel('Time [s]')
    axs[1].set_ylabel(r'VACF $\langle \vec{v}(0) \cdot \vec{v}(t) \rangle$')
    axs[1].set_title('Velocity Autocorrelation Function')
    axs[1].legend()
    axs[1].grid()


    axs[2].set_xlabel('Frequency [Hz]')
    axs[2].set_ylabel(r'Power Spectrum $G(\omega)$')
    axs[2].set_title('Velocity Power Spectrum')
    axs[2].legend()
    axs[2].grid()

    axs[3].set_title("Diffusion Constants")
    axs[3].set_ylabel("D")
    axs[3].set_xlabel("Method (lines)")
    axs[3].legend()
    axs[3].grid()

    plt.tight_layout()
    plt.savefig(f"Diffusion_PF_{packing_fraction:.4f}_T_{given_temperatures}.png")
    plt.show()

    return D




#---Plot Functions-----
def plot_energies(time_total, kin_energies, pot_energies, tot_energies, given_temperatures, packing_fraction):
    
    plt.figure(figsize=(10, 6))
    for temp in given_temperatures:
        plt.plot(time_total, kin_energies[temp], label=f"Kinetic Energy T={temp}")
        plt.plot(time_total, pot_energies[temp], label=f"Potential Energy T={temp}")
        plt.plot(time_total, tot_energies[temp], label=f"Total Energy T={temp}")
    plt.xlabel("Time [s]")
    plt.ylabel("Energy [J]")
    plt.title(f"Energy vs Time - PF_{packing_fraction:.4f}_T_{given_temperatures}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"Energy_Balance_PF_{packing_fraction:.4f}_T_{given_temperatures}.png")
    #plt.show()

def plot_temperature(time_total, kinetic_temperature, given_temperatures, packing_fraction):
    plt.figure(figsize=(8, 5))
    for temp in given_temperatures:
        plt.plot(time_total, kinetic_temperature[temp], linestyle='-', marker='d', markersize=3, label=f"T={temp}")
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Temperature [K or Reduced Units]")
    plt.title(f"Temperature vs Time - PF_{packing_fraction:.4f}_T_{given_temperatures}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"Temperature_PF_{packing_fraction:.4f}_T_{given_temperatures}.png")
    #plt.show()

def plot_radial_distribution(r_array, radial_distributions, given_temperatures, packing_fraction):
    plt.figure(figsize=(8, 5))
    for temp in given_temperatures:
        time_average_g = np.mean(radial_distributions[temp], axis=0)
        #print("shape_time_average_radial_dis", time_average_g)
        plt.plot(r_array, time_average_g, linestyle='-', marker='d', markersize=3, label=f"T={temp}")
    plt.axhline(1.0, color="red")
    plt.legend()
    plt.xlabel("r []]")
    plt.ylabel("Radial Distribution")
    plt.title(f"Radial Distribution vs r - PF_{packing_fraction:.4f}_T_{given_temperatures}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"RDF_PF_{packing_fraction:.4f}_T_{given_temperatures}.png")
    #plt.show()


def plot_initial_condition(filename, particals_position, particals_velocity, packing_fraction, box_w, box_h, initial_T):
    print(f"Packing fraction: {packing_fraction:.4f}")
    print("True Number of particals", particals_position.shape[0])
    # OBERVATION: Packingfracion is when random between 0.5 and 0.6

    # Plotting
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, box_w)
    ax.set_ylim(0, box_h)
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_title(f'Particles on Surface with PF = {packing_fraction:.4f}, and T = {initial_T}')
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
    plt.savefig(f"{filename}_PF_{packing_fraction:.4f}_T_{initial_T}.png")
    #plt.show()

#-------

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

def  initialize_particals_on_surface(number_of_particles, radius_of_particales, box_width, box_height, max_attempts=50000):
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

def initialize_particals_ordered(number_of_particles, radius_of_particales, varaince_of_positij, box_width, box_height):
    """ Initialize ordered Particals with certain radius and count
    with ceretain variance in the order of the center
    """
    #np.arange(0, box_h, 2*)
    #return particals, packing_fraction


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
#@numba.njit
def wang_frenkel(r, rc,  epsilon=1.0, sigma=1.0):
    #print("delta", delta)
    #print("r", r)
    mask = (r < rc) & (r > 0)  # r > 0 to avoid division by zero
    wf = np.zeros_like(r)
    wf[mask] = epsilon * ((sigma / r[mask])**2 - 1) * (((rc / r[mask])**2 - 1)) ** 2
    return 0.5 * np.sum(wf)  # 0.5 to avoid double counting

#@numba.njit
def wang_frenkel_force_vector(r, delta, rc, epsilon=1.0, sigma=1.0):
    
    forces = np.zeros([r.shape[0], r.shape[1], 2])
    
    mask = (r < rc) & (r > 0)  # Avoid division by zero
    
    r_valid = r[mask]
    
    term1 = (sigma / r_valid)**2 - 1
    term2 = (rc / r_valid)**2 - 1
    
    dU_dr = (-2 * epsilon * sigma**2 / r_valid**3) * (term2**2) + \
            (4 * epsilon * term1 * term2 * rc**2 / r_valid**3)

    # Calculate unit vectors safely
    #with np.errstate(divide='ignore', invalid='ignore'):
    unit_vectors = np.nan_to_num(delta[mask] / r_valid[:, None])  # Shape (M, 2)

    # Calculate final force vectors
    
    forces[mask] = -dU_dr[:, None] * unit_vectors  # Multiply scalar force by unit vector

    total_forces = np.sum(forces, axis=1)  # Shape: (N, 2)

    return total_forces


def prepare_cell_lists(positions, box_w, box_h, rc):
    num_cells_x = int(box_w // rc)
    num_cells_y = int(box_h // rc)
    num_cells = num_cells_x * num_cells_y

    cell_ids_x = (positions[:, 0] // rc).astype(np.int32) % num_cells_x
    cell_ids_y = (positions[:, 1] // rc).astype(np.int32) % num_cells_y
    cell_ids = cell_ids_y * num_cells_x + cell_ids_x

    cell_indices = [[] for _ in range(num_cells)]
    for idx, cell_id in enumerate(cell_ids):
        cell_indices[cell_id].append(idx)

    # Convert lists to Numba-friendly arrays
    cell_indices = [np.array(cell, dtype=np.int32) for cell in cell_indices]

    # Precompute neighbor maps
    neighbor_offsets = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [-1, 1]], dtype=np.int32)
    neighbor_map = []

    for cx in range(num_cells_x):
        for cy in range(num_cells_y):
            neighbors = []
            for dx, dy in neighbor_offsets:
                nx = (cx + dx) % num_cells_x
                ny = (cy + dy) % num_cells_y
                neighbor_id = ny * num_cells_x + nx
                neighbors.append(neighbor_id)
            neighbor_map.append(np.array(neighbors, dtype=np.int32))

    return cell_indices, neighbor_map


@njit(parallel=True)
def compute_distances_with_cutoff_numba(positions, box_w, box_h, cell_indices, neighbor_map, rc):
    N = positions.shape[0]
    rc2 = rc * rc
    box_size = np.array([box_w, box_h])

    distance_matrix = np.zeros((N, N, 2))

    # prange is from numba und Parallelisiert den for loop auf mehrere cpus
    for current_cell in prange(len(neighbor_map)):
        particles_i = cell_indices[current_cell]
        n_i = len(particles_i)
        if n_i == 0:
            continue

        # 1 Abstände innerhalb der aktuellen Zelle berechnen (ohne doppelte Berechnungen)
        for idx1 in range(n_i):
            i = particles_i[idx1]
            for idx2 in range(idx1 + 1, n_i):  # idx2 > idx1 → keine Doppelberechnung
                j = particles_i[idx2]

                rij = positions[j] - positions[i]
                rij -= np.round(rij / box_size) * box_size
                dist2 = rij[0]**2 + rij[1]**2

                if dist2 < rc2:
                    distance_matrix[i, j] = rij

        # 2 Abstände zu Nachbarzellen berechnen
        for neighbor_cell in neighbor_map[current_cell]:
            if neighbor_cell == current_cell:
                continue  # Wurde oben schon behandelt

            particles_j = cell_indices[neighbor_cell]
            if len(particles_j) == 0:
                continue

            for i in particles_i:
                for j in particles_j:
                    rij = positions[j] - positions[i]
                    rij -= np.round(rij / box_size) * box_size
                    dist2 = rij[0]**2 + rij[1]**2

                    if dist2 < rc2:
                        distance_matrix[i, j] = rij

    return distance_matrix

def calculate_forces_and_potential_between_particals(positions, box_w, box_h, cell_list, neighbours_map, rc, r_array, delta_epsilon):
    """
    Calculate forces between particles based on the Wang-Frenkel potential.
    """
    #cell_list, neighbours_map = prepare_cell_lists(positions, box_w, box_h, rc)
    #partical_distances = compute_distances_with_cutoff_numba(positions, box_w, box_h, cell_list, neighbours_map, rc)
    partical_distances = calculate_distance_between_particals(positions, box_w, box_h)
    #print(f"Partical distances: {partical_distances.shape}")
    N = partical_distances.shape[0]
    #print(f"R values: {r_values.shape}")
    # Calculate forces using the Wang-Frenkel potential
    r = np.linalg.norm(partical_distances, axis=-1)  # Shape: (N, N)
    g = radial_distribution(r, r_array, delta_epsilon, box_w, box_h)
    wf_force_values = wang_frenkel_force_vector(r, partical_distances,  rc=rc)
    wf_pot_energy_values = wang_frenkel(r, rc=rc)
    #print("wf_pot_energy_values", wf_pot_energy_values)
    return wf_force_values, wf_pot_energy_values, g
    # Calculate forces using the Wang-Frenkel potential

@njit()
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


def velocity_verlet(positions, velocities, forces, dt, box_size, cell_list, neighbour_map, delta_thermostat, temperature, step, thermo_interval, r_array, delta_epsilon, m=1.0, kb = 1.0, rc=2.5):
    """One step of the velocity-Verlet algorithm."""
    # Half-step velocity update
    velocities_half = velocities.copy() +  0.5 * forces * dt / m

    # Rescale Velocity - Thermostat
    # Apply Heyes thermostat occasionally
    if step % thermo_interval == 0:
        kin_energy_half_step = 0.5 * m * np.sum(np.linalg.norm(velocities_half, axis=1)**2)
        z = rescale_factor_heyes_thermostat(delta_thermostat)

        if acceptance_prob_thermostat(kin_energy_half_step, z, temperature, kb, velocities.shape[1], velocities.shape[0]):
            velocities_half *= z

    # Position update
    positions += velocities_half * dt
    positions %= box_size  # Periodic boundary conditions

    # Calculate Force and Potential Energy
    forces_new, potential_energy, g = calculate_forces_and_potential_between_particals(positions, box_size[0], box_size[0], cell_list, neighbour_map, rc, r_array, delta_epsilon)

    # Full-step velocity update
    velocities = velocities_half + 0.5 * forces_new * dt / m
    
    # Calculate Energy
    kinetic_energy = 0.5 * m * np.sum(np.linalg.norm(velocities, axis=1)**2)
    total_energy = kinetic_energy + potential_energy
    
    # Calculate Temperature of the System
    N = positions.shape[0]
    d = velocities.shape[1]
    Nf = d * N - d - 1
    kinetic_temperature = (kinetic_energy) / (Nf * kb)

    return positions, velocities, forces_new, potential_energy, kinetic_energy, total_energy, kinetic_temperature, g

    
def rescale_factor_heyes_thermostat(delta):
    """Apply Heyes rescaling to velocities."""
    xi = np.random.uniform(-delta, delta) # -ln(delta)
    z = np.exp(-xi)  
    return z

def acceptance_prob_thermostat(kin_energy, z, T, kB, d, N):
    exponent = d * (N - 1)
    weight = z**exponent * np.exp(-kin_energy * (z**2 - 1) / (kB * T))
    return np.random.uniform() < min(1, weight)

def radial_distribution(rij, r_array, delta_epsilon, box_w, box_h):
    N = rij.shape[0]  # Anzahl Teilchen
    N2 = N * N        # N^2
    L2 = box_w * box_h  # Fläche der Box

    # Entferne Nullabstände (z. B. i == j)
    non_zero_rij = rij[rij != 0.0]

    # Halbe Diagonale für periodische Entfernung
    R = 0.5 * np.sqrt(box_w**2 + box_h**2)

    radial_distribution = np.zeros(len(r_array))

    for i, r in enumerate(r_array):
        # Periodischer Abstand im Betrag
        diff = np.abs((non_zero_rij - r + R) % (2 * R) - R)
        #diff = np.abs((non_zero_rij - r))
        # Bin-Filter
        within_bin = diff < delta_epsilon
        count = np.count_nonzero(within_bin) / 2

        # Normierung
        pair_count = N * (N - 1) 
        shell_area = 2 * np.pi * r * delta_epsilon  # Fläche des Rings
        ideal_count =  (pair_count / L2) * shell_area          # Erwartete Anzahl im Ring
        g_r_value = count / ideal_count

        radial_distribution[i] = g_r_value

    return radial_distribution


def print_initial_conditions(num_particals, radius, box_size, initial_T, delta_epsilon, delta_t, num_steps):
    print("num_particals", num_particals)
    print("radius", radius)
    print("box_size", box_size)
    print("initial_T", initial_T)
    print("delta_epsilon", delta_epsilon)
    print("delta_t", delta_t)
    print("num_steps", num_steps)

#@njit(parallel=True)
def diffusion_constant_with_r(r_s, dt):
    num_time_steps, num_particles, dim = r_s.shape
    total_time = np.arange(num_time_steps) * dt
    msd = for_loop_auto_correlation_r(num_time_steps, r_s)

    fit_start = num_time_steps // 3
    fitt_end = 1000
    slope, _ = np.polyfit(total_time[fit_start:-fitt_end], msd[fit_start:-fitt_end], 1)
    D = slope / (2 * dim)
    return D, msd


#@njit(parallel=True)
def diffusion_constant_with_v_(v_s, dt):
    num_time_steps, num_particles, dim = v_s.shape
    v_s -= np.mean(v_s, axis=1, keepdims=True)
    # First 20% of timsteps - discard because is not balanced
    timesteps_equilibrium = num_time_steps // 5
    clean_time_steps = num_time_steps - timesteps_equilibrium
    vel_auto_cor = for_loop_auto_correlation_v(clean_time_steps, v_s[timesteps_equilibrium:])

    # Cutoff 
    cutoff = int(0.5 * len(vel_auto_cor))
    #cutoff = np.argmax(vel_auto_cor <= 0) if np.any(vel_auto_cor <= 0) else len(vel_auto_cor)
    D = np.trapezoid(vel_auto_cor[:cutoff], dx=dt) / dim

    return D, vel_auto_cor, clean_time_steps

def diffusion_constant_fft(v_s, dt):
    num_steps, num_partical, dim = v_s.shape
    v_s -= np.mean(v_s, axis=1, keepdims=True)
    # First 20% of timsteps - discard because is not balanced
    timesteps_equilibrium = num_steps // 2
    clean_timeseps = num_steps - timesteps_equilibrium
    # Reshape to (num_steps, total_dims)
    v_flat = v_s[timesteps_equilibrium:].reshape(clean_timeseps, -1) # Mitteln über dimensionen und partical 

    v_s_fft = np.fft.fft(v_flat, axis=0)
    power_spectrum = np.abs(v_s_fft) ** 2 

    # Average of all particals 
    G_omega = np.mean(power_spectrum, axis=1) / (clean_timeseps * dt)

    vel_auto_cor = np.real(np.fft.ifft(G_omega)) / clean_timeseps

    # Cutoff because periodic at the end
    cutoff = int(0.5 * len(vel_auto_cor))
    #cutoff = np.argmax(vel_auto_cor <= 0) if np.any(vel_auto_cor <= 0) else len(vel_auto_cor)
    D = np.trapezoid(vel_auto_cor[:cutoff], dx=dt) / dim

    # Freqnedys for Gomega plot
    freqs = np.fft.fftfreq(num_steps, d=dt)


    return D, vel_auto_cor, freqs, G_omega , clean_timeseps

@njit(parallel=True)
def for_loop_auto_correlation_r(num_time_steps, r_s):
    msd = np.zeros(num_time_steps)
    
    for tau in prange(num_time_steps):
        print("current r tau", tau)
        diff = r_s[tau:] - r_s[:num_time_steps - tau] # Periodische Verschiebung berücksichtigt
        msd[tau] = np.mean(np.sum(diff**2, axis=2))

    return msd

@njit(parallel=True)
def for_loop_auto_correlation_v(num_time_steps, v_s):
    vel_auto_cor = np.zeros(num_time_steps)

    for tau in prange(num_time_steps):
        print("current v tau", tau)
        v_tau = v_s[tau:, :, :]
        v_t = v_s[:num_time_steps - tau, :, :]
        v_dot_pro = np.sum(v_tau * v_t, axis=-1)  # shape: (T - tau, N)
        vel_auto_cor[tau] = np.mean(v_dot_pro)    # → Skalar

    return vel_auto_cor



#----------------Programm--------------------------------------

# Simulation Parameters
num_particles = 300
radius = 1.
box_w, box_h = 30.0, 30.0
initial_T = 4500.0
m = 1.0
kb = 1.0  # Boltzmann constasnt
sigma = 1.0 # lengtsh scsale
epsilon = 1.0  # eneregy scale
delta_t = 0.001 * np.sqrt(m * sigma ** 2 / epsilon) # time step
rc = box_w   # Cutoff radius for neighbor search
rc_skin = 0. * rc
rc_de = rc + rc_skin
delta_thermostat = 0.015
num_steps = 15000
time_total = np.arange(num_steps) * delta_t 
thermo_interval = 2


kin_energies = {}
pot_energies = {}
tot_energies = {}
kinetic_temperature = {}
radial_distributions = {}
r_s = {}
v_s = {}

given_temperatures = [initial_T]
delta_epsilon = radius / 4
r_max = np.sqrt(box_w ** 2 + box_h ** 2) / 2
r_array = np.arange(1. * radius, r_max, delta_epsilon)



# Calculate initial particals position, velocity etc.
particals_position, packing_fraction = initialize_particals_on_surface(num_particles, radius, box_w, box_h)
particals_velocity = initialize_velocities_of_particals(len(particals_position), initial_T, kb, m)
print("True number particals", particals_position.shape[0])
# Plot initial conditions
#plot_initial_condition("initial_particals", particals_position, particals_velocity, packing_fraction, box_w, box_h, initial_T)



#------- Sart Simulation-----
for temp in given_temperatures:
    # Set initial positions and velocities
    positions = particals_position.copy()
    velocities = particals_velocity.copy()  
    last_positions = positions.copy()

    # Initialize dictionarys
    kin_energies[temp] = np.zeros(num_steps)
    pot_energies[temp] = np.zeros(num_steps)
    tot_energies[temp] = np.zeros(num_steps)
    kinetic_temperature[temp] = np.zeros(num_steps)
    radial_distributions[temp] = np.zeros((num_steps, len(r_array)))
    
    # Position and Velocities over all Timesteps
    r_s[temp] = np.zeros((num_steps + 1, positions.shape[0], positions.shape[1]))
    v_s[temp] = np.zeros((num_steps + 1, velocities.shape[0], velocities.shape[1]))
    r_s[temp][0, :, :] = positions
    v_s[temp][0, :, :] = velocities



    # Calculate cell_list and neighbour list for the first time
    cell_list, neighbour_map = prepare_cell_lists(positions, box_w, box_h, rc_de)
    
    # initial forces and potential
    forces, potential, _ = calculate_forces_and_potential_between_particals(positions, box_w, box_h, cell_list, neighbour_map, rc, r_array, delta_epsilon)

    for i in tqdm(range(num_steps), desc="Simulating"):
        #print("current step:", i)
        # max displacement for adjusting neighbourts_list (Optional)
        max_displacement = np.max(np.linalg.norm(positions - last_positions, axis=1))
        if max_displacement > rc_skin / 8:
            cell_list, neighbour_map = prepare_cell_lists(positions, box_w, box_h, rc_de)
            last_positions = positions.copy()  # Reset displacement tracking

        #cell_list, neighbour_map = prepare_cell_lists(positions, box_w, box_h, rc_de)
        # make Timestep and return necessary Variables
        positions, velocities, forces, pot_energies[temp][i], kin_energies[temp][i], tot_energies[temp][i], kinetic_temperature[temp][i], radial_distributions[temp][i, :] = velocity_verlet(positions, velocities, forces, delta_t, [box_w, box_h], cell_list, neighbour_map, delta_thermostat, temp, i, thermo_interval, r_array, delta_epsilon, rc=rc)
        r_s[temp][i + 1, :, :] = positions
        v_s[temp][i + 1, :, :] = velocities

    





#-----------------Plot Results---------------

# Print Parameter
print_initial_conditions(num_particles, radius, [box_w, box_h], initial_T, delta_epsilon, num_steps, delta_t)
# Plot final Conditin
plot_initial_condition("final_parical_positions", positions, velocities, packing_fraction, box_w, box_h, initial_T)
# Plot Energies  s
plot_energies(time_total, kin_energies, pot_energies, tot_energies, given_temperatures, packing_fraction)
# Plot Temperature
plot_temperature(time_total, kinetic_temperature, given_temperatures, packing_fraction)
# Plot Radial Distribution
plot_radial_distribution(r_array, radial_distributions, given_temperatures, packing_fraction)

# Plot Diffusion Constant
D = plot_diffusion_constant(r_s, v_s, delta_t, given_temperatures)
#print("Diffusi9n Constant", D)
#----------------------End----------------------



# %%
