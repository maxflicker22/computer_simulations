#%%
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as consts

# Generate the (i,j) coordinate grid
def plot_grid():
    # Create a grid of indices
    i_vals, j_vals = np.meshgrid(np.arange(grid_range), np.arange(grid_range), indexing='ij')

    # Plot the grid of unit vectors
    plt.figure(figsize=(8, 8))
    plt.quiver(j_vals, i_vals, grid_of_unitvec[:, :, 0], grid_of_unitvec[:, :, 1], pivot='middle')
    plt.gca().invert_yaxis()  # So (0,0) is at top-left like your array indexing
    plt.title("10×10 Grid of Unit Vectors")
    plt.grid()
    plt.show()


# -------------------------------------------------------------

# a) Generate 10x10 grid of unit Vector
unitvector = lambda phi : np.array([np.cos(phi), np.sin(phi)])

# Define Grid Parameter
grid_range = 10
range_of_angels = [0, 2 * np.pi] # 0 to 2pi

# random Seed
np.random.seed(42) # Set seed for reproducibility

# Generate 10x10 grid of unit vectors
grid_of_unitvec = np.array([[unitvector(np.random.uniform(min(range_of_angels), max(range_of_angels))) for _ in range(grid_range)] for _ in range(grid_range)])

#print("grid_of_unitvector shape" , grid_of_unitvec.shape)
#print("grid_of_unitvector" , grid_of_unitvec)

# Define rotation Matrix function depending on the angle theta
rot_matrix = lambda theta: np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])

epsilon = 1

def rotated_one_random_unit_vec():
    rotation_angle = np.random.uniform(0, 2*np.pi)
    #print("rotation_angle: ", rotation_angle)
    (rotated_index_i, rotated_index_j) = np.random.randint(0, grid_range, 2)
    grid_of_unitvec[rotated_index_i, rotated_index_j] = np.dot(rot_matrix(rotation_angle), grid_of_unitvec[rotated_index_i, rotated_index_j])
    return rotation_angle, rotated_index_i, rotated_index_j 


#rotated_one_random_unit_vec()




def back_rotate_to_old_state(rotation_angle, rotated_index_i, rotated_index_j):
    # Revert to the old state (not shown here)
    #print("rotation_angle: ", rotation_angle)
    grid_of_unitvec[rotated_index_i, rotated_index_j] = np.dot(rot_matrix(-rotation_angle), grid_of_unitvec[rotated_index_i, rotated_index_j]).copy()


def potential_energy_U(u_i, u_j):
    return -1 * epsilon * (np.sum(u_i * u_j, axis=-1))**2 # axis -1 is the last axis, so it sums over the last axis; x and y components

# Calculate the potential energy of the grid and returns the total energy
def calculate_potential_energy_on_grid(): 
    # Returns the total energy of the grid
    # Periodisch verschobene Nachbarn
    left  = np.roll(grid_of_unitvec, shift=1, axis=1)
    right = np.roll(grid_of_unitvec, shift=-1, axis=1)
    up    = np.roll(grid_of_unitvec, shift=1, axis=0)
    down  = np.roll(grid_of_unitvec, shift=-1, axis=0)

    # Energiebeiträge berechnen
    energy_left  = potential_energy_U(grid_of_unitvec, left)
    energy_right = potential_energy_U(grid_of_unitvec, right)
    energy_up    = potential_energy_U(grid_of_unitvec, up)
    energy_down  = potential_energy_U(grid_of_unitvec, down)

    # Gesamte lokale Energie
    pot_energy_grid = energy_left + energy_right + energy_up + energy_down

    total_energy = np.sum(pot_energy_grid) / 2 # because each pair is counted twice
    return total_energy

# Acceptance probability Metropolis
def probability_to_accept(old_energy, new_energy, T):
    # Metropolis acceptance probability
    if new_energy < old_energy:
        return 1
    else:
        return np.exp(-(new_energy - old_energy) / (consts.k * T)) # k is Boltzmann constant, T is temperature
    
# Monte Carlo simulation
def monte_carlo_simulation(n_steps, T):
    interaction_energy_list = np.zeros(n_steps)
    initial_energy = calculate_potential_energy_on_grid()
    interaction_energy_list[0] = initial_energy
    accepted_steps = 0
    S_list = []
    Q_list = []
    for step  in range(1, n_steps):
        #print("Step: ", step)
        # Store the old energy
        old_energy = initial_energy.copy()

        # Rotate one random unit vector
        r= rotated_one_random_unit_vec()
        rotated_angel, rotated_index_i, rotated_index_j = r[0], r[1], r[2]

        # Calculate the new energy
        new_energy = calculate_potential_energy_on_grid()

        # Calculate the acceptance probability
        acceptance_prob = probability_to_accept(old_energy, new_energy, T)
        #print("Acceptance Probability: ", acceptance_prob)
        # Accept or reject the new state
        if np.random.uniform(0, 1) < acceptance_prob: #np. random uniform(0, 1) generates a random number between 0 and 1 inclusive zero exclusive one
            accepted_steps += 1
            initial_energy = new_energy  # Update the energy only if accepted
            interaction_energy_list[step] = new_energy.copy() # Store the new energy
            Q, S = nematic_order_tensor(grid_of_unitvec)
            Q_list.append(Q)
            S_list.append(S)
        
        else:
            # Revert to the old state (not shown here)
            interaction_energy_list[step] = old_energy.copy() # Store the old energy
            back_rotate_to_old_state(rotated_angel, rotated_index_i, rotated_index_j)
            Q, S = nematic_order_tensor(grid_of_unitvec)
            Q_list.append(Q)
            S_list.append(S)
            pass

    return interaction_energy_list, accepted_steps, Q_list, S_list




# b) & c) write monte carlo script and plot the interaction energy
# Ab hier einkommentieren wenn b c ausgeführt werden soll

#-------------------------------------------------

# pot_energy_grid = np.zeros((grid_range, grid_range))
# T_list = [0.01, 300, 1000, 5000, 10000, 100000] # K

# last_interaction_energy = []
# accepted_steps_list = []


# plt.figure(figsize=(12, 8))  # Create one big figure BEFORE the loop

# for i in range(len(T_list)):
#     n_steps = 15000
#     T = T_list[1]
#     #plot_grid()
    
#     interaction_energy_list, accepted_steps = monte_carlo_simulation(n_steps, T)
#     #plot_grid()

#     # Plot each interaction energy curve, label it by temperature
#     plt.plot(interaction_energy_list, label=f"T = {T} K")

#     print("Accepted Steps: ", accepted_steps)
#     accepted_steps_list.append(accepted_steps)
#     print("Last interaction energy: ", interaction_energy_list[-1])
#     last_interaction_energy.append(interaction_energy_list[-1])

#     # Generate a new 10x10 grid of unit vectors (reset for next temperature)
#     #np.random.seed(42) # Hier einkommentieren für immer gleichen start
#     grid_of_unitvec = np.array([
#         [unitvector(np.random.uniform(min(range_of_angels), max(range_of_angels))) for _ in range(grid_range)]
#         for _ in range(grid_range)
#     ])

# # After the loop: add labels, title, legend, grid
# plt.xlabel("Monte Carlo Steps")
# plt.ylabel("Interaction Energy")
# plt.title("Monte Carlo Simulation of Interaction Energy for Different Temperatures")
# plt.legend()
# plt.grid()
# #plt.savefig("monte_carlo_simulation_vs_steps.png", dpi=300)  # Save the figure
# plt.show()
#----------------------------------------------------------


##### RESULT #####
# Hängt nicht von Temperatur ab, da boltzmann Konstante viel zu klein ist
# Hängt von Startkonfiguratin ab
# Wenn meed Seed erzeugt wird und für unterschiedliche Temperaturen mit Boltzmann Konstante, dann kommt man exakt
# auf die gleiche Ergebnisse (Accepted Steps und Interaction Energy)
# niedrigster Energiezustand ist -200 diese nähern sich alle konfigurationen an, aber nicht exakt gleicdh
# Wenn energie normal wäre dann würde es mit boltzjann konstanten auch besser gehen

# Circa nach 4000 Schritten ist die Energie konstant und scheint Zustand scheint zu konvergieren

## INFO ##
# plot grid ist zurzeit auskommentiert, weil sonst interaction energys in einem plot nicht funktioniert
# -------------------------------------------------------------



    
    
### additional information ###

# Detailed Balance: Conndition ⎪ πi * pij = πj * pji
# Is detail balanced # is reversable, is a form of symmetry
# Is it Detail Balanced???
# Yes, because the probability to comming from the global state and comming back i sthe same
# Picking the Molecule is uniform and picking the angel as well, additionally the angel with the molecule gets rotated as the same probability as the angel what you need if you want to come back to the previous step. for example. rotate π, and you need π again to come back to the old state




### Task2 ###
def nematic_order_tensor(grid_of_unitvec):
    # grid_of_unitvec: (Nx, Ny, 2) array of unit vectors
    u_x = grid_of_unitvec[:, :, 0]
    u_y = grid_of_unitvec[:, :, 1]

    # Compute averages
    u_xx_mean = np.mean(u_x * u_x)  # <u_x * u_x>
    u_yy_mean = np.mean(u_y * u_y)  # <u_y * u_y>
    u_xy_mean = np.mean(u_x * u_y)  # <u_x * u_y>
    
    # Build 2x2 tensor (without subtracting identity yet)
    Q_tensor = np.array([
        [2 * u_xx_mean, 2 * u_xy_mean],
        [2 * u_xy_mean, 2 * u_yy_mean]
    ])
    
    # Subtract identity matrix # Kronecka
    Q_tensor -= np.eye(2)

    # Calculate eigenvalues
    eigenvalues = np.linalg.eigvals(Q_tensor)
    S = np.max(eigenvalues)

    return Q_tensor, S


# def nematic_order_parameter(grid_of_unitvec):
#     general_director_field = np.zeros(grid_of_unitvec.shape)
#     general_angel = np.pi / 2
#     general_director_field[:, :, 0] = np.cos(general_angel)
#     general_director_field[:, :, 1] = np.sin(general_angel)

#     identity_matrix = np.eye(grid_of_unitvec.shape[0])

#     S_matrix = 2 * (np.sum(grid_of_unitvec * general_director_field, axis=-1))**2 - identity_matrix
#     S = np.sum(S_matrix) / S_matrix.size

#     return S
    


# pot_energy_grid = np.zeros((grid_range, grid_range))
# T_list = {"T1": 300 / consts.k, "T2": 300}
# last_interaction_energy = []
# accepted_steps_list = []

# S_list = {"T1": [], "T2": []}
# Q_list = {"T1": [], "T2": []}
# plot_grid()

# plt.figure(figsize=(12, 8))  # Create one big figure BEFORE the loop

# for key, T in T_list.items():

#     np.random.seed(42) # Hier einkommentieren für immer gleichen start
#     grid_of_unitvec = np.array([
#         [unitvector(np.random.uniform(min(range_of_angels), max(range_of_angels))) for _ in range(grid_range)]
#         for _ in range(grid_range)
#     ])

#     # Number of Monte Carlo steps
#     n_steps = 100000

#     interaction_energy_list, accepted_steps, Q_list_mc, S_list_mc = monte_carlo_simulation(n_steps, T)
    
#     print("Nematic order tensor Q: ", Q_list_mc)
#     print("Nematic order parameter S: ", S_list_mc)
#     S_list[key] = S_list_mc
#     Q_list[key] = Q_list_mc
#     #plot_grid()

#     # Plot each interaction energy curve, label it by temperature
#     plt.plot(interaction_energy_list, label=f"T = {T} K")

#     print("Accepted Steps: ", accepted_steps)
#     accepted_steps_list.append(accepted_steps)
#     print("Last interaction energy: ", interaction_energy_list[-1])
#     last_interaction_energy.append(interaction_energy_list[-1])

#     # Generate a new 10x10 grid of unit vectors (reset for next temperature)
    

# # After the loop: add labels, title, legend, grid
# plt.xlabel("Monte Carlo Steps")
# plt.ylabel("Interaction Energy")
# plt.title("Monte Carlo Simulation of Interaction Energy for Different Temperatures")
# plt.legend()
# plt.grid()
# #plt.savefig("monte_carlo_simulation_vs_steps.png", dpi=300)  # Save the figure
# plt.show()

# print("S_list: ", S_list)


# plot_grid()

#-----------------------------------------------------------

### Task 2 _ a - Plot Histogramm

def plot_S_histograms(S_list):
    # Create figure and two subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot for T1 (high temperature, random behavior)
    axes[0].hist(S_list["T1"], bins=40, color='skyblue', edgecolor='black')
    axes[0].set_title(r"$k_B T \gg \epsilon$")
    axes[0].set_xlabel("S Order Parameter")
    axes[0].set_ylabel("Frequency")
    axes[0].grid(True)

    # Plot for T2 (low temperature, strong order)
    axes[1].hist(S_list["T2"], bins=40, color='lightgreen', edgecolor='black')
    axes[1].set_title(r"$k_B T \ll \epsilon$")
    axes[1].set_xlabel("S Order Parameter")
    axes[1].set_ylabel("Frequency")
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig("S_histograms.png", dpi=300)  # Save the figure
    plt.show()

#plot_S_histograms(S_list)

### Task 2 _ b - Autocorrelation
def calculate_autocorrelation_with_fix_tau(S_list, tau):
    monte_carlo_steps = len(S_list)
    normalizatin_term = monte_carlo_steps - tau
    sum_over = monte_carlo_steps - tau
    S_list_x = S_list[:sum_over]
    S_list_y = S_list[tau:]
    S_list_x_minus_mean = S_list_x - np.mean(S_list_x)
    S_list_y_minus_mean = S_list_y - np.mean(S_list_y)

    nominator = np.sum(S_list_x_minus_mean * S_list_y_minus_mean)
    denominator = np.sqrt(np.sum(S_list_x_minus_mean**2) * np.sum(S_list_y_minus_mean**2))
    return nominator / denominator


def calculate_autocorrelation(S_list):
    tau_list = np.arange(0, 500, 1)  # Tau values from 0 to 500
    autocorrelation_list = np.zeros(len(tau_list))

    # Calculate the autocorrelation for each tau
    for tau in tau_list:
        autocorrelation_list[tau] = calculate_autocorrelation_with_fix_tau(S_list, tau)
    
    return autocorrelation_list

def plot_autocorrelation(S_list):
    # Create figure and two subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # Calculate autocorrelation for T1 and T2
    autocorrelation_T1 = calculate_autocorrelation(S_list["T1"])
    autocorrelation_T2 = calculate_autocorrelation(S_list["T2"])
    
    integrate_autocorrelation_T1 = integrate_autocorrelation(autocorrelation_T1)
    integrate_autocorrelation_T2 = integrate_autocorrelation(autocorrelation_T2)
    # Plot for T1 (high temperature, random behavior)
    axes[0].plot(autocorrelation_T1, color='skyblue', label="Autocorrelation")
    axes[0].plot(np.exp(-1 * np.arange(0, 500, 1) / integrate_autocorrelation_T1), color='red', linestyle='--', label="Exponential Decay")
    axes[0].set_title(rf"Autocorrelation $k_B T \gg \epsilon$ with $\tau_{{\mathrm{{int}}}} = {integrate_autocorrelation_T1:.2f}$")
    axes[0].set_xlabel("S Order Parameter")
    axes[0].set_ylabel("Autocorrelation")
    axes[0].grid(True)
    axes[0].legend()


    # Plot for T2 (low temperature, strong order)
    axes[1].plot(autocorrelation_T2, color='lightgreen', label="Autocorrelation")
    axes[1].plot(np.exp(-1 * np.arange(0, 500, 1) / integrate_autocorrelation_T2), color='red', linestyle='--', label="Exponential Decay")
    axes[1].set_title(rf"Autocorrelation $k_B T \ll \epsilon$ with $\tau_{{\mathrm{{int}}}} = {integrate_autocorrelation_T2:.2f}$")
    axes[1].set_xlabel("S Order Parameter")
    axes[1].set_ylabel("Autocorrelation")
    axes[1].grid(True)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("autocorrelation.png", dpi=300)  # Save the figure
    plt.show()


def integrate_autocorrelation(autocorrelation_list):
    # Integrate the autocorrelation function
    # Find the first index where it is negative
    first_neg_index = np.where(autocorrelation_list < 0)[0]
    if first_neg_index.size == 0:
        integration_range = np.arange(0, len(autocorrelation_list), 1)
    else:
        integration_range = np.arange(0, first_neg_index[0], 1)

    integrated_autocorrelation = 1/2 + np.sum(autocorrelation_list[integration_range] * (1 - (integration_range + 1) / len(autocorrelation_list)))
    print("Integrated Autocorrelation: ", integrated_autocorrelation)
    return integrated_autocorrelation
#  Call the function to plot the autocorrelation

#plot_autocorrelation(S_list)



#-----------------------------------------------------------

### Task 3 ###

def calculate_error_of_S_mean(variance_of_S_list, tau_int, n_steps):
    variance = variance_of_S_list / n_steps * tau_int * 2
    error = np.sqrt(variance)
    return error

pot_energy_grid = np.zeros((grid_range, grid_range))
T_list_part1 = np.linspace(0.01, 1 / consts.k, 500) # K]
T_list_part2 = np.linspace(1 / consts.k, 4 / consts.k, 20) # K]
T_list_combined = np.concatenate((T_list_part1, T_list_part2))
last_interaction_energy = []
accepted_steps_list = []

S_list_mean = []
S_error = []

plot_grid()



for T in T_list_combined:

    np.random.seed(42) # Hier einkommentieren für immer gleichen start
    grid_of_unitvec = np.array([
        [unitvector(np.random.uniform(min(range_of_angels), max(range_of_angels))) for _ in range(grid_range)]
        for _ in range(grid_range)
    ])

    # Number of Monte Carlo steps
    n_steps = 10000

    interaction_energy_list, accepted_steps, Q_list_mc, S_list_mc = monte_carlo_simulation(n_steps, T)
    
    print("Nematic order tensor Q: ", Q_list_mc)
    print("Nematic order parameter S: ", S_list_mc)
    S_list_mean.append(np.mean(S_list_mc))
    autocorrelation = calculate_autocorrelation(S_list_mc)
    tau_int = integrate_autocorrelation(autocorrelation)
    S_error.append(calculate_error_of_S_mean(np.var(S_list_mc), tau_int, n_steps))
    #plot_grid()

    

    print("Accepted Steps: ", accepted_steps)
    accepted_steps_list.append(accepted_steps)
    print("Last interaction energy: ", interaction_energy_list[-1])
    last_interaction_energy.append(interaction_energy_list[-1])

    # Generate a new 10x10 grid of unit vectors (reset for next temperature)
    
plt.figure(figsize=(12, 8))  # Create one big figure BEFORE the loop
# Plot each interaction energy curve, label it by temperature
plt.plot(S_list_mean, label=f"T = {T} K")
# After the loop: add labels, title, legend, grid
plt.xlabel("Monte Carlo Steps")
plt.ylabel("Interaction Energy")
plt.title("Monte Carlo Simulation of Interaction Energy for Different Temperatures")
plt.legend()
plt.grid()
#plt.savefig("monte_carlo_simulation_vs_steps.png", dpi=300)  # Save the figure
plt.show()



# %%
