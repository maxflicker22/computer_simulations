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
range_of_angels = [0, 0]

# Generate 10x10 grid of unit vectors
grid_of_unitvec = np.array([[unitvector(np.random.uniform(min(range_of_angels), max(range_of_angels))) for _ in range(grid_range)] for _ in range(grid_range)])

#print("grid_of_unitvector shape" , grid_of_unitvec.shape)
#print("grid_of_unitvector" , grid_of_unitvec)

# Define rotation Matrix function depending on the angle theta
rot_matrix = lambda theta: np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])

# Rotates one random unitvector of the grid around theeta
rotated_index_i = 0
rotated_index_j = 0
rotation_angle = 0

def rotated_one_random_unit_vec():
    rotation_angle = np.random.uniform(0, 2*np.pi)
    (rotated_index_i, rotated_index_j) = np.random.randint(0, 10, 2)
    grid_of_unitvec[rotated_index_i, rotated_index_j] = np.dot(rot_matrix(rotation_angle), grid_of_unitvec[rotated_index_i, rotated_index_j])


#rotated_one_random_unit_vec()




def back_rotate_to_old_state():
    # Revert to the old state (not shown here)
    grid_of_unitvec[rotated_index_i, rotated_index_j] = np.dot(rot_matrix(-rotation_angle), grid_of_unitvec[rotated_index_i, rotated_index_j])


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
def probability_to_accept(old_energy, new_energy):
    # Metropolis acceptance probability
    if new_energy < old_energy:
        return 1
    else:
        return np.exp(-(new_energy - old_energy) / (consts.k * T)) # k is Boltzmann constant, T is temperature
    
# Monte Carlo simulation
def monte_carlo_simulation(n_steps):
    interaction_energy_list = np.zeros(n_steps)
    print("interaction_energy_list shape: ", interaction_energy_list[0].shape)
    initial_energy = calculate_potential_energy_on_grid()
    print("interaction_energy_list shape: ", interaction_energy_list[0].shape)
    interaction_energy_list[0] = initial_energy
    accepted_steps = 0
    for step  in range(1, n_steps):
        print("Step: ", step)
        # Store the old energy
        old_energy = initial_energy.copy()

        # Rotate one random unit vector
        rotated_one_random_unit_vec()

        # Calculate the new energy
        new_energy = calculate_potential_energy_on_grid()

        # Calculate the acceptance probability
        acceptance_prob = probability_to_accept(old_energy, new_energy)

        # Accept or reject the new state
        if np.random.uniform(0, 1) < acceptance_prob: #np. random uniform(0, 1) generates a random number between 0 and 1 inclusive zero exclusive one
            accepted_steps += 1
            initial_energy = new_energy  # Update the energy only if accepted
            interaction_energy_list[step] = new_energy.copy() # Store the new energy
        else:
            # Revert to the old state (not shown here)
            interaction_energy_list[step] = old_energy.copy() # Store the old energy
            back_rotate_to_old_state()

            pass

    return interaction_energy_list, accepted_steps




# b) write monte carlo script
epsilon = 1
pot_energy_grid = np.zeros((grid_range, grid_range))
T = 0.0000000000000000000000001 # K

n_steps = 5000
plot_grid()
interaction_energy_list, accepted_steps = monte_carlo_simulation(n_steps)
plot_grid()

plt.figure(figsize=(10, 6))
plt.plot(interaction_energy_list, label="Interaction Energy")
plt.xlabel("Monte Carlo Steps")
plt.ylabel("Interaction Energy")
plt.title("Monte Carlo Simulation of Interaction Energy")
plt.legend()
plt.grid()
plt.show()

print("Accepted Steps: ", accepted_steps)
print("last interaction energy: ", interaction_energy_list[-1])



    
    
    






# Detailed Balance: Conndition ⎪ πi * pij = πj * pji
# Is detail balanced # is reversable, is a form of symmetry
# Is it Detail Balanced???
# Yes, because the probability to comming from the global state and comming back i sthe same
# Picking the Molecule is uniform and picking the angel as well, additionally the angel with the molecule gets rotated as the same probability as the angel what you need if you want to come back to the previous step. for example. rotate π, and you need π again to come back to the old state


    



# %%
