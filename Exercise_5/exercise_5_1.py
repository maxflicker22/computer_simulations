import numpy as np
import matplotlib.pyplot as plt



def plot_acceptance_rate(ks, acceptance_rate):
    """
    Plot the acceptance rate against the k values.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(ks, acceptance_rate, marker='o', linestyle='-', color='purple')
    plt.xlabel('k values')
    plt.ylabel('Acceptance Rate')
    plt.title('Acceptance Rate vs. k values')
    plt.grid(True)
    plt.show()

def plot_energie_datas(beta_k, min_energies, average_energies, scaled_variance_energies, Energies_list):
    # Convert beta_k and energy lists to numpy arrays for consistent plotting
    betas = np.array(beta_k[1:])  # Skip the first beta (beta_start)
    min_energies = np.array(min_energies)
    average_energies = np.array(average_energies)
    scaled_variance_energies = np.array(scaled_variance_energies)

    # Flatten all energies into a single list for plotting
    all_energies = []
    all_steps = []

    step_counter = 0
    for beta in betas:
        energies_at_beta = Energies_list[beta]
        all_energies.extend(energies_at_beta)
        all_steps.extend(range(step_counter, step_counter + len(energies_at_beta)))
        step_counter += len(energies_at_beta)

    # Create a figure with 4 subplots (3 horizontal + 1 full-width)
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 3)

    # 1️⃣ Minimum Energy Plot
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(betas, min_energies, marker='o', linestyle='-', color='royalblue', label='Min Energy')
    ax1.set_xlabel(r'$\beta$')
    ax1.set_ylabel('Minimum Energy')
    ax1.set_title('Minimum Energy vs. Inverse Temperature')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()

    # 2️⃣ Average Energy Plot
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(betas, average_energies, marker='s', linestyle='-', color='seagreen', label='Average Energy')
    ax2.set_xlabel(r'$\beta$')
    ax2.set_ylabel('Average Energy')
    ax2.set_title('Average Energy vs. Inverse Temperature')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()

    # 3️⃣ Scaled Variance Plot
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(betas, scaled_variance_energies, marker='^', linestyle='-', color='darkorange', label='Scaled Variance Energy')
    ax3.set_xlabel(r'$\beta$')
    ax3.set_ylabel(r'Scaled Variance Energy')
    ax3.set_title(r'Scaled Variance Energy vs. Inverse Temperature')
    ax3.grid(True, linestyle='--', alpha=0.6)
    ax3.legend()

    # 4️⃣ Energy vs. Global Step Index Plot (spanning all columns)
    ax4 = fig.add_subplot(gs[1, :])
    ax4.plot(all_steps, all_energies, linestyle='-', color='purple', label='Energy')
    ax4.set_xlabel('Step Index')
    ax4.set_ylabel('Energy')
    ax4.set_title('Energy vs. Step Index (Global Time Series)')
    ax4.grid(True, linestyle='--', alpha=0.6)
    ax4.legend()

    plt.suptitle('Energy Metrics vs. Inverse Temperature (Simulated Annealing)', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def create_initial_city_positions(num_cities, box_size, seed):
    """
    Create initial city positions within a square box.
    """
    np.random.seed(seed)  # Set the random seed for reproducibility
    cities = np.random.rand(num_cities, 2) * box_size
    return cities , np.arange(num_cities)

def plot_cities(cities, box_size, radius_of_city):
    """
    Plot the cities on a 2D plane.
    """
    plt.figure(figsize=(8, 8))
    plt.xlim(0, box_size)
    plt.ylim(0, box_size)
    plt.scatter(cities[:, 0], cities[:, 1], s=radius_of_city*100, c='blue', alpha=0.5, label='Cities')
    plt.title('City Positions')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.grid()
    plt.legend()
    plt.show()
    print("Initial city positions created successfully.")
    print("Number of cities created:", len(cities))
    print("City positions:\n", cities)

def calculate_distance_of_Salesman(cities):
    """
    Calculate the distance between all orderd cities.
    """
    distance = cities[1:, :] - cities[:-1, :]
    loop_distance = cities[0, :] - cities[-1, :]
    distance = np.vstack((distance, loop_distance))
    # Calculate the total distance
    return np.sum(np.linalg.norm(distance, axis=1))

def reorder_citiy_configuratin(cities, order):
    """
    Reorder the cities based on the given order.
    """
    return cities[order]

def propose_new_city_configuration(order, num_cities):
    """
    Propose a new city configuration.
    """
    first_index = np.random.randint(1, num_cities - 2)
    second_index = np.random.randint(first_index + 1, num_cities)
    new_order = order.copy()
    new_order[first_index], new_order[second_index] = new_order[second_index], new_order[first_index]
    print(f"Proposed new order: {new_order}")
    return new_order

def acceptance_probability(old_distance, new_distance, beta):
    """
    Calculate the acceptance probability for the new configuration.
    """
    if new_distance < old_distance:
        return 1.0
    else:
        return np.exp((old_distance - new_distance) * beta)

def try_update_order(cities, old_distance, order, beta):
    """
    Try to update the order of cities based on the Metropolis criterion.
    """

    new_order = propose_new_city_configuration(order, len(order))
    new_distance = calculate_distance_of_Salesman(cities[new_order])
    
    prob = acceptance_probability(old_distance, new_distance, beta)
    
    if np.random.rand() < prob:
        print("Accepted new configuration.")
        return new_order, new_distance
    else:
        print("Rejected new configuration.")
        return order, old_distance
    
def update_beta(beta, k, q):
    """
    Update the beta parameter based on the given k and q.
    """
    return beta * (k ** q)


def discuss_energies(energies_at_specific_beta, beta):
    """
    Discuss the energies at a specific beta value.
    """
    print(f"Energies at beta = {beta}:")
    energies = np.array(energies_at_specific_beta)
    length = len(energies)
    if length == 0:
        print("No energies recorded for this beta value.")
        return
    min_energy = np.min(energies)
    average_energy = np.mean(energies)
    variance_energy = np.var(energies)
    scaled_variance_energy = variance_energy * beta ** 2
    return  min_energy, average_energy, scaled_variance_energy
    




# Define parameters 
box_size = 1.0  # Sizse of the box
num_cities = 15  # Number of cities
radius_of_city = 1.0  # Radius of each city
seed = 42  # Random seed for reproducibility
beta_start = 1 # Inverse temperature parameter
beta_k = []
ks = np.arange(1, 10)  # Range of k values for beta update
print("ks:", ks)
q = 2
steps_per_temperature = num_cities ** 2
# Initialize the list to store energies for each beta value
Energies_list = {}

# Initialize lists to store energies at specific beta values
min_energies = []
average_energies = []
scaled_variance_energies = []
acceptance_rate = []



# Create initial city positions and order
cities, order = create_initial_city_positions(num_cities, box_size, seed=seed)
# Calculate the initial distance
old_disance = calculate_distance_of_Salesman(cities[order])

# Add First Beta to Beta_k list
beta_k.append(beta_start)
# Delete random seed to avoid confusion
np.random.seed()

# Start loop over cooling steps
for k in ks:
    if k == 1:
        beta_k.append(update_beta(beta_start, k, q))
    else:
        beta_k.append(update_beta(beta_k[-1], k, q))

    # Initialize the energies list for the current beta
    Energies_list[beta_k[-1]] = []
    Energies_list[beta_k[-1]].append(old_disance)
    print("Energy_list for beta =", Energies_list[beta_k[-1]])

    accepted_energies = 1

    # Loop over steps for the current beta
    old_distance = Energies_list[beta_k[-1]][-1]
    for step in range(steps_per_temperature):
        order, current_distance = try_update_order(cities, old_distance, order, beta_k[-1])
        if current_distance != old_distance:
            Energies_list[beta_k[-1]].append(current_distance)
            old_distance = current_distance  # <== Hier wird die Referenzdistanz aktualisiert!

        print("Current Step:", step + 1, "/", steps_per_temperature)
    
    # Evaluation of Accepted Energies
    accepted_energies = len(Energies_list[beta_k[-1]])
    acceptance_rate.append(accepted_energies / steps_per_temperature)       
    print("Acceepted Energies:", accepted_energies)

    # Discuss energies at the current beta value
    min_energy, average_energy, scaled_variance_energy = discuss_energies(Energies_list[beta_k[-1]], beta_k[-1])
    min_energies.append(min_energy)
    average_energies.append(average_energy)
    scaled_variance_energies.append(scaled_variance_energy)




# Plotting the cities
plot_cities(cities, box_size, radius_of_city)

# Plot the energy data
plot_energie_datas(beta_k, min_energies, average_energies, scaled_variance_energies, Energies_list)

# Plot the acceptance rate
plot_acceptance_rate(ks, acceptance_rate)



