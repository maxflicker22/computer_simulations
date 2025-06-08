import numpy as np
import matplotlib.pyplot as plt

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
    second_index = np.random.randint(first_index + 1, num_cities - 1)
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
box_size = 10.0  # Size of the box
num_cities = 20  # Number of cities
radius_of_city = 1.0  # Radius of each city
seed = 42  # Random seed for reproducibility
beta_start = 1 # Inverse temperature parameter
beta_k = []
ks = np.arange(1, 110)  # Range of k values for beta update
q = 1.01
steps_per_temperature = num_cities ** 2
# Initialize the list to store energies for each beta value
Energies_list = {}

# Initialize lists to store energies at specific beta values
min_energies = []
average_energies = []
scaled_variance_energies = []



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
    for step in range(steps_per_temperature):
        # Try to update the order of cities
        order, current_distance = try_update_order(cities, old_disance, order, beta_k[-1])
              
        if Energies_list[beta_k[-1]][accepted_energies - 1] != current_distance:
            # Store the energy (distance) for the current beta
            Energies_list[beta_k[-1]].append(current_distance)

        print("Current Step:", step + 1, "/", steps_per_temperature)
    
    # Evaluation of Accepted Energies
    accepted_energies = len(Energies_list[beta_k[-1]])       
    print("Acceepted Energies:", accepted_energies)

    # Discuss energies at the current beta value
    min_energy, average_energy, scaled_variance_energy = discuss_energies(Energies_list[beta_k[-1]], beta_k[-1])
    min_energies.append(min_energy)
    average_energies.append(average_energy)
    scaled_variance_energies.append(scaled_variance_energy)


# Convert beta_k and energy lists to numpy arrays for consistent plotting
betas = np.array(beta_k[1:])  # Skip the first beta (beta_start)
min_energies = np.array(min_energies)
average_energies = np.array(average_energies)
scaled_variance_energies = np.array(scaled_variance_energies)

# Create a figure with 3 subplots
plt.figure(figsize=(16, 5))

# 1️⃣ Minimum Energy Plot
plt.subplot(1, 3, 1)
plt.plot(betas, min_energies, marker='o', linestyle='-', color='royalblue', label='Min Energy')
plt.xlabel(r'$\beta$')
plt.ylabel('Minimum Energy')
plt.title('Minimum Energy vs. Inverse Temperature')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# 2️⃣ Average Energy Plot
plt.subplot(1, 3, 2)
plt.plot(betas, average_energies, marker='s', linestyle='-', color='seagreen', label='Average Energy')
plt.xlabel(r'$\beta$')
plt.ylabel('Average Energy')
plt.title('Average Energy vs. Inverse Temperature')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# 3️⃣ Scaled Variance Plot
plt.subplot(1, 3, 3)
plt.plot(betas, scaled_variance_energies, marker='^', linestyle='-', color='darkorange', label='Scaled Variance Energy')
plt.xlabel(r'$\beta$')
plt.ylabel(r'Scaled Variance Energy')
plt.title(r'Scaled Variance Energy vs. Inverse Temperature')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

plt.suptitle('Energy Metrics vs. Inverse Temperature (Simulated Annealing)', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()



   






# Plotting the cities
#plot_cities(cities, box_size, radius_of_city)



