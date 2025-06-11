import numpy as np
import matplotlib.pyplot as plt

#--------------------------------- Plot Functions ---------------------------------

def plot_acceptance_rate(ks, acceptance_rate, alternative_pacc=False):
    """
    Plot the acceptance rate against the k values.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(ks, acceptance_rate, marker='o', linestyle='-', color='purple')
    plt.xlabel('k values')
    plt.ylabel('Acceptance Rate')
    plt.title('Acceptance Rate vs. k values')
    plt.grid(True)
    plt.tight_layout()
    filename = 'acceptance_rate_vs_k.png'
    if alternative_pacc:
        pacc = "alternative"
    else:
        pacc = "standard"
    filename = f"q:{q}_L:{steps_per_temperature}_pacc:{pacc}" + filename
    plt.savefig(filename)  # Save the figure as an image file
    plt.show()

def plot_energie_datas(beta_k, min_energies, average_energies, scaled_variance_energies, Energies_list, alternative_pacc=False):
    # Convert beta_k and energy lists to numpy arrays for consistent plotting
    betas = np.array(beta_k)  # Skip the first beta (beta_start)
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
    filename = 'energy_metrics_vs_inverse_temperature.png'

    if alternative_pacc:
        pacc = "alternative"
    else:
        pacc = "standard"
    filename = f"q:{q}_L:{steps_per_temperature}_pacc:{pacc}" + filename

    plt.savefig(filename)  # Save the figure as an image file
    plt.show()

#--------------------------------- Programm Functions ---------------------------------

def create_initial_city_positions(num_cities, box_size, seed):
    """
    Create initial city positions within a square box.
    """
    np.random.seed(seed)  # Set the random seed for reproducibility
    cities = np.random.rand(num_cities, 2) * box_size
    return cities , np.arange(num_cities)


def plot_cities(cities, box_size, radius_of_city, best_path_flag=False, alternative_pacc=False):
    """
    Plot the cities on a 2D plane.
    """
    path_distance = calculate_distance_of_Salesman(cities)
    plt.figure(figsize=(8, 8))
    plt.xlim(0, box_size)
    plt.ylim(0, box_size)
    filename = 'city_positions.png'
    if best_path_flag:
        plt.plot(cities[:, 0], cities[:, 1], c='blue', alpha=0.5, label='Path')
        plt.scatter(cities[:, 0], cities[:, 1], s=radius_of_city*100, c='blue', alpha=0.5, label='Cities')
        plt.title('City Positions with Path\nTotal Distance: {:.2f}'.format(path_distance))
        filename = 'city_positions_best_path.png'
    else:
        plt.scatter(cities[:, 0], cities[:, 1], s=radius_of_city*100, c='blue', alpha=0.5, label='Cities')
        plt.title('City Positions')
        filename = 'city_positions_initial_path.png'

    if alternative_pacc:
        pacc = "alternative"
    else:
        pacc = "standard"
    filename = f"q:{q}_L:{steps_per_temperature}_pacc:{pacc}" + filename
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)  # Save the figure as an image file^
    plt.show()
    #print("Initial city positions created successfully.")
    #print("Number of cities created:", len(cities))
    #print("City positions:\n", cities)

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

def propose_new_city_configuration(order):
    """
    Propose a new city configuration.
    """
    num_cities = len(order)
    first_index = np.random.randint(1, num_cities - 2)
    second_index = np.random.randint(first_index + 1, num_cities)
    new_order = order.copy()
    new_order[first_index], new_order[second_index] = new_order[second_index], new_order[first_index]
    #print(f"Proposed new order: {new_order}")
    return new_order

def acceptance_probability(old_distance, new_distance, beta, alternative_pacc=False):
    """
    Calculate the acceptance probability for the new configuration.
    """
    if new_distance < old_distance:
        return 1.0
    else:
        if alternative_pacc:
            # Alternative acceptance probability calculation
            return 1 / (1 + np.exp(beta * (new_distance - old_distance)))
        else:
            # Standard acceptance probability calculation
            #print(f"Old distance: {old_distance}, New distance: {new_distance}, Beta: {beta}")
            #print(f"Acceptance probability: {np.exp((old_distance - new_distance) * beta)}")
            return np.exp((old_distance - new_distance) * beta)

def try_update_order(cities, old_distance, order, beta):
    """
    Try to update the order of cities based on the Metropolis criterion.
    """

    new_order = propose_new_city_configuration(order)
    new_distance = calculate_distance_of_Salesman(cities[new_order])
    
    prob = acceptance_probability(old_distance, new_distance, beta, alternative_pacc=alternative_pacc)
    
    if np.random.rand() < prob:
        #print("Accepted new configuration.")
        return new_order, new_distance
    else:
        #print("Rejected new configuration.")
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
    #print(f"Energies at beta = {beta}:")
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

def convergence_check(energies, threshold, step, threshold_acceptance_rate):
    """
    Check if the energies have converged.
    Criteria:
    At least 10 energy differences exist.
    Each difference is below the threshold.
    Each previous difference is >= the following one (monotonically decreasing).
    Acceptance rate is below the threshold_acceptance_rate.
    """
    windowsize = 2  # Number of differences to check
    if len(energies) < windowsize:  # Need at least 11 energies to get 10 differences
        return False
    if step < windowsize:
        return False

    # Calculate the last 10 differences
    diffs = np.abs(np.diff(energies[-windowsize:]))  # this gives 10 differences

    # Check if all differences are below threshold
    if not np.all(diffs < threshold):
        return False

    # Check if differences are monotonically decreasing
    if not np.all(diffs[:-1] >= diffs[1:]):
        return False

    # Calculate acceptance rate
    rate = len(energies) / step
    if rate >= threshold_acceptance_rate:
        return False

    print("Convergence check: Energies have converged.")
    print("Last differences:", diffs)
    print("Acceptance rate:", rate)

    return True


    

def acceptance_rate_check(accepted_moves, total_moves, threshold):
    """
    Check if the acceptance rate is below a threshold.
    """
    rate = accepted_moves / total_moves
    return rate < threshold
    

#--------------------------------- Programm---------------------------------


# Define parameters 

#### Fixed Parameters von hier
box_size = 1.0  # Size of the box
num_cities = 50  # Number of cities
radius_of_city = 1.0  # Radius of each city
seed = 42  # Random seed for reproducibility
beta_start = 1 # Inverse temperature parameter
#### bis hier

# Initialize variables
beta_k = []
ks = np.arange(1, 100)  # Range of k values for beta update
q = 1.0 # Cooling rate
steps_per_temperature = 1 * (num_cities ** 2)
threshold = 1e-4  # Convergence threshold
threshold_acceptanc_rate = 0.1#

alternative_pacc = False  # Use alternative acceptance probability calculation

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

# Delete random seed to avoid confusion
np.random.seed(1234)

# Conveergence flag
convergence_reached = False

# Start loop over cooling steps
for k in ks:

    if k == 1:
        beta_k.append(update_beta(beta_start, k, q))
    else:
        beta_k.append(update_beta(beta_k[-1], k, q))

    # Initialize the energies list for the current beta
    Energies_list[beta_k[-1]] = []
    Energies_list[beta_k[-1]].append(old_disance)
    #print("Energy_list for beta =", Energies_list[beta_k[-1]])

    accepted_energies = 0

    # Loop over steps for the current beta
    old_distance = Energies_list[beta_k[-1]][-1].copy()

    for step in range(steps_per_temperature):

        # Try Update the Current Order
        order, current_distance = try_update_order(cities, old_distance, order, beta_k[-1])

        if current_distance != old_distance:
            Energies_list[beta_k[-1]].append(current_distance)
            old_distance = current_distance.copy()  # <== Hier wird die Referenzdistanz aktualisiert!

        #print("Current Step:", step + 1, "/", steps_per_temperature)

        # Check for convergence
        if convergence_check(Energies_list[beta_k[-1]], threshold, step + 1, threshold_acceptanc_rate):
        #if acceptance_rate_check(len(Energies_list[beta_k[-1]]), step + 1, threshold_acceptanc_rate):   
            print("Convergence reached at step", step + 1)
            print("Final order of cities:", order)
            print("Final distance:", Energies_list[beta_k[-1]][-1])
            accepted_energies = len(Energies_list[beta_k[-1]])
            convergence_reached = True
            print("convergence_reached:", convergence_reached)
            break
    
    # Evaluation of Accepted Energies
    accepted_energies = len(Energies_list[beta_k[-1]])
    acceptance_rate.append(accepted_energies / steps_per_temperature)       
    #print("Acceepted Energies:", accepted_energies)

    # Discuss energies at the current beta value
    min_energy, average_energy, scaled_variance_energy = discuss_energies(Energies_list[beta_k[-1]], beta_k[-1])
    min_energies.append(min_energy)
    average_energies.append(average_energy)
    scaled_variance_energies.append(scaled_variance_energy)

    if convergence_reached:
        print("convergence_reached status = :", convergence_reached)
        break


    if k == ks[-1]:
        print("Final order of cities:", order)
        print("Final distance:", Energies_list[beta_k[-1]][-1])
        print("Final beta value:", beta_k[-1])
        print("Acceptance rate:", acceptance_rate[-1])
        print("Nothing converged anymore, stopping the simulation.")


#--------------------------------- Plot Execution---------------------------------

# Plotting the cities
plot_cities(cities, box_size, radius_of_city, best_path_flag=True, alternative_pacc=alternative_pacc)

# Plot the energy data
plot_energie_datas(beta_k, min_energies, average_energies, scaled_variance_energies, Energies_list, alternative_pacc=alternative_pacc)

# Plot the acceptance rate
plot_acceptance_rate(ks[:len(acceptance_rate)], acceptance_rate, alternative_pacc=alternative_pacc)

# Plotting the cities
plot_cities(cities[order], box_size, radius_of_city, best_path_flag=True, alternative_pacc=alternative_pacc)




