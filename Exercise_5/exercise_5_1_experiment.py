import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import os

#--------------------------------- Plot Functions ---------------------------------

def plot_acceptance_rate(ks, acceptance_rate, alternative_pacc, q=None, steps_per_temperature=None):
    """
    Plot the acceptance rate against the k values.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(ks, acceptance_rate, marker='o', linestyle='-', color='purple')
    plt.xlabel('k values')
    plt.ylabel('Acceptance Rate')
    
    # Add details to the title
    pacc_label = "alternative" if alternative_pacc else "standard"
    title = f'Acceptance Rate vs. k values\n(pacc: {pacc_label}, q: {q}, L: {steps_per_temperature})'
    plt.title(title)
    
    plt.grid(True)
    plt.tight_layout()
    
    filename = f"q:{q}_L:{steps_per_temperature}_pacc:{pacc_label}_acceptance_rate_vs_k.png"
    plt.savefig(filename)
    #plt.show()


def plot_energie_datas(beta_k, min_energies, average_energies, scaled_variance_energies, Energies_list, alternative_pacc, q=None, steps_per_temperature=None):
    # Convert beta_k and energy lists to numpy arrays for consistent plotting
    betas = np.array(beta_k)
    min_energies = np.array(min_energies)
    average_energies = np.array(average_energies)
    scaled_variance_energies = np.array(scaled_variance_energies)

    all_energies = []
    all_steps = []

    step_counter = 0
    for beta in betas:
        energies_at_beta = Energies_list[beta]
        all_energies.extend(energies_at_beta)
        all_steps.extend(range(step_counter, step_counter + len(energies_at_beta)))
        step_counter += len(energies_at_beta)

    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 3)

    # Subplots here remain unchanged
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

    # Add details to the title
    pacc_label = "alternative" if alternative_pacc else "standard"
    super_title = f'Energy Metrics vs. Inverse Temperature (Simulated Annealing)\n(pacc: {pacc_label}, q: {q}, L: {steps_per_temperature})'
    plt.suptitle(super_title, fontsize=16)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    filename = f"q:{q}_L:{steps_per_temperature}_pacc:{pacc_label}_energy_metrics_vs_inverse_temperature.png"
    plt.savefig(filename)
    #plt.show()


#--------------------------------- Programm Functions ---------------------------------

def create_initial_city_positions(num_cities, box_size, seed):
    """
    Create initial city positions within a square box.
    """
    np.random.seed(seed)  # Set the random seed for reproducibility
    cities = np.random.rand(num_cities, 2) * box_size
    return cities , np.arange(num_cities)


def plot_cities(cities, box_size, radius_of_city, best_path_flag=False, alternative_pacc=False, q=None, steps_per_temperature=None):
    path_distance = calculate_distance_of_Salesman(cities)
    plt.figure(figsize=(8, 8))
    plt.xlim(0, box_size)
    plt.ylim(0, box_size)
    pacc_label = "alternative" if alternative_pacc else "standard"

    if best_path_flag:
        plt.plot(cities[:, 0], cities[:, 1], c='blue', alpha=0.5, label='Path')
        plt.scatter(cities[:, 0], cities[:, 1], s=radius_of_city*100, c='blue', alpha=0.5, label='Cities')
        title = (f'City Positions with Path (Total Distance: {path_distance:.2f})\n'
                 f'(pacc: {pacc_label}, q: {q}, L: {steps_per_temperature})')
        filename = f"q:{q}_L:{steps_per_temperature}_pacc:{pacc_label}_city_positions_best_path.png"
    else:
        plt.scatter(cities[:, 0], cities[:, 1], s=radius_of_city*100, c='blue', alpha=0.5, label='Cities')
        title = f'City Positions\n(pacc: {pacc_label}, q: {q}, L: {steps_per_temperature})'
        filename = f"q:{q}_L:{steps_per_temperature}_pacc:{pacc_label}_city_positions_initial_path.png"

    plt.title(title)
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    #plt.show()


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

def acceptance_probability(old_distance, new_distance, beta, alternative_pacc):
    """
    Calculate the acceptance probability for the new configuration.
    """
    if alternative_pacc:
        # Alternative acceptance probability calculation
        return 1 / (1 + np.exp(beta * (new_distance - old_distance)))
    else:
        if new_distance < old_distance:
            return 1.0
        else:   
            # Standard acceptance probability calculation
            return np.exp((old_distance - new_distance) * beta)

def try_update_order(cities, old_distance, order, beta, alternative_pacc):
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
    return beta * np.power(k, q, dtype=np.float64)

def update_beta_specific_heat(beta_k, energies, delta=0.1):
    """
    Compute the next beta using specific heat-based cooling.

    """
    E = np.array(energies)
    E_mean = np.mean(E)
    E_std = np.std(E)

    if E_mean == 0 or delta == 0:
        raise ValueError("E_mean or delta is zero, cannot compute new beta.")

    r = 1 - (delta / (beta_k * E_std))
    if r <= 0:
        r = 1e-3  # avoid division by zero or negative r

    beta_k1 = beta_k / r

    return beta_k1


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

def run_simulated_annealing(q, beta_start, ks, steps_per_temperature, num_cities, box_size, radius_of_city, threshold, threshold_acceptanc_rate, seed, alternative_pacc):
    
    delta = 0.1  # Delta for specific heat-based cooling
    
    final_steps_to_convergence = 0  # Variable to track the final steps to convergence
    final_path_distance = 0.  # Variable to track the best path distance found

    # Initialize the list to store energies for each beta value
    beta_k = []  # List to store beta values
    Energies_list = {}


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
            beta_k.append(update_beta_specific_heat(beta_k[-1], Energies_list.get(beta_k[-1], []), delta=delta))
            #beta_k.append(update_beta(beta_k[-1], k, q))

        # Initialize the energies list for the current beta
        Energies_list[beta_k[-1]] = []
        Energies_list[beta_k[-1]].append(old_disance)
        #print("Energy_list for beta =", Energies_list[beta_k[-1]])


        # Loop over steps for the current beta
        old_distance = Energies_list[beta_k[-1]][-1].copy()

        for step in range(steps_per_temperature):
            # Try Update the Current Order
            order, current_distance = try_update_order(cities, old_distance, order, beta_k[-1], alternative_pacc=alternative_pacc)

            if current_distance != old_distance:
                Energies_list[beta_k[-1]].append(current_distance)
                old_distance = current_distance.copy()  # <== Hier wird die Referenzdistanz aktualisiert!

            #print("Current Step:", step + 1, "/", steps_per_temperature)

            # Check for convergence
            if convergence_check(Energies_list[beta_k[-1]], threshold, step + 1, threshold_acceptanc_rate):
            #if acceptance_rate_check(len(Energies_list[beta_k[-1]]), step + 1, threshold_acceptanc_rate): 
                convergence_reached = True
                print("convergence_reached:", convergence_reached)
                break
         
        #print("Acceepted Energies:", accepted_energies)

        final_steps_to_convergence = step + 1
        final_path_distance = Energies_list[beta_k[-1]][-1]
        
        if convergence_reached:
            #print("convergence_reached status = ", convergence_reached)
            break


        if k == ks[-1]:
            print("Final order of cities:", order)
            print("Final distance:", Energies_list[beta_k[-1]][-1])
            print("Final beta value:", beta_k[-1])
            print("Nothing converged anymore, stopping the simulation.")


    #--------------------------------- Plot Execution---------------------------------

    # Plotting the cities
    #plot_cities(cities, box_size, radius_of_city, best_path_flag=False, alternative_pacc=alternative_pacc, q=q, steps_per_temperature=steps_per_temperature)

    # Plotting the cities
    #plot_cities(cities[order], box_size, radius_of_city, best_path_flag=True, alternative_pacc=alternative_pacc, q=q, steps_per_temperature=steps_per_temperature)

    ## Reults d
    # Notes on parameter effects:
    # q: Higher q means faster cooling → faster convergence but higher risk of local minima.
    #    Lower q means slower cooling → slower but better exploration, higher chance of finding the global minimum.
    # L: More steps per temp (higher L) → better sampling, more stable solutions but slower runtime.
    #    Lower L → faster runtime but risk of skipping good solutions.
    # pacc: Standard (exp) → fast convergence, risk of getting stuck.
    #       Logistic (1/(1+exp)) → smoother exploration, better at avoiding local minima but may converge slower.
    #                            -> jumps easier from local minima because probability distribution is smoother

    
    parameters = {
        "beta_start": beta_start,
        "alternative_pacc": alternative_pacc,
        "beta_k": beta_k[-1],
        "q": q,
        "steps_per_temperature": steps_per_temperature,
        "threshold": threshold,
        "threshold_acceptanc_rate": threshold_acceptanc_rate,


    }
    results = {
        "final_steps_to_convergence": final_steps_to_convergence,
        "final_path_distance": final_path_distance,
        "final_beta_value": beta_k[-1],
    }

    return parameters, results



if __name__ == "__main__":
    # Define parameters 

    #### Fixed Parameters von hier
    box_size = 1.0  # Size of the box
    num_cities = 50  # Number of cities
    radius_of_city = 1.0  # Radius of each city
    seed = 42  # Random seed for reproducibility
    beta_start = 1 # Inverse temperature parameter
    #### bis hier

    # Initialize variables
    ks = np.arange(1, 80)  # Range of k values for beta update
    threshold = 5e-5  # Convergence threshold
    threshold_acceptanc_rate = 0.1 # Acceptance rate threshold for convergence

    q = [0.4, 0.6, 0.8, 1.05, 1.15, 1.35, 1.5]# Cooling rate
    steps_per_temperature = [x * (num_cities **2) for x in [1, 2, 3, 4, 5]]  # Steps per temperature
    alternative_pacc = [False, True]  # Use alternative acceptance probability calculation

    print("steps_per_temperature:", steps_per_temperature)

    # Output file
    filename_csv = "specific_heat_simulated_annealing_results.csv"
    file_exists = os.path.exists(filename_csv)

    # Run the simulated annealing algorithm
    for q_, steps_, pacc in itertools.product(q, steps_per_temperature, alternative_pacc): 
        
        parameters, results = run_simulated_annealing(q_, beta_start, ks, steps_, num_cities, box_size, radius_of_city, threshold, threshold_acceptanc_rate, seed, pacc)
        # Print the parameters and results
        print("Parameters:", parameters)
        print("Results:", results)
        # Save parameters and results to a CSV file
        # Merge both dictionaries into one flat dictionary (single row)
        row = {**parameters, **results}
        df = pd.DataFrame([row])
        df.to_csv(filename_csv, mode='a', header=not file_exists, index=False)
        file_exists = True
        print("Results saved to simulated_annealing_results.csv")
        # Save the beta_k values to a CSV file


    



