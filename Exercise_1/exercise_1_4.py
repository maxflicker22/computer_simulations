import numpy as np
import scipy.constants as consts
import matplotlib.pyplot as plt

# Define Function for sampling of g(x) with uniform variables
def generate_x_from_gx(uniform_rand_numbers):
    # How to get this Formular:
    # given g(x) = exp(-x)
    # Calc G(x) = exp(-x_min) - exp(-x)
    # G(x) = u # u = uniform distribution
    # x = G^-1(u)
    return -1 * np.log(uniform_rand_numbers)

# Frequentist Approach
def frequentist_approach(N, bin_width, counts):
    Ni_array = counts * N * bin_width
    sigma = np.sqrt(Ni_array * (1 - (Ni_array / N))) # Frequentist Approach
    return sigma / (N * bin_width) # Normalizing

# Rejection sampling function
def rejection_sampling(condition_func, gx_sampler, n_samples):
    samples = []
    while len(samples) < n_samples:
        # Sample von uniform
        u = np.random.uniform(0, 1)
        # Sample from g(x)
        x_sample = gx_sampler(np.random.uniform(0, 1))

        # Check condition
        if u <= condition_func(x_sample):  # Acceptance condition
            samples.append(x_sample)

    return np.array(samples)

# Sample from constant pdf -> uniform
def sample_x_for_const_gx(x):
    return np.random.uniform(0, 6)


def bayesian_pi_mean(N_i, N, n_b):
    """
    Computes the Bayesian posterior mean of p_i for bin i.

    Parameters:
    - N_i: count in bin i
    - N: total count (sum of all bins)
    - n_b: number of bins

    Returns:
    - posterior mean of p_i
    """
    return (N_i + 1) / (N + n_b + 1)


def bayesian_pi_second_moment(N_i, N, n_b):
    """
    Computes the second moment of p_i under the Bayesian posterior.

    Parameters:
    - N_i: count in bin i
    - N: total count (sum of all bins)
    - n_b: number of bins

    Returns:
    - second moment of p_i
    """
    pi_mean = bayesian_pi_mean(N_i, N, n_b)
    return ((N_i + 2) / (N + n_b + 2)) * pi_mean

def bayesian_pi_std(N_i, N, n_b, bin_widh):
    """
    Computes the standard deviation (sigma) of p_i under the Bayesian posterior.

    Parameters:
    - N_i: count in bin i
    - N: total count (sum of all bins)
    - n_b: number of bins

    Returns:
    - standard deviation (sigma) of p_i
    """
    mean = bayesian_pi_mean(N_i, N, n_b)
    second_moment = bayesian_pi_second_moment(N_i, N, n_b)
    variance = second_moment - mean**2
    return np.sqrt(variance) / (bin_width) # Normalizing

# Define functions g(x) and f(x)
gx = lambda x: np.exp(-x)  # Proposal distribution
fx = lambda x: np.exp(-x ** 2 / 2) * np.sqrt(2 / np.pi)  # Target distribution

# Compute c using a reasonable range of x values
x_vals = np.linspace(0, 10, 1000)
c = np.max(fx(x_vals) / gx(x_vals))  # Maximum ratio of f(x) / g(x)

# Define scaled proposal function
cgx = lambda x: c * gx(x)

# Condition function f / (c * g)
condition_func = lambda x: fx(x) / cgx(x)

# Alternative Condition Function
max_fx = np.max(fx(x_vals)) * 1
condition_func_alternaiv = lambda x: fx(x) / max_fx

# Initialize Variables
xmin = 0
xmax = 1
n = 10000
kB = consts.k
T = 300     # K
m = 1
bins = 20

# Define Sigma
sigma = np.sqrt(kB * T / m)
sigma = 1

def sample_3_independent_v(condition_func, generate_x_from_gx, n):
    # Sample 3 different Velocitys with Rejections
    v1 = rejection_sampling(condition_func, generate_x_from_gx, n)
    v2 = rejection_sampling(condition_func, generate_x_from_gx, n)
    v3 = rejection_sampling(condition_func, generate_x_from_gx, n)

    # Calculate magnitude of complete v
    return sigma * np.sqrt(v1 ** 2 + v2 ** 2 + v3 ** 2)

v = sample_3_independent_v(condition_func, generate_x_from_gx, n)
v_alternaiv = sample_3_independent_v(condition_func_alternaiv, sample_x_for_const_gx, n)

v_vals = np.linspace(0, 5, 500)
pv = np.sqrt(2 / np.pi) * v_vals ** 2 * np.exp(-(v_vals ** 2) / 2)

# Calculate Frequentist Approach for Histograms
counts, bin_edges = np.histogram(v, bins=bins, density=True)
counts_alternativ, bin_edges_alternativ = np.histogram(v_alternaiv, bins=bins, density=True)
# Bin-Mitte berechnen für die Fehlerbalken
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
bin_centers_alternativ = (bin_edges_alternativ[:-1] + bin_edges_alternativ[1:]) / 2

# Calculate Bin width
bin_width = bin_edges[1] - bin_edges[0]
bin_width_alternativ = bin_edges_alternativ[1] - bin_edges_alternativ[0]

# Errorbalken berechnen mittels Frequentist approach
error = frequentist_approach(n, bin_width, counts)
error_alternativ = frequentist_approach(n, bin_width_alternativ, counts_alternativ)

# Errorbalken mittels Baysian approach
error_baysian = bayesian_pi_std(counts * n * bin_width, n, bins, bin_width)

# Plot histogram of sampled velocities
plt.figure(figsize=(8, 6))
plt.hist(v, bins=bins, density=True, alpha=0.6, label="Sampled Velocity Distribution")
plt.plot(v_vals, pv, color='red', linewidth=2, label="Maxwell-Boltzmann Theoretical Curve")
# Fehlerbalken hinzufügen
plt.errorbar(bin_centers + bin_width / 8, counts, yerr=error, fmt='o', color='r', label="Error bars frequentist", alpha=0.7)
plt.errorbar(bin_centers - bin_width / 8, counts, yerr=error_baysian, fmt='x', color='b', label="Error bars baysian", alpha=0.7)

# Add labels and title
plt.xlabel("Velocity Magnitude (v)")
plt.ylabel("Probability Density")
plt.title("Velocity Distribution Compared to Maxwell-Boltzmann Theorem")
plt.legend()
plt.grid(True)

# Show the plot
plt.show()


# Plot histogram of sampled velocities
plt.figure(figsize=(8, 6))
plt.hist(v_alternaiv, bins=bins, density=True, alpha=0.6, label="Sampled Velocity Distribution")
plt.plot(v_vals, pv, color='red', linewidth=2, label="Maxwell-Boltzmann Theoretical Curve")
# Fehlerbalken hinzufügen
plt.errorbar(bin_centers_alternativ, counts_alternativ, yerr=error_alternativ, fmt='o', color='r', label="Error bars")

# Add labels and title
plt.xlabel("Velocity Magnitude (v)")
plt.ylabel("Probability Density")
plt.title("Velocity Distribution Compared to Maxwell-Boltzmann Theorem - Alternativ")
plt.legend()
plt.grid(True)

# Show the plot
plt.show()

def compare_errors_over_N(N_values, bins):
    """
    Compare Bayesian and Frequentist errors over different total sample sizes N.

    Parameters:
    - N_values: array-like, list of total sample sizes
    - bin_width: width of the histogram bins (from reference histogram)
    - counts_ref: reference normalized histogram (same for all N, scaled accordingly)
    - bins: number of bins
    """
    freq_errors = []
    bayes_errors = []
    sqrtN_errors = []

    for N in N_values:
        # Recalculate errors
        samples = sample_3_independent_v(condition_func, generate_x_from_gx, N)
        counts, bin_edges = np.histogram(samples, bins=bins, range=(0, 6), density=True)
        bin_width = bin_edges[1] - bin_edges[0]
        freq = frequentist_approach(N, bin_width, counts)
        bayes = bayesian_pi_std(counts * N * bin_width, N, bins, bin_width)
        freq_errors.append(np.mean(freq))
        bayes_errors.append(np.mean(bayes))
        sqrtN_errors.append(1 / np.sqrt(N))

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(N_values, freq_errors, label="Frequentist Error", marker='o')
    plt.plot(N_values, bayes_errors, label="Bayesian Error", marker='x')
    plt.plot(N_values, sqrtN_errors, label=r"$1/\sqrt{N}$", linestyle='--')
    plt.xlabel("Number of Samples (N)")
    plt.ylabel("Average Error")
    plt.title("Comparison of Errors vs N")
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.show()


# Evaluate Errors for different N
N_values = np.logspace(1, 4.7, 15, dtype=int)  # From 1e2 (100) to ~5e4
N_values = np.linspace(20, 3000, 300)  # From 1e2 (100) to ~5e4
compare_errors_over_N(N_values, bins)
