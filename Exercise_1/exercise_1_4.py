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
n = 100000
kB = consts.k
T = 300     # K
m = 1
bins = 100

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

# Plot histogram of sampled velocities
plt.figure(figsize=(8, 6))
plt.hist(v, bins=bins, density=True, alpha=0.6, label="Sampled Velocity Distribution")
plt.plot(v_vals, pv, color='red', linewidth=2, label="Maxwell-Boltzmann Theoretical Curve")

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

# Add labels and title
plt.xlabel("Velocity Magnitude (v)")
plt.ylabel("Probability Density")
plt.title("Velocity Distribution Compared to Maxwell-Boltzmann Theorem - Alternativ")
plt.legend()
plt.grid(True)

# Show the plot
plt.show()

