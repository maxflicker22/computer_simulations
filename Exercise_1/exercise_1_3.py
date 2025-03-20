from exercise_1_2 import generate_x_from_gx
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as consts

# Initialize Variables
xmin = 0
xmax = 1
n = 100000
kB = consts.k
T = 300     # K
m = 1
bins = 100

# Generate n uniform numbers
uniform_rand_numbers = np.random.uniform(xmin, xmax, n)

# Define Sigma
sigma = np.sqrt(kB * T / m)

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


# Rejection sampling function
def rejection_sampling(condition_func, gx_sampler, n_samples):
    samples = []
    for i in range(n_samples):
        # Sample einmal von u und einmal von gx
        # nicht von gx mit u sampln sonsg kommt
        # g(x) leicht modifiziert heraus
        # Sample von uniform
        u = np.random.uniform(0, 1)
        # Sample from g(x)
        x_sample = gx_sampler(np.random.uniform(0, 1))

        # Check condition
        if u <= condition_func(x_sample):  # Acceptance condition
            samples.append(x_sample)

    return np.array(samples)

# Run rejection sampling
X_prime = rejection_sampling(condition_func, generate_x_from_gx, n)
print("X_lengthg: ", n)
print("X_prime", X_prime)
print("X_prime_lengthg: ", len(X_prime))
print("Probability for Acceptanc 1/c = ", c)


# Plot histogram of accepted samples
plt.title("Histogram sampled from f(x) with g(x) = exp(-x)")
plt.hist(X_prime, bins=bins, density=True, alpha=0.6, label="Histogram f(x)")
plt.plot(x_vals, fx(x_vals))
plt.legend()
plt.show()


### c - Choose different g(x) ###

# choose for g(x) constant funktion
# X is now uniform with max Value of f(x)

# Condition function f / (max(f(x)) * 1)
max_fx = np.max(fx(x_vals))
max_fx = np.max(fx(x_vals)) * 1
condition_func = lambda x: fx(x) / max_fx

# Sample from constant pdf -> uniform
def sample_x_for_const_gx(x):
    return np.random.uniform(0, 6)

# Run rejection sampling
X_prime = rejection_sampling(condition_func, sample_x_for_const_gx, n)
print("X_lengthg: ", n)
print("X_prime", X_prime)
print("X_prime_lengthg: ", len(X_prime))
print("Probability for Acceptanc 1/c = ", max_fx)


# Plot histogram of accepted samples
plt.title("Histogram sampled from f(x) with c*g(x) = max(f(x))")
plt.hist(X_prime, bins=bins, density=True, alpha=0.6, label="Histogram f(x)")
plt.plot(x_vals, fx(x_vals))
plt.legend()
plt.show()

