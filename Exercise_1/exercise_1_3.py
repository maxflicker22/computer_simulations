from exercise_1_2 import generate_x_from_gx
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as consts

# Initialize Variables
xmin = 0
xmax = 1
n = 10000
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

# Generate X from g(x) - Exercise 2
X = generate_x_from_gx(uniform_rand_numbers)

# Rejection sampling function
def rejection_sampling(u, condition_func, x, n_samples):
    samples = []
    for i in range(n_samples):
        u = np.random.uniform(0, 1)
        if u <= condition_func(x[i]):  # Acceptance condition
            samples.append(x[i])

    return np.array(samples)

# Run rejection sampling
X_prime = rejection_sampling(uniform_rand_numbers, condition_func, X, n)
print("X", X)
print("X_lengthg: ", len(X))
print("X_prime", X_prime)
print("X_prime_lengthg: ", len(X_prime))

# Plot functions
plt.title("Involved Functions")
plt.scatter(X, fx(X), label="f(x)")
plt.scatter(X, gx(X), label="g(x)")
plt.scatter(X, cgx(X), label="c * g(x)")
plt.scatter(X, fx(X) / cgx(X), label="f(x) / (c * g(x))")
plt.legend()
plt.show()

# Plot histogram of accepted samples
plt.title("Histogram sampled from f(x)")
plt.hist(X_prime, bins=bins, density=True, alpha=0.6, label="Histogram f(x)")
plt.plot(x_vals, fx(x_vals))
plt.legend()
plt.show()
