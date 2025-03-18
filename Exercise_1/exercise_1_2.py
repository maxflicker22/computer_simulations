import numpy as np
import matplotlib.pyplot as plt


### 2 Inverse Transformation Sampling ###

# Initialize Variables
x_min = 0
x_max = 1
n = 100000

# Generate n uniform numbers
uniform_rand_numbers = np.random.uniform(x_min, x_max, n)

# Define Function for sampling of g(x) with uniform variables
def generate_x_from_gx(uniform_rand_numbers):
    # How to get this Formular:
    # given g(x) = exp(-x)
    # Calc G(x) = exp(-x_min) - exp(-x)
    # G(x) = u # u = uniform distribution
    # x = G^-1(u)
    return -1 * np.log(uniform_rand_numbers)


# Compare Histogram with g(x)
x_compare = np.linspace(0, 10, 10000)
gx_fit = np.exp(-x_compare)

# Plot Histogram and g(x)
plt.title("Inverse Transformation Sampling")
plt.hist(generate_x_from_gx(uniform_rand_numbers), bins=100, density=True, label="Histogram of generated x")
plt.plot(x_compare, gx_fit, label="g(x)")
plt.ylabel("PDF g(x)")
plt.xlabel("Random Variable x")
plt.legend()
plt.show()





