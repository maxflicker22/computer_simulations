import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.constants as const

### Exercise 1 - Density and velocity distribution of an ideal gas ###

# Volume
V = 1

## 1. Density fluctuations ##

# a) & b)
# Generate uniformly distributed random numbers
n = 10
x_min = 0
x_max = 1
bins = 100


def plot_histogram_with_errors(x_min, x_max, n, bins):
    #Generiere unifrom n random variables
    uniform_ran_numbers = np.random.uniform(x_min, x_max, n)

    # Histogramm mit plt.hist()
    # PDF = integral über alle Werte muss eins sein, kleine bin width bedeutet hohe y werte

    counts, bin_edges, _ = plt.hist(uniform_ran_numbers, bins=bins, density=True, alpha=0.7, color='b', edgecolor='k', label="Histogram (PDF)")

    # Bin-Mitte berechnen für die Fehlerbalken
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]

    # Calculate Error bar for each bin
    # count in Histogramm folgt Poisson daher Erwartungswert = Varianz
    # Daher fehler = sqrt(Erwatungswert)
    sigma = np.sqrt(counts * n * bin_width) / (n * bin_width)

    #Theoretischer Wert von pdf = 1 (Uniform
    theo_pdf = 1

    # Check ob 68% der Balken die theoretische 1 Linie erreichen (Im Sigma intervall liegen)
    bin_cross_one = np.sum((counts - sigma <= theo_pdf) & (theo_pdf <= counts + sigma))
    percent_bin_cross_one = (bin_cross_one / bins) * 100
    print(f"Bei N = {n};  Prozent an Bins die im 68% intervall liegen: ", percent_bin_cross_one)

    # Fehlerbalken hinzufügen
    plt.errorbar(bin_centers, counts, yerr=sigma, fmt='o', color='r', label="Error bars")

    # Theoretische Dichte für die Gleichverteilung
    plt.axhline(1, color='g', linestyle='dashed', label="Theoretical PDF (Uniform)")

    # Achsenbeschriftung und Titel
    plt.xlabel("Value")
    plt.ylabel("Probability Density")
    plt.title("Histogram with Error Bars using plt.hist()")
    plt.legend()
    plt.show()
    return



#plot_histogram_with_errors(x_min, x_max, 10, bins)
#plot_histogram_with_errors(x_min, x_max, 100, bins)
plot_histogram_with_errors(x_min, x_max, 1000, bins)
plot_histogram_with_errors(x_min, x_max, 100000, bins)


# c) & d)
#Examine the fluctuations of the height of an individual bar



def examine_fluctuations_of_heigt(N):
    M = 1000
    bins = 500
    x_min = 0
    x_max = 1

    uni_ran_num_list = []
    counts_list = []


    for j in range(M):
        uniform_random_numbers = np.random.uniform(x_min, x_max, N)

        counts, bin_edges = np.histogram(uniform_random_numbers, bins=bins, density=True)

        uni_ran_num_list.append(uniform_random_numbers)
        counts_list.append(counts)

    # Berechne Bin Weite
    bin_width = bin_edges[1] - bin_edges[0]

    # Convert lists to numpy arrays
    uni_ran_num_array = np.array(uni_ran_num_list)  # Shape: (M, N)
    counts_array = np.array(counts_list)  # Shape: (M, bins)
    counts_not_normalized_array = counts_array * N * bin_width


    # Select bin index
    i = 6  # 6th Bin (corresponding to 0.5 < x < 0.6)

    # Compute expectation value and variance for the bin heights across M samples
    expectation_value_bin_i = np.mean(counts_array[:, i])
    variance_bin_i = np.var(counts_array[:, i])
    std_bin_i = np.sqrt(variance_bin_i)  # Standard deviation

    # Generate histogram of bin heights
    plt.hist(counts_array[:, i], bins=bins, density=True, alpha=0.7, color='g', edgecolor='k', label="Histogram (PDF)")

    # Generate Normal distribution (Gaussian curve) for overlay
    x_values = np.linspace(min(counts_array[:, i]), max(counts_array[:, i]), 100)
    normal_dist = stats.norm.pdf(x_values, expectation_value_bin_i, std_bin_i)

    # Plot normal distribution curve
    plt.plot(x_values, normal_dist, 'r-', label="Normal Distribution")

    # Plot Erwartungswert der Höhe
    plt.axvline(expectation_value_bin_i, color='r', linestyle='dashed', label="Expectations Value over M Samples")


    # Achsenbeschriftung und Titel
    plt.xlabel(f"Height Value of Bin {i}")
    plt.ylabel("Probability Density")
    plt.xlim([0.8, 1.2])
    plt.title(f"Histogram over M with N = {N}")
    plt.legend()
    plt.show()


    # d)
    # Calculate isothermal compressibility
    V = 1
    kB = const.k # Boltzmann k
    T = 300 # Kelvin
    Vi = V / bins
    roh = N/V

    # Caluclate Varianc and Exception Value of Ni
    mean_Ni = np.mean(counts_not_normalized_array[:, i])
    variance_Ni = np.var(counts_not_normalized_array[:, i])

    # Isothermal compressibility approximation
    kappa_T = (Vi / (kB * T)) * (variance_Ni / mean_Ni ** 2)

    # Isothermal from ideal Gas
    kappa_T_ideal = 1 / (roh * kB * T)

    # Print results
    print(f"Approximated isothermal compressibility: {kappa_T:.6e} (1/Pa)")
    print(f"Ideal gas isothermal compressibility: {kappa_T_ideal:.6e} (1/Pa)")

    # Compare results
    difference = abs(kappa_T - kappa_T_ideal)
    print(f"Difference: {difference:.6e} (1/Pa)")



examine_fluctuations_of_heigt(1000)
examine_fluctuations_of_heigt(100000)




