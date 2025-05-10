import numpy as np
import matplotlib.pyplot as plt

# Define constants
epsilon = 1. # depth of potential well
sigma = 1. # finite distance at which potential is zero
rc = 2.5  # cutoff for WF potential

# Distance range
r = np.linspace(0., 3.0, 500)

# Lennard-Jones potential
def lennard_jones(r, epsilon, sigma):
    return 4 * epsilon * ((sigma / r)**12 - (sigma / r)**6)

# Simplified Wang-Frenkel potential
def wang_frenkel(r, epsilon, sigma, rc):
    wf = np.zeros_like(r)
    mask = r < rc
    wf[mask] = epsilon * ((sigma / r[mask])**2 - 1) * (((rc / r[mask])**2 - 1)) ** 2
    return wf

# Compute potentials
U_LJ = lennard_jones(r, epsilon, sigma)
U_WF = wang_frenkel(r, epsilon, sigma, rc)

# Plotting
plt.figure(figsize=(8, 6))
#plt.plot(r, U_LJ, label='Lennard-Jones', linewidth=2)
plt.plot(r, U_WF, label='Wang-Frenkel (simplified)', linewidth=2, linestyle='--')
plt.axvline(rc, color='gray', linestyle=':', label='Cutoff $r_c$')
plt.xlabel('Inter-particle distance $r$')
plt.ylabel('Potential Energy $U(r)$')
plt.title('Comparison of Lennard-Jones and Wang-Frenkel Potentials')
plt.legend()
plt.grid(True)
plt.ylim(-1.5, 5)
plt.tight_layout()
plt.show()
