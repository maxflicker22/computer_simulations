# Alles schön geordnet rdf > 1# Simulation Parameters
num_particles = 70
radius = 1.
box_w, box_h = 100.0, 100.0
initial_T = 1.
m = 1.0
kb = 1.0  # Boltzmann constant
sigma = 1.0 # length scsale
epsilon = 1.0  # eneregy scale
delta_t = 0.0005 * np.sqrt(m * sigma ** 2 / epsilon) # time step
rc = box_w   # Cutoff radius for neighbor search
rc_skin = 0. * rc
rc_de = rc + rc_skin
delta_thermostat = 0.015
num_steps = 10000
time_total = np.arange(num_steps) * delta_t
thermo_interval = 2