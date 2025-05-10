import numpy as np
import matplotlib.pyplot as plt

def is_overlapping(new_position, existing_positions, radius_of_particales, box_width, box_height):
    """
    Check if the particle overlaps with existing particles.
    """
    if len(existing_positions) == 0:
        return False  # No particles yet, no overlap
    box_size = np.array([box_width, box_height])
    raw_distances = existing_positions - new_position
    raw_distances = raw_distances + box_size / 2 # shift the distance
    raw_distances = raw_distances % box_size - box_size / 2 # find shortest distance
    distances = np.linalg.norm(raw_distances, axis=1) # distandces with periodic boundaries
    return np.any(distances < 2 * radius_of_particales)

    


def initialize_particals_on_surface(number_of_particles, radius_of_particales, box_width, box_height, max_attempts=5000):
    """
    Initialize particles on the surface.
    """
    particels_list = []
    attempts = 0
    packing_fraction = number_of_particles * np.pi * radius_of_particales**2 / (box_width * box_height)
   
    while len(particels_list) < number_of_particles and attempts < max_attempts:
        
        # Randomly generate a position
        x = np.random.uniform(0, box_width)
        y = np.random.uniform(0, box_height)
        new_position = np.array([x, y])
        # Check if the particle overlaps with existing particles
        if not is_overlapping(new_position, np.array(particels_list), radius_of_particales, box_width, box_height):
            particels_list.append(new_position)

        attempts += 1

    return np.array(particels_list)


# Simulation Parameters
num_particles = 50
radius = 1.0
box_w, box_h = 10, 10

particles = initialize_particals_on_surface(num_particles, radius, box_w, box_h)

# Plotting
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(0, box_w)
ax.set_ylim(0, box_h)
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_title('Particles on Surface')
ax.set_aspect('equal', adjustable='box')
ax.grid(True)

# Draw Particles and Circles
for pos in particles:
    circle = plt.Circle(pos, radius, color='lightblue', fill=False, linestyle='--')
    ax.add_patch(circle)
    ax.plot(pos[0], pos[1], 'o', color='blue', markersize=5)

plt.show()

