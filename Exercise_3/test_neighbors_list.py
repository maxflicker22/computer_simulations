import numpy as np
from numba import njit, prange

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



def initialize_particals_on_surface(number_of_particles, radius_of_particales, box_width, box_height, max_attempts=50000):
    """
    Initialize particles on the surface.
    """
    particels_position_list = []
    attempts = 0
    packing_fraction = number_of_particles * np.pi * radius_of_particales**2 / (box_width * box_height)
   
    while len(particels_position_list) < number_of_particles and attempts < max_attempts:
        
        # Randomly generate a position
        x = np.random.uniform(0, box_width)
        y = np.random.uniform(0, box_height)
        new_position = np.array([x, y])
        # Check if the particle overlaps with existing particles
        if not is_overlapping(new_position, np.array(particels_position_list), radius_of_particales, box_width, box_height):
            particels_position_list.append(new_position)



        attempts += 1
    
    packing_fraction = len(particels_position_list) * np.pi * radius_of_particales**2 / (box_width * box_height)
    particals = np.array(particels_position_list)
    return particals, packing_fraction




def prepare_cell_lists(positions, box_w, box_h, rc):
    num_cells_x = int(box_w // rc)
    num_cells_y = int(box_h // rc)
    num_cells = num_cells_x * num_cells_y

    cell_ids_x = (positions[:, 0] // rc).astype(np.int32) % num_cells_x
    cell_ids_y = (positions[:, 1] // rc).astype(np.int32) % num_cells_y
    cell_ids = cell_ids_y * num_cells_x + cell_ids_x

    cell_indices = [[] for _ in range(num_cells)]
    for idx, cell_id in enumerate(cell_ids):
        cell_indices[cell_id].append(idx)

    # Convert lists to Numba-friendly arrays
    cell_indices = [np.array(cell, dtype=np.int32) for cell in cell_indices]

    # Precompute neighbor maps
    neighbor_offsets = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [-1, 1]], dtype=np.int32)
    neighbor_map = []

    for cx in range(num_cells_x):
        for cy in range(num_cells_y):
            neighbors = []
            for dx, dy in neighbor_offsets:
                nx = (cx + dx) % num_cells_x
                ny = (cy + dy) % num_cells_y
                neighbor_id = ny * num_cells_x + nx
                neighbors.append(neighbor_id)
            neighbor_map.append(np.array(neighbors, dtype=np.int32))

    return cell_indices, neighbor_map




@njit(parallel=True)
def compute_distances_with_cutoff_numba(positions, box_w, box_h, cell_indices, neighbor_map, rc):
    N = positions.shape[0]
    rc2 = rc * rc
    box_size = np.array([box_w, box_h])

    distance_matrix = np.zeros((N, N, 2))

    for current_cell in prange(len(neighbor_map)):
        particles_i = cell_indices[current_cell]
        n_i = len(particles_i)
        if n_i == 0:
            continue

        # 1️⃣ Abstände innerhalb der aktuellen Zelle berechnen (ohne doppelte Berechnungen)
        for idx1 in range(n_i):
            i = particles_i[idx1]
            for idx2 in range(idx1 + 1, n_i):  # idx2 > idx1 → keine Doppelberechnung
                j = particles_i[idx2]

                rij = positions[j] - positions[i]
                rij -= np.round(rij / box_size) * box_size
                dist2 = rij[0]**2 + rij[1]**2

                if dist2 < rc2:
                    distance_matrix[i, j] = rij

        # 2️⃣ Abstände zu Nachbarzellen berechnen
        for neighbor_cell in neighbor_map[current_cell]:
            if neighbor_cell == current_cell:
                continue  # Wurde oben schon behandelt

            particles_j = cell_indices[neighbor_cell]
            if len(particles_j) == 0:
                continue

            for i in particles_i:
                for j in particles_j:
                    rij = positions[j] - positions[i]
                    rij -= np.round(rij / box_size) * box_size
                    dist2 = rij[0]**2 + rij[1]**2

                    if dist2 < rc2:
                        distance_matrix[i, j] = rij

    return distance_matrix



num_particals = 5000
box_h = 10.
box_w = 10.
rc = 2.5



particals_pos, _ = initialize_particals_on_surface(num_particals, 1., box_w, box_h)
cell_list, neighbor_cells = prepare_cell_lists(particals_pos, box_w, box_h, rc)
delta = compute_distances_with_cutoff_numba(particals_pos, box_w, box_h, cell_list, neighbor_cells, rc)
print("cell_list", cell_list)
print("neighbors_cell", neighbor_cells)
print("delta", delta)