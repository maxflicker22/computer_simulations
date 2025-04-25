#%%
import numpy as np
import matplotlib.pyplot as plt

# a) Generate 10x10 grid of unit Vector
unitvector = lambda phi : np.array([np.cos(phi), np.sin(phi)])

grid_range = 10
range_of_angels = [2*np.pi, 2*np.pi]
grid_of_unitvec = np.array([[unitvector(np.random.uniform(min(range_of_angels), max(range_of_angels))) for _ in range(grid_range)] for _ in range(grid_range)])

#print("grid_of_unitvector shape" , grid_of_unitvec.shape)
#print("grid_of_unitvector" , grid_of_unitvec)

rot_matrix = lambda theta: np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])

# Rotates one random unitvector of the grid around theeta
def rotated_one_random_unit_vec(theta):
    (i, j) = np.random.randint(0, 10, 2)
    grid_of_unitvec[i, j] = np.dot(rot_matrix(theta), grid_of_unitvec[i, j])


rotated_one_random_unit_vec(np.pi/2)

# Detailed Balance: Conndition ⎪ πi * pij = πj * pji
# Check if detail balanced # is reversable, is a form of symmetry
def check_detail_balance():
    theta = np.pi
    left_sight = grid_of_unitvec[0, 0, 0] * rot_matrix(theta)[0, 1]
    right_sight = grid_of_unitvec[0, 0, 1] * rot_matrix(theta)[1, 0]
    print("left_sight", left_sight)
    print("right_sight", right_sight)
    if np.array_equal(left_sight, right_sight):
        print("Yeah!! Detail Balance!!!")
    else:
        print("No reversibility")


check_detail_balance()

    



# %%
