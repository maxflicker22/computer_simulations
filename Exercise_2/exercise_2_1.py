import numpy as np
import matplotlib.pyplot as plt

# a) Generate 10x10 grid of unit Vector
unitvector = lambda phi : np.array([np.cos(phi), np.sin(phi)])

grid_range = 10
range_of_angels = [0, 2*np.pi]
grid_of_unitvec = np.array([[unitvector(np.random.uniform(min(range_of_angels), max(range_of_angels))) for _ in range(grid_range)] for _ in range(grid_range)])

print("grid_of_unitvector shape" , grid_of_unitvec.shape)
print("grid_of_unitvector" , grid_of_unitvec)
