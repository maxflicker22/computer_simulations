import numpy as np


def diffusion_constant_with_v_(v_s, dim, dt, total_time):
    v_0 = v_s[0, :, :]
    v_t = v_s[1:, :, :]
    v_dot_pro = np.sum(v_0 * v_t, axis=-1)

    v_dot_pro_mean = np.mean(v_dot_pro, axis=1)

    # Integrate over dt
    D = np.trapezoid(v_dot_pro_mean, dx=dt) / dim

    return D

num_steps = 22
num_partical = 3
dim = 2
dt = 0.1
total_time = (np.arange(0, 1.1, dt))

v_s = np.ones((num_steps, num_partical, dim))

D = diffusion_constant_with_v_(v_s, dim, dt, total_time)

print(D)

