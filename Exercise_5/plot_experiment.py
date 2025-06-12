import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# CSV laden und bereinigen
df = pd.read_csv("specific_heat_simulated_annealing_results.csv")
cols = ["q", "steps_per_temperature", "final_steps_to_convergence", "final_path_distance"]
for col in cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df["alternative_pacc"] = df["alternative_pacc"].astype(bool)
df = df.replace([np.inf, -np.inf], np.nan).dropna()

# Aufteilen
df_std = df[df["alternative_pacc"] == False]
df_alt = df[df["alternative_pacc"] == True]

# Achsenlimits
xlim = (df["q"].min(), df["q"].max())
ylim = (df["steps_per_temperature"].min(), df["steps_per_temperature"].max())
zlim_steps = (df["final_steps_to_convergence"].min(), df["final_steps_to_convergence"].max())
zlim_dist = (df["final_path_distance"].min(), df["final_path_distance"].max())

# Plot Setup
fig = plt.figure(figsize=(10, 8))  # kompaktere Figure
font_kwargs = {"fontsize": 7}

def plot_3d(ax, df, z_col, cmap, title, zlim):
    x = df["q"].values
    y = df["steps_per_temperature"].values
    z = df[z_col].values

    surf = ax.plot_trisurf(x, y, z, cmap=cmap, linewidth=0.2, edgecolor='none', antialiased=True)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)

    ax.set_xlabel("q", **font_kwargs)
    ax.set_ylabel("L", **font_kwargs)
    #ax.set_zlabel(z_col.replace("_", " ").capitalize(), **font_kwargs)
    ax.set_title(title, fontsize=10)

        # Manuelles Label links unten
    ax.text(
        xlim[1], ylim[1], zlim[1],  # Position am "linken unteren Eck"
        z_col.replace("_", " ").capitalize(),
        fontsize=9,
        rotation=90,
        ha='left',
        va='bottom'
    )


    # Achsenticks kleiner
    ax.tick_params(labelsize=8)

    return surf

# Best Konfiguraion
# Top 5 Konfigurationen mit minimaler final_path_distance
top5 = df.nsmallest(10, "final_path_distance")

# Relevante Spalten für bessere Lesbarkeit auswählen
cols_to_show = [
    "q", "steps_per_temperature", "alternative_pacc",
    "final_path_distance", "final_steps_to_convergence"
]

print("\nTop 5 Konfigurationen mit kleinstem final_path_distance:")
print(top5[cols_to_show].to_string(index=False))


# Subplot 1: Distance – Standard
ax1 = fig.add_subplot(2, 2, 1, projection='3d')
surf1 = plot_3d(ax1, df_std, "final_path_distance", "Oranges", "Path Distance (Standard p_acc)", zlim_dist)
fig.colorbar(surf1, ax=ax1, shrink=0.4, aspect=8, pad=0.08)

# Subplot 2: Distance – Alternative
ax2 = fig.add_subplot(2, 2, 2, projection='3d')
surf2 = plot_3d(ax2, df_alt, "final_path_distance", "Oranges", "Path Distance (Alternative p_acc)", zlim_dist)
fig.colorbar(surf2, ax=ax2, shrink=0.4, aspect=8, pad=0.08)

# Subplot 3: Steps – Standard
ax3 = fig.add_subplot(2, 2, 3, projection='3d')
surf3 = plot_3d(ax3, df_std, "final_steps_to_convergence", "Greens", "Steps to Convergence (Standard p_acc)", zlim_steps)
fig.colorbar(surf3, ax=ax3, shrink=0.4, aspect=8, pad=0.08)

# Subplot 4: Steps – Alternative
ax4 = fig.add_subplot(2, 2, 4, projection='3d')
surf4 = plot_3d(ax4, df_alt, "final_steps_to_convergence", "Greens", "Steps to Convergence (Alternative p_acc)", zlim_steps)
fig.colorbar(surf4, ax=ax4, shrink=0.4, aspect=8, pad=0.08)

plt.suptitle("Simulated Annealing – 3D Parameter Effects", fontsize=13)
plt.subplots_adjust(hspace=0.35, wspace=0.15, top=0.93, bottom=0.07)
plt.savefig("specific_heat_sa_3d_subplots_clean.png", dpi=200)
plt.show()


    ## Reults d
    # Notes on parameter effects:
    # q: Higher q means faster cooling → faster convergence but higher risk of local minima.
    #    Lower q means slower cooling → slower but better exploration, higher chance of finding the global minimum.
    # L: More steps per temp (higher L) → better sampling, more stable solutions but slower runtime.
    #    Lower L → faster runtime but risk of skipping good solutions.
    # pacc: Standard (exp) → fast convergence, risk of getting stuck.
    #       Logistic (1/(1+exp)) → smoother exploration, better at avoiding local minima but may converge slower.
    #                            -> jumps easier from local minima because probability distribution is smoother



""" outcome op 10"""
"""Top 5 Konfigurationen mit kleinstem final_path_distance:
   q  steps_per_temperature  alternative_pacc  final_path_distance  final_steps_to_convergence
0.60                   7500              True             6.174558                        7500
1.35                   2500             False             6.657424                         184
1.15                   5000              True             6.690504                         161
1.10                   5000             False             6.716265                        5000
0.80                   7500              True             6.739344                         781
0.90                   5000             False             6.739658                        2120
0.40                   5000             False             6.765056                        5000
1.10                  10000              True             6.815059                       10000
1.30                   5000             False             6.835869                          83
1.05                   2500             False             6.928982                        2500
"""

