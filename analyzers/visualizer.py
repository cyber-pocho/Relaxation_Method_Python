import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle


data = pd.read_csv('advanced_potential_data.csv')

rho_unique = data['rho'].unique()
z_unique = data['z'].unique()
N_rho = len(rho_unique)
N_z = len(z_unique)

print(f"Grid: {N_rho} × {N_z} points")

RHO = data['rho'].values.reshape(N_rho, N_z)
Z = data['z'].values.reshape(N_rho, N_z)
V = data['V'].values.reshape(N_rho, N_z)
E = data['E_magnitud'].values.reshape(N_rho, N_z)
EPS = data['epsilon'].values.reshape(N_rho, N_z)
LAYER = data['capa'].values.reshape(N_rho, N_z)

layers = []
for i in range(int(LAYER.max()) + 1):
    mask = (LAYER == i)
    if mask.any():
        z_vals = Z[mask]
        layers.append({
            'id': i,
            'z_min': z_vals.min(),
            'z_max': z_vals.max(),
            'eps_mean': EPS[mask].mean()
        })

print(f"Detectado {len(layers)} layers:")
for layer in layers:
    print(f"  Layer {layer['id']}: z ∈ [{layer['z_min']:.2f}, {layer['z_max']:.2f}], "
          f"<ε> = {layer['eps_mean']:.2f}")


fig = plt.figure(figsize=(18, 12))

# Distribucion de Potencial
ax1 = plt.subplot(2, 3, 1)
levels_V = np.linspace(V.min(), V.max(), 40)
contour1 = ax1.contourf(Z, RHO, V, levels=levels_V, cmap='RdBu_r')

for i in range(len(layers) - 1):
    z_interface = layers[i]['z_max']
    ax1.axvline(x=z_interface, color='white', linestyle='--', linewidth=2, alpha=0.7)

plt.colorbar(contour1, ax=ax1, label='Potencial V (V)')
ax1.set_xlabel('z', fontsize=11)
ax1.set_ylabel('ρ', fontsize=11)
ax1.set_title('Distribucion del potencial', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.2)

# Magnitud del campo electrico
ax2 = plt.subplot(2, 3, 2)
levels_E = np.linspace(0, np.percentile(E, 95), 40)  # Avoid outliers
contour2 = ax2.contourf(Z, RHO, E, levels=levels_E, cmap='hot')

for i in range(len(layers) - 1):
    z_interface = layers[i]['z_max']
    ax2.axvline(x=z_interface, color='cyan', linestyle='--', linewidth=2, alpha=0.7)

plt.colorbar(contour2, ax=ax2, label='|E| (V/m)')
ax2.set_xlabel('z', fontsize=11)
ax2.set_ylabel('ρ', fontsize=11)
ax2.set_title('Magnitud del campo E', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.2)

# Distribucion de la permitividad 

ax3 = plt.subplot(2, 3, 3)
contour3 = ax3.contourf(Z, RHO, EPS, levels=40, cmap='viridis')

for i in range(len(layers) - 1):
    z_interface = layers[i]['z_max']
    ax3.axvline(x=z_interface, color='white', linestyle='--', linewidth=2, alpha=0.7)

plt.colorbar(contour3, ax=ax3, label='ε(E)')
ax3.set_xlabel('z', fontsize=11)
ax3.set_ylabel('ρ', fontsize=11)
ax3.set_title('Distribucion de Permitividad (No lineal)', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.2)

# Densidad de energia

ax4 = plt.subplot(2, 3, 4)
energy_density = 0.5 * EPS * E**2
levels_u = np.linspace(0, np.percentile(energy_density, 95), 40)
contour4 = ax4.contourf(Z, RHO, energy_density, levels=levels_u, cmap='plasma')

for i in range(len(layers) - 1):
    z_interface = layers[i]['z_max']
    ax4.axvline(x=z_interface, color='cyan', linestyle='--', linewidth=2, alpha=0.7)

plt.colorbar(contour4, ax=ax4, label='u (J/m³)')
ax4.set_xlabel('z', fontsize=11)
ax4.set_ylabel('ρ', fontsize=11)
ax4.set_title('Densidad de energia', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.2)

# Potencial en el eje z
ax5 = plt.subplot(2, 3, 5)
z_axis_data = data[data['rho'] == 0.0]
ax5.plot(z_axis_data['z'], z_axis_data['V'], 'b-', linewidth=2.5, label='V(ρ=0, z)')

for layer in layers:
    color = plt.cm.Set3(layer['id'] % 12)
    ax5.axvspan(layer['z_min'], layer['z_max'], alpha=0.2, color=color, 
                label=f'Layer {layer["id"]} (ε={layer["eps_mean"]:.1f})')

ax5.set_xlabel('z', fontsize=11)
ax5.set_ylabel('V(ρ=0, z) (V)', fontsize=11)
ax5.set_title('Potencial en el eje z', fontsize=12, fontweight='bold')
ax5.legend(fontsize=9)
ax5.grid(True, alpha=0.3)

# Campo E para el eje z
ax6 = plt.subplot(2, 3, 6)
ax6.plot(z_axis_data['z'], z_axis_data['E_magnitud'], 'r-', linewidth=2.5, label='|E|(ρ=0, z)')

for layer in layers:
    color = plt.cm.Set3(layer['id'] % 12)
    ax6.axvspan(layer['z_min'], layer['z_max'], alpha=0.2, color=color)

ax6.set_xlabel('z', fontsize=11)
ax6.set_ylabel('|E|(ρ=0, z) (V/m)', fontsize=11)
ax6.set_title('Campo Electrico en el eje z', fontsize=12, fontweight='bold')
ax6.legend(fontsize=9)
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('advanced_results.png', dpi=300, bbox_inches='tight')
print("Saved: advanced_results.png")
plt.show()

# Segunda figura
fig2 = plt.figure(figsize=(15, 10))

#permitividad vs Campo electrico (Efecto Kerr)
ax7 = plt.subplot(2, 2, 1)
for layer in layers:
    mask = (LAYER == layer['id'])
    E_layer = E[mask]
    EPS_layer = EPS[mask]
    
    sort_idx = np.argsort(E_layer)
    ax7.plot(E_layer[sort_idx], EPS_layer[sort_idx], '.', alpha=0.3, 
             label=f'Layer {layer["id"]}', markersize=1)

ax7.set_xlabel('|E| (V/m)', fontsize=11)
ax7.set_ylabel('ε(E)', fontsize=11)
ax7.set_title('Permitividad nolineal: Efecto Kerr', fontsize=12, fontweight='bold')
ax7.legend()
ax7.grid(True, alpha=0.3)

# Visualizacion de capas
ax8 = plt.subplot(2, 2, 2)
z_range = np.linspace(z_unique.min(), z_unique.max(), 1000)
eps_profile = np.zeros_like(z_range)

for i, z_val in enumerate(z_range):
    for layer in layers:
        if layer['z_min'] <= z_val <= layer['z_max']:
            eps_profile[i] = layer['eps_mean']
            break

ax8.plot(z_range, eps_profile, 'b-', linewidth=3)
ax8.fill_between(z_range, 0, eps_profile, alpha=0.3)

for i in range(len(layers) - 1):
    z_interface = layers[i]['z_max']
    ax8.axvline(x=z_interface, color='r', linestyle='--', linewidth=2, label='Interface' if i == 0 else '')

ax8.set_xlabel('z', fontsize=11)
ax8.set_ylabel('Promedio ε', fontsize=11)
ax8.set_title('Estructura de capas dielectricas', fontsize=12, fontweight='bold')
ax8.legend()
ax8.grid(True, alpha=0.3)

# Lineas de Campo
ax9 = plt.subplot(2, 2, 3)
#Computar componentes de campo electrico
E_rho = np.zeros_like(V)
E_z = np.zeros_like(V)

d_rho = rho_unique[1] - rho_unique[0]
d_z = z_unique[1] - z_unique[0]

for i in range(1, N_rho - 1):
    for j in range(1, N_z - 1):
        E_rho[i, j] = -(V[i+1, j] - V[i-1, j]) / (2 * d_rho)
        E_z[i, j] = -(V[i, j+1] - V[i, j-1]) / (2 * d_z)

skip = 3
Z_down = Z[::skip, ::skip]
RHO_down = RHO[::skip, ::skip]
E_z_down = E_z[::skip, ::skip]
E_rho_down = E_rho[::skip, ::skip]

#plotear streamlines

strm = ax9.streamplot(Z_down, RHO_down, E_z_down, E_rho_down, 
                      color=np.sqrt(E_z_down**2 + E_rho_down**2),
                      cmap='cool', linewidth=1.5, density=1.5, arrowsize=1.5)

for i in range(len(layers) - 1):
    z_interface = layers[i]['z_max']
    ax9.axvline(x=z_interface, color='yellow', linestyle='--', linewidth=2, alpha=0.7)

plt.colorbar(strm.lines, ax=ax9, label='|E| (V/m)')
ax9.set_xlabel('z', fontsize=11)
ax9.set_ylabel('ρ', fontsize=11)
ax9.set_title('Electric Field Lines', fontsize=12, fontweight='bold')
ax9.grid(True, alpha=0.2)

# perfiles radiales para diferentes posiciones z

ax10 = plt.subplot(2, 2, 4)

z_positions = [z_unique[len(z_unique)//4], 0.0, z_unique[3*len(z_unique)//4]]
colors = ['blue', 'green', 'red']

for z_pos, color in zip(z_positions, colors):
    j_closest = np.argmin(np.abs(z_unique - z_pos))
    z_actual = z_unique[j_closest]
    
    ax10.plot(rho_unique, V[:, j_closest], '-', color=color, linewidth=2,
              label=f'z = {z_actual:.2f}')

ax10.set_xlabel('ρ', fontsize=11)
ax10.set_ylabel('V(ρ, z)', fontsize=11)
ax10.set_title('Perfiles de potencial radial', fontsize=12, fontweight='bold')
ax10.legend()
ax10.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('advanced_analysis.png', dpi=300, bbox_inches='tight')
print("Saved: advanced_analysis.png")
plt.show()

print("\n" + "="*50)
print("SIMULATION STATISTICS")
print("="*50)

print(f"\nPotential:")
print(f"  Min: {V.min():.6e} V")
print(f"  Max: {V.max():.6e} V")
print(f"  Range: {V.max() - V.min():.6e} V")

print(f"\nElectric Field:")
print(f"  Min: {E.min():.6e} V/m")
print(f"  Max: {E.max():.6e} V/m")
print(f"  Mean: {E.mean():.6e} V/m")
print(f"  Std: {E.std():.6e} V/m")

print(f"\nPermittivity (showing nonlinear effects):")
for layer in layers:
    mask = (LAYER == layer['id'])
    eps_layer = EPS[mask]
    print(f"  Layer {layer['id']}:")
    print(f"    Min ε: {eps_layer.min():.6f}")
    print(f"    Max ε: {eps_layer.max():.6f}")
    print(f"    Mean ε: {eps_layer.mean():.6f}")
    print(f"    Nonlinear variation: {(eps_layer.max() - eps_layer.min()):.6e}")

print(f"\nEnergy Density:")
print(f"  Max: {energy_density.max():.6e} J/m³")
print(f"  Total energy (integrated): {energy_density.sum() * 2 * np.pi * d_rho * d_z:.6e} J")

print("\n" + "="*50)
print("Visualization complete!")
print("="*50)
