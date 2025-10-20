
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def dielectric_interface_solver():
    # ========== PROBLEM PARAMETERS ==========
    
    # Physical parameters
    eps1 = 2.0          # Relative permittivity for z > 0
    eps2 = 4.0          # Relative permittivity for z < 0
    q = 1.0             # Point charge magnitude
    a = 0.3             # Charge position: (0, 0, a) with a > 0
    
    # Domain size (cylindrical coordinates: ρ, z)
    rho_max = 1.0       # Maximum radial distance
    z_min = -1.0        # Bottom of domain
    z_max = 1.0         # Top of domain
    
    # Grid resolution
    N_rho = 81          # Number of points in ρ direction
    N_z = 161           # Number of points in z direction
    
    # Relaxation parameters
    omega = 1.8         # SOR factor (1=Gauss-Seidel, 1.8-1.9 often optimal)
    max_iter = 30000    # Maximum iterations
    tol = 1e-6          # Convergence tolerance
    
    # ========== GRID SETUP ==========
    
    rho = np.linspace(0, rho_max, N_rho)
    z = np.linspace(z_min, z_max, N_z)
    d_rho = rho[1] - rho[0]
    d_z = z[1] - z[0]
    
    RHO, Z = np.meshgrid(rho, z, indexing='ij')
    
    # Find grid indices
    i_charge = np.argmin(np.abs(rho - 0))      # ρ = 0
    j_charge = np.argmin(np.abs(z - a))        # z = a
    j_interface = np.argmin(np.abs(z - 0))     # z = 0
    
    print(f"Grid: {N_rho} × {N_z} points")
    print(f"Charge at grid point: (i={i_charge}, j={j_charge})")
    print(f"Interface at j={j_interface} (z={z[j_interface]:.4f})")
    print(f"Permittivities: ε1={eps1}, ε2={eps2}")
    
    # ========== PERMITTIVITY FIELD ==========
    
    eps = np.ones((N_rho, N_z))
    eps[:, z >= 0] = eps1
    eps[:, z < 0] = eps2
    
    # ========== CHARGE DENSITY ==========
    
    # Distribute point charge over a small Gaussian to avoid singularity
    sigma_charge = 1.5 * max(d_rho, d_z)  # Smoothing width
    rho_charge = np.zeros((N_rho, N_z))
    
    for i in range(N_rho):
        for j in range(N_z):
            r_sq = rho[i]**2 + (z[j] - a)**2
            rho_charge[i, j] = q * np.exp(-r_sq / (2 * sigma_charge**2))
    
    # Normalize to ensure total charge = q
    # In cylindrical coords: ∫∫ ρ_charge * 2πρ dρ dz = q
    total_charge = 0
    for i in range(N_rho):
        for j in range(N_z):
            if i == 0:
                # At ρ=0, use ρ ≈ d_rho/2 for integration
                total_charge += rho_charge[i, j] * 2 * np.pi * (d_rho/2) * d_rho * d_z
            else:
                total_charge += rho_charge[i, j] * 2 * np.pi * rho[i] * d_rho * d_z
    
    rho_charge *= q / total_charge
    
    # ========== INITIALIZE POTENTIAL ==========
    
    V = np.zeros((N_rho, N_z))
    
    # Boundary conditions: V = 0 at all outer boundaries
    V[0, :] = 0      # ρ = 0 (axis, but will be updated)
    V[-1, :] = 0     # ρ = rho_max
    V[:, 0] = 0      # z = z_min
    V[:, -1] = 0     # z = z_max
    
    # ========== PRECOMPUTE STENCIL COEFFICIENTS ==========
    
    inv_drho2 = 1 / d_rho**2
    inv_dz2 = 1 / d_z**2
    
    # ========== RELAXATION ITERATION ==========
    
    hist_res = []
    hist_du = []
    
    print(f"\nStarting SOR iteration (ω={omega})...")
    
    for iteration in range(max_iter):
        max_update = 0
        max_residual = 0
        
        # Sweep through interior points
        for i in range(N_rho):
            for j in range(1, N_z - 1):  # Exclude z boundaries
                
                # Skip outer radial boundary
                if i == N_rho - 1:
                    continue
                
                # Special treatment at the interface (z=0)
                at_interface = (j == j_interface)
                
                # Get local permittivity
                eps_center = eps[i, j]
                
                if i == 0:
                    # ========== AXIS (ρ=0): USE SYMMETRY ==========
                    # At ρ=0: ∂V/∂ρ = 0, so V(−dρ) = V(+dρ)
                    # Laplacian in cylindrical coords simplifies:
                    # ∇²V = ∂²V/∂ρ² + (1/ρ)∂V/∂ρ + ∂²V/∂z²
                    # At ρ→0: (1/ρ)∂V/∂ρ → ∂²V/∂ρ², so ∇²V = 2∂²V/∂ρ² + ∂²V/∂z²
                    
                    if not at_interface:
                        # Standard Poisson equation
                        V_rho2 = 2 * inv_drho2  # Coefficient for ∂²V/∂ρ²
                        V_z2 = inv_dz2
                        
                        V_gs = ((V[i+1, j]) * V_rho2 +
                                (V[i, j+1] + V[i, j-1]) * V_z2 +
                                rho_charge[i, j] / eps_center) / (2 * V_rho2 + 2 * V_z2)
                    else:
                        # At interface: enforce continuity conditions
                        # ε1 ∂V/∂z|+ = ε2 ∂V/∂z|−
                        eps_up = eps[i, j+1]
                        eps_down = eps[i, j-1]
                        
                        num = (2 * V[i+1, j] * inv_drho2 +
                               eps_up * V[i, j+1] * inv_dz2 +
                               eps_down * V[i, j-1] * inv_dz2 +
                               rho_charge[i, j] / eps_center)
                        denom = 2 * inv_drho2 + (eps_up + eps_down) * inv_dz2
                        V_gs = num / denom
                
                else:
                    # ========== OFF-AXIS POINTS (ρ>0) ==========
                    
                    if not at_interface:
                        # Standard Poisson in cylindrical coords
                        # ∇·[ε∇V] = ε∇²V (if ε is constant in this cell)
                        # ∇²V = ∂²V/∂ρ² + (1/ρ)∂V/∂ρ + ∂²V/∂z²
                        
                        # Finite difference: (1/ρ)∂V/∂ρ ≈ (V[i+1]-V[i-1])/(2ρ*dρ)
                        V_rho2 = (V[i+1, j] + V[i-1, j]) * inv_drho2
                        V_rho1 = (V[i+1, j] - V[i-1, j]) / (2 * rho[i] * d_rho)
                        V_z2 = (V[i, j+1] + V[i, j-1]) * inv_dz2
                        
                        num = V_rho2 + V_rho1 + V_z2 + rho_charge[i, j] / eps_center
                        denom = 2 * inv_drho2 + 2 * inv_dz2
                        V_gs = num / denom
                    
                    else:
                        # At interface: account for permittivity jump
                        # ∇·[ε∇V] requires special treatment at z=0
                        eps_up = eps[i, j+1]
                        eps_down = eps[i, j-1]
                        
                        # Radial part (ε continuous in ρ direction)
                        V_rho2 = (V[i+1, j] + V[i-1, j]) * inv_drho2
                        V_rho1 = (V[i+1, j] - V[i-1, j]) / (2 * rho[i] * d_rho)
                        
                        # Axial part with discontinuous ε
                        V_z_contrib = (eps_up * V[i, j+1] + eps_down * V[i, j-1]) * inv_dz2
                        
                        num = (V_rho2 + V_rho1) * eps_center + V_z_contrib + rho_charge[i, j]
                        denom = eps_center * (2 * inv_drho2 + 2 * inv_dz2) + (eps_up + eps_down - 2*eps_center) * inv_dz2
                        V_gs = num / denom
                
                # Apply SOR
                V_old = V[i, j]
                V_new = (1 - omega) * V_old + omega * V_gs
                
                # Track convergence
                update = abs(V_new - V_old)
                if update > max_update:
                    max_update = update
                
                V[i, j] = V_new
        
        # Reapply boundary conditions
        V[-1, :] = 0
        V[:, 0] = 0
        V[:, -1] = 0
        
        # Compute residual
        for i in range(1, N_rho - 1):
            for j in range(1, N_z - 1):
                if i == 0:
                    continue  # Skip axis for residual (already handled)
                
                # Compute residual: ∇·[ε∇V] + ρ should be ≈ 0
                lap_rho = (V[i+1, j] - 2*V[i, j] + V[i-1, j]) * inv_drho2
                lap_rho1 = (V[i+1, j] - V[i-1, j]) / (2 * rho[i] * d_rho)
                lap_z = (V[i, j+1] - 2*V[i, j] + V[i, j-1]) * inv_dz2
                
                residual = eps[i, j] * (lap_rho + lap_rho1 + lap_z) + rho_charge[i, j]
                
                if abs(residual) > max_residual:
                    max_residual = abs(residual)
        
        hist_du.append(max_update)
        hist_res.append(max_residual)
        
        # Check convergence
        if iteration % 500 == 0:
            print(f"Iter {iteration}: max|ΔV|={max_update:.3e}, residual={max_residual:.3e}")
        
        if max_update < tol and max_residual < tol:
            print(f"\nConverged at iteration {iteration}")
            print(f"Final: max|ΔV|={max_update:.3e}, residual={max_residual:.3e}")
            break
    else:
        print(f"\nStopped at max_iter={max_iter}")
        print(f"Final: max|ΔV|={max_update:.3e}, residual={max_residual:.3e}")
    
    # ========== ANALYTICAL SOLUTION ==========
    
    q_image = q * (eps1 - eps2) / (eps1 + eps2)
    
    V_analytical = np.zeros((N_rho, N_z))
    for i in range(N_rho):
        for j in range(N_z):
            r1 = np.sqrt(rho[i]**2 + (z[j] - a)**2)
            r2 = np.sqrt(rho[i]**2 + (z[j] + a)**2)
            
            if r1 < 1e-10:
                r1 = 1e-10  # Avoid singularity
            if r2 < 1e-10:
                r2 = 1e-10
            
            if z[j] >= 0:
                # Medium 1: V = q/(4πε1·r1) + q'/(4πε1·r2)
                V_analytical[i, j] = (q / r1 + q_image / r2) / (4 * np.pi * eps1)
            else:
                # Medium 2: V = 2q/(4πε2·r1) * (ε1/(ε1+ε2))
                # This comes from the transmitted wave
                V_analytical[i, j] = (2 * eps1 / (eps1 + eps2)) * q / (4 * np.pi * eps2 * r1)
    
    # ========== VISUALIZATION ==========
    
    # 1. Potential map (2D contour)
    fig = plt.figure(figsize=(15, 5))
    
    # Numerical solution
    ax1 = plt.subplot(1, 3, 1)
    levels = np.linspace(V.min(), V.max(), 30)
    contour1 = ax1.contourf(Z.T, RHO.T, V.T, levels=levels, cmap='RdBu_r')
    ax1.axhline(y=0, color='k', linestyle='--', linewidth=1, label='Interface')
    ax1.plot(a, 0, 'ro', markersize=8, label='Charge')
    plt.colorbar(contour1, ax=ax1, label='V (numerical)')
    ax1.set_xlabel('z')
    ax1.set_ylabel('ρ')
    ax1.set_title('Numerical Solution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Analytical solution
    ax2 = plt.subplot(1, 3, 2)
    contour2 = ax2.contourf(Z.T, RHO.T, V_analytical.T, levels=levels, cmap='RdBu_r')
    ax2.axhline(y=0, color='k', linestyle='--', linewidth=1, label='Interface')
    ax2.plot(a, 0, 'ro', markersize=8, label='Charge')
    plt.colorbar(contour2, ax=ax2, label='V (analytical)')
    ax2.set_xlabel('z')
    ax2.set_ylabel('ρ')
    ax2.set_title('Analytical Solution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Difference
    ax3 = plt.subplot(1, 3, 3)
    diff = V - V_analytical
    contour3 = ax3.contourf(Z.T, RHO.T, diff.T, levels=30, cmap='seismic')
    ax3.axhline(y=0, color='k', linestyle='--', linewidth=1)
    plt.colorbar(contour3, ax=ax3, label='V_num - V_ana')
    ax3.set_xlabel('z')
    ax3.set_ylabel('ρ')
    ax3.set_title('Difference')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('potential_maps.png', dpi=300, bbox_inches='tight')
    print("Saved: potential_maps.png")
    plt.show()
    
    # 2. Potential along z-axis (ρ=0)
    fig = plt.figure(figsize=(10, 6))
    
    plt.plot(z, V[0, :], 'b-', linewidth=2, label='Numerical')
    plt.plot(z, V_analytical[0, :], 'r--', linewidth=2, label='Analytical')
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.5, label='Interface')
    plt.axvline(x=a, color='g', linestyle=':', alpha=0.5, label='Charge position')
    plt.xlabel('z', fontsize=12)
    plt.ylabel('V(ρ=0, z)', fontsize=12)
    plt.title('Potential along the z-axis', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('potential_z_axis.png', dpi=300, bbox_inches='tight')
    print("Saved: potential_z_axis.png")
    plt.show()
    
    # 3. Convergence history
    fig = plt.figure(figsize=(10, 6))
    
    plt.semilogy(hist_du, 'b-', linewidth=1.5, label='max |ΔV|')
    plt.semilogy(hist_res, 'r-', linewidth=1.5, label='max |residual|')
    plt.axhline(y=tol, color='k', linestyle='--', alpha=0.5, label=f'Tolerance = {tol}')
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Error', fontsize=12)
    plt.title('Convergence History', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig('convergence_history.png', dpi=300, bbox_inches='tight')
    print("Saved: convergence_history.png")
    plt.show()
    
    # Print error statistics
    error = np.abs(V - V_analytical)
    print(f"\nError Statistics:")
    print(f"Max absolute error: {error.max():.6e}")
    print(f"Mean absolute error: {error.mean():.6e}")
    print(f"RMS error: {np.sqrt((error**2).mean()):.6e}")
    
    return V, V_analytical, RHO, Z, hist_du, hist_res

# Run the solver
if __name__ == "__main__":
    V, V_analytical, RHO, Z, hist_du, hist_res = dielectric_interface_solver()