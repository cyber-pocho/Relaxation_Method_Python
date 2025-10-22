
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def dielectric_interface_solver():
    # ========== PARÁMETROS DEL PROBLEMA ==========
    
    # Parámetros físicos
    eps1 = 2.0          # Permitividad relativa para z > 0
    eps2 = 4.0          # Permitividad relativa para z < 0
    q = 1.0             # Magnitud de la carga puntual
    a = 0.3             # Posición de la carga: (0, 0, a) con a > 0
    
    # Tamaño del dominio (coordenadas cilíndricas: ρ, z)
    rho_max = 1.0       # Distancia radial máxima
    z_min = -1.0        # Parte inferior del dominio
    z_max = 1.0         # Parte superior del dominio
    
    # Resolución de la malla
    N_rho = 81          # Número de puntos en dirección ρ
    N_z = 161           # Número de puntos en dirección z
    
    # Parámetros de relajación
    omega = 1.8         # Factor SOR (1=Gauss-Seidel, 1.8-1.9 a menudo óptimo)
    max_iter = 30000    # Iteraciones máximas
    tol = 1e-6          # Tolerancia de convergencia
    
    # ========== CONFIGURACIÓN DE LA MALLA ==========
    
    rho = np.linspace(0, rho_max, N_rho)
    z = np.linspace(z_min, z_max, N_z)
    d_rho = rho[1] - rho[0]
    d_z = z[1] - z[0]
    
    RHO, Z = np.meshgrid(rho, z, indexing='ij')
    
    # Encontrar índices de la malla
    i_charge = np.argmin(np.abs(rho - 0))      # ρ = 0
    j_charge = np.argmin(np.abs(z - a))        # z = a
    j_interface = np.argmin(np.abs(z - 0))     # z = 0
    
    print(f"Malla: {N_rho} × {N_z} puntos")
    print(f"Carga en punto de malla: (i={i_charge}, j={j_charge})")
    print(f"Interfaz en j={j_interface} (z={z[j_interface]:.4f})")
    print(f"Permitividades: ε1={eps1}, ε2={eps2}")
    
    # ========== CAMPO DE PERMITIVIDAD ==========
    
    eps = np.ones((N_rho, N_z))
    eps[:, z >= 0] = eps1
    eps[:, z < 0] = eps2
    
    # ========== DENSIDAD DE CARGA ==========
    
    # Distribuir carga puntual sobre una pequeña gaussiana para evitar singularidad
    sigma_charge = 1.5 * max(d_rho, d_z)  # Ancho de suavizado
    rho_charge = np.zeros((N_rho, N_z))
    
    for i in range(N_rho):
        for j in range(N_z):
            r_sq = rho[i]**2 + (z[j] - a)**2
            rho_charge[i, j] = q * np.exp(-r_sq / (2 * sigma_charge**2))
    
    # Normalizar para asegurar que la carga total = q
    # En coords cilíndricas: ∫∫ ρ_charge * 2πρ dρ dz = q
    total_charge = 0
    for i in range(N_rho):
        for j in range(N_z):
            if i == 0:
                # En ρ=0, usar ρ ≈ d_rho/2 para integración
                total_charge += rho_charge[i, j] * 2 * np.pi * (d_rho/2) * d_rho * d_z
            else:
                total_charge += rho_charge[i, j] * 2 * np.pi * rho[i] * d_rho * d_z
    
    rho_charge *= q / total_charge
    
    # ========== INICIALIZAR POTENCIAL ==========
    
    V = np.zeros((N_rho, N_z))
    
    # Condiciones de frontera: V = 0 en todos los bordes externos
    V[0, :] = 0      # ρ = 0 (eje, pero se actualizará)
    V[-1, :] = 0     # ρ = rho_max
    V[:, 0] = 0      # z = z_min
    V[:, -1] = 0     # z = z_max
    
    # ========== PRECALCULAR COEFICIENTES DE PLANTILLA ==========
    
    inv_drho2 = 1 / d_rho**2
    inv_dz2 = 1 / d_z**2
    
    # ========== ITERACIÓN DE RELAJACIÓN ==========
    
    hist_res = []
    hist_du = []
    
    print(f"\nIniciando iteración SOR (ω={omega})...")
    
    for iteration in range(max_iter):
        max_update = 0
        max_residual = 0
        
        # Recorrer puntos interiores
        for i in range(N_rho):
            for j in range(1, N_z - 1):  # Excluir fronteras en z
                
                # Saltar frontera radial externa
                if i == N_rho - 1:
                    continue
                
                # Tratamiento especial en la interfaz (z=0)
                at_interface = (j == j_interface)
                
                # Obtener permitividad local
                eps_center = eps[i, j]
                
                if i == 0:
                    # ========== EJE (ρ=0): USAR SIMETRÍA ==========
                    # En ρ=0: ∂V/∂ρ = 0, entonces V(−dρ) = V(+dρ)
                    # El laplaciano en coords cilíndricas se simplifica:
                    # ∇²V = ∂²V/∂ρ² + (1/ρ)∂V/∂ρ + ∂²V/∂z²
                    # En ρ→0: (1/ρ)∂V/∂ρ → ∂²V/∂ρ², entonces ∇²V = 2∂²V/∂ρ² + ∂²V/∂z²
                    
                    if not at_interface:
                        # Ecuación de Poisson estándar
                        V_rho2 = 2 * inv_drho2  # Coeficiente para ∂²V/∂ρ²
                        V_z2 = inv_dz2
                        
                        V_gs = ((V[i+1, j]) * V_rho2 +
                                (V[i, j+1] + V[i, j-1]) * V_z2 +
                                rho_charge[i, j] / eps_center) / (2 * V_rho2 + 2 * V_z2)
                    else:
                        # En la interfaz: aplicar condiciones de continuidad
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
                    # ========== PUNTOS FUERA DEL EJE (ρ>0) ==========
                    
                    if not at_interface:
                        # Poisson estándar en coords cilíndricas
                        # ∇·[ε∇V] = ε∇²V (si ε es constante en esta celda)
                        # ∇²V = ∂²V/∂ρ² + (1/ρ)∂V/∂ρ + ∂²V/∂z²
                        
                        # Diferencias finitas: (1/ρ)∂V/∂ρ ≈ (V[i+1]-V[i-1])/(2ρ*dρ)
                        V_rho2 = (V[i+1, j] + V[i-1, j]) * inv_drho2
                        V_rho1 = (V[i+1, j] - V[i-1, j]) / (2 * rho[i] * d_rho)
                        V_z2 = (V[i, j+1] + V[i, j-1]) * inv_dz2
                        
                        num = V_rho2 + V_rho1 + V_z2 + rho_charge[i, j] / eps_center
                        denom = 2 * inv_drho2 + 2 * inv_dz2
                        V_gs = num / denom
                    
                    else:
                        # En la interfaz: considerar salto de permitividad
                        # ∇·[ε∇V] requiere tratamiento especial en z=0
                        eps_up = eps[i, j+1]
                        eps_down = eps[i, j-1]
                        
                        # Parte radial (ε continua en dirección ρ)
                        V_rho2 = (V[i+1, j] + V[i-1, j]) * inv_drho2
                        V_rho1 = (V[i+1, j] - V[i-1, j]) / (2 * rho[i] * d_rho)
                        
                        # Parte axial con ε discontinua
                        V_z_contrib = (eps_up * V[i, j+1] + eps_down * V[i, j-1]) * inv_dz2
                        
                        num = (V_rho2 + V_rho1) * eps_center + V_z_contrib + rho_charge[i, j]
                        denom = eps_center * (2 * inv_drho2 + 2 * inv_dz2) + (eps_up + eps_down - 2*eps_center) * inv_dz2
                        V_gs = num / denom
                
                # Aplicar SOR
                V_old = V[i, j]
                V_new = (1 - omega) * V_old + omega * V_gs
                
                # Seguimiento de convergencia
                update = abs(V_new - V_old)
                if update > max_update:
                    max_update = update
                
                V[i, j] = V_new
        
        # Reaplicar condiciones de frontera
        V[-1, :] = 0
        V[:, 0] = 0
        V[:, -1] = 0
        
        # Calcular residuo
        for i in range(1, N_rho - 1):
            for j in range(1, N_z - 1):
                if i == 0:
                    continue  # Saltar eje para residuo (ya manejado)
                
                # Calcular residuo: ∇·[ε∇V] + ρ debe ser ≈ 0
                lap_rho = (V[i+1, j] - 2*V[i, j] + V[i-1, j]) * inv_drho2
                lap_rho1 = (V[i+1, j] - V[i-1, j]) / (2 * rho[i] * d_rho)
                lap_z = (V[i, j+1] - 2*V[i, j] + V[i, j-1]) * inv_dz2
                
                residual = eps[i, j] * (lap_rho + lap_rho1 + lap_z) + rho_charge[i, j]
                
                if abs(residual) > max_residual:
                    max_residual = abs(residual)
        
        hist_du.append(max_update)
        hist_res.append(max_residual)
        
        # Verificar convergencia
        if iteration % 500 == 0:
            print(f"Iter {iteration}: max|ΔV|={max_update:.3e}, residuo={max_residual:.3e}")
        
        if max_update < tol and max_residual < tol:
            print(f"\nConvergido en iteración {iteration}")
            print(f"Final: max|ΔV|={max_update:.3e}, residuo={max_residual:.3e}")
            break
    else:
        print(f"\nDetenido en max_iter={max_iter}")
        print(f"Final: max|ΔV|={max_update:.3e}, residuo={max_residual:.3e}")
    
    # ========== SOLUCIÓN ANALÍTICA ==========
    
    q_image = q * (eps1 - eps2) / (eps1 + eps2)
    
    V_analytical = np.zeros((N_rho, N_z))
    for i in range(N_rho):
        for j in range(N_z):
            r1 = np.sqrt(rho[i]**2 + (z[j] - a)**2)
            r2 = np.sqrt(rho[i]**2 + (z[j] + a)**2)
            
            if r1 < 1e-10:
                r1 = 1e-10  # Evitar singularidad
            if r2 < 1e-10:
                r2 = 1e-10
            
            if z[j] >= 0:
                # Medio 1: V = q/(4πε1·r1) + q'/(4πε1·r2)
                V_analytical[i, j] = (q / r1 + q_image / r2) / (4 * np.pi * eps1)
            else:
                # Medio 2: V = 2q/(4πε2·r1) * (ε1/(ε1+ε2))
                # Esto proviene de la onda transmitida
                V_analytical[i, j] = (2 * eps1 / (eps1 + eps2)) * q / (4 * np.pi * eps2 * r1)
    
    # ========== VISUALIZACIÓN ==========
    
    # 1. Mapa de potencial (contorno 2D)
    fig = plt.figure(figsize=(15, 5))
    
    # Solución numérica
    ax1 = plt.subplot(1, 3, 1)
    levels = np.linspace(V.min(), V.max(), 30)
    contour1 = ax1.contourf(Z.T, RHO.T, V.T, levels=levels, cmap='RdBu_r')
    ax1.axhline(y=0, color='k', linestyle='--', linewidth=1, label='Interfaz')
    ax1.plot(a, 0, 'ro', markersize=8, label='Carga')
    plt.colorbar(contour1, ax=ax1, label='V (numérico)')
    ax1.set_xlabel('z')
    ax1.set_ylabel('ρ')
    ax1.set_title('Solución Numérica')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Solución analítica
    ax2 = plt.subplot(1, 3, 2)
    contour2 = ax2.contourf(Z.T, RHO.T, V_analytical.T, levels=levels, cmap='RdBu_r')
    ax2.axhline(y=0, color='k', linestyle='--', linewidth=1, label='Interfaz')
    ax2.plot(a, 0, 'ro', markersize=8, label='Carga')
    plt.colorbar(contour2, ax=ax2, label='V (analítico)')
    ax2.set_xlabel('z')
    ax2.set_ylabel('ρ')
    ax2.set_title('Solución Analítica')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Diferencia
    ax3 = plt.subplot(1, 3, 3)
    diff = V - V_analytical
    contour3 = ax3.contourf(Z.T, RHO.T, diff.T, levels=30, cmap='seismic')
    ax3.axhline(y=0, color='k', linestyle='--', linewidth=1)
    plt.colorbar(contour3, ax=ax3, label='V_num - V_ana')
    ax3.set_xlabel('z')
    ax3.set_ylabel('ρ')
    ax3.set_title('Diferencia')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mapas_potencial.png', dpi=300, bbox_inches='tight')
    print("Guardado: mapas_potencial.png")
    plt.show()
    
    # 2. Potencial a lo largo del eje z (ρ=0)
    fig = plt.figure(figsize=(10, 6))
    
    plt.plot(z, V[0, :], 'b-', linewidth=2, label='Numérico')
    plt.plot(z, V_analytical[0, :], 'r--', linewidth=2, label='Analítico')
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.5, label='Interfaz')
    plt.axvline(x=a, color='g', linestyle=':', alpha=0.5, label='Posición de carga')
    plt.xlabel('z', fontsize=12)
    plt.ylabel('V(ρ=0, z)', fontsize=12)
    plt.title('Potencial a lo largo del eje z', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('potencial_eje_z.png', dpi=300, bbox_inches='tight')
    print("Guardado: potencial_eje_z.png")
    plt.show()
    
    # 3. Historial de convergencia
    fig = plt.figure(figsize=(10, 6))
    
    plt.semilogy(hist_du, 'b-', linewidth=1.5, label='max |ΔV|')
    plt.semilogy(hist_res, 'r-', linewidth=1.5, label='max |residuo|')
    plt.axhline(y=tol, color='k', linestyle='--', alpha=0.5, label=f'Tolerancia = {tol}')
    plt.xlabel('Iteración', fontsize=12)
    plt.ylabel('Error', fontsize=12)
    plt.title('Historial de Convergencia', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig('historial_convergencia.png', dpi=300, bbox_inches='tight')
    print("Guardado: historial_convergencia.png")
    plt.show()
    
    # Imprimir estadísticas de error
    error = np.abs(V - V_analytical)
    print(f"\nEstadísticas de Error:")
    print(f"Error absoluto máximo: {error.max():.6e}")
    print(f"Error absoluto medio: {error.mean():.6e}")
    print(f"Error RMS: {np.sqrt((error**2).mean()):.6e}")
    
    return V, V_analytical, RHO, Z, hist_du, hist_res

# Ejecutar el solucionador
if __name__ == "__main__":
    V, V_analytical, RHO, Z, hist_du, hist_res = dielectric_interface_solver()