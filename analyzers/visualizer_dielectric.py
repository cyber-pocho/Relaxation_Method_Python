import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

def visualizar_resultados_csv():
    """
    Visualizador para los archivos CSV generados por el solucionador
    de interfaz dieléctrica en C
    """
    
    # ========== CARGAR DATOS ==========
    try:
        # Cargar datos de potencial
        df_potential = pd.read_csv('potential_data.csv')
        
        # Cargar datos del eje z
        df_z_axis = pd.read_csv('z_axis_data.csv')
        
        # Cargar datos de convergencia
        df_convergence = pd.read_csv('convergence_data.csv')
        
        
    except FileNotFoundError as e:
        print(f"\n Error: No se pudo encontrar el archivo {e.filename}")
        return
    
    # ========== PROCESAR DATOS DE POTENCIAL ==========
    
    # Obtener valores únicos de rho y z
    rho_vals = df_potential['rho'].unique()
    z_vals = df_potential['z'].unique()
    
    N_rho = len(rho_vals)
    N_z = len(z_vals)
    
    print(f"Dimensiones de la malla: {N_rho} × {N_z}")
    print(f"Rango de ρ: [{rho_vals.min():.3f}, {rho_vals.max():.3f}]")
    print(f"Rango de z: [{z_vals.min():.3f}, {z_vals.max():.3f}]")
    
    # Crear arreglos 2D
    RHO = df_potential['rho'].values.reshape((N_rho, N_z))
    Z = df_potential['z'].values.reshape((N_rho, N_z))
    V_num = df_potential['V_numerical'].values.reshape((N_rho, N_z))
    V_ana = df_potential['V_analytical'].values.reshape((N_rho, N_z))
    epsilon = df_potential['epsilon'].values.reshape((N_rho, N_z))
    
    # Calcular diferencia
    diff = V_num - V_ana
    
    # ========== ESTADÍSTICAS DE ERROR ==========
    print("ESTADÍSTICAS DE ERROR")
    error_abs = np.abs(diff)
    print(f"Error absoluto máximo:  {error_abs.max():.6e}")
    print(f"Error absoluto medio:   {error_abs.mean():.6e}")
    print(f"Error RMS:              {np.sqrt((error_abs**2).mean()):.6e}")
    print(f"Error relativo medio:   {(error_abs / (np.abs(V_ana) + 1e-10)).mean():.6e}")
    
    # ========== VISUALIZACIÓN ==========
    
    # Configurar estilo
    plt.style.use('default')
    
    # 1. MAPAS DE POTENCIAL (2D)
    fig = plt.figure(figsize=(16, 5))
    
    # Niveles de contorno compartidos
    v_min = min(V_num.min(), V_ana.min())
    v_max = max(V_num.max(), V_ana.max())
    levels = np.linspace(v_min, v_max, 30)
    
    # Solución numérica
    ax1 = plt.subplot(1, 3, 1)
    contour1 = ax1.contourf(Z.T, RHO.T, V_num.T, levels=levels, cmap='RdBu_r')
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='Interfaz')
    ax1.set_xlabel('z', fontsize=12)
    ax1.set_ylabel('ρ', fontsize=12)
    ax1.set_title('Solución Numérica', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3, linestyle=':')
    cbar1 = plt.colorbar(contour1, ax=ax1, label='V (numérico)')
    
    # Solución analítica
    ax2 = plt.subplot(1, 3, 2)
    contour2 = ax2.contourf(Z.T, RHO.T, V_ana.T, levels=levels, cmap='RdBu_r')
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='Interfaz')
    ax2.set_xlabel('z', fontsize=12)
    ax2.set_ylabel('ρ', fontsize=12)
    ax2.set_title('Solución Analítica', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3, linestyle=':')
    cbar2 = plt.colorbar(contour2, ax=ax2, label='V (analítico)')
    
    # Diferencia
    ax3 = plt.subplot(1, 3, 3)
    contour3 = ax3.contourf(Z.T, RHO.T, diff.T, levels=30, cmap='seismic')
    ax3.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='Interfaz')
    ax3.set_xlabel('z', fontsize=12)
    ax3.set_ylabel('ρ', fontsize=12)
    ax3.set_title('Diferencia (Numérico - Analítico)', fontsize=14, fontweight='bold')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3, linestyle=':')
    cbar3 = plt.colorbar(contour3, ax=ax3, label='V_num - V_ana')
    
    plt.tight_layout()
    plt.savefig('visualizacion_mapas_potencial.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. POTENCIAL A LO LARGO DEL EJE Z
    fig = plt.figure(figsize=(10, 6))
    
    z_axis = df_z_axis['z'].values
    V_num_z = df_z_axis['V_numerical'].values
    V_ana_z = df_z_axis['V_analytical'].values
    
    plt.plot(z_axis, V_num_z, 'b-', linewidth=2.5, label='Numérico', alpha=0.8)
    plt.plot(z_axis, V_ana_z, 'r--', linewidth=2, label='Analítico', alpha=0.8)
    plt.axvline(x=0, color='black', linestyle='--', linewidth=1.5, 
                alpha=0.5, label='Interfaz')
    
    plt.xlabel('z', fontsize=13)
    plt.ylabel('V(ρ=0, z)', fontsize=13)
    plt.title('Potencial a lo largo del eje z', fontsize=15, fontweight='bold')
    plt.legend(fontsize=12, loc='best')
    plt.grid(True, alpha=0.3, linestyle=':')
    plt.tight_layout()
    plt.savefig('visualizacion_eje_z.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. HISTORIAL DE CONVERGENCIA
    fig = plt.figure(figsize=(11, 6))
    
    iteraciones = df_convergence['iteration'].values
    max_update = df_convergence['max_update'].values
    max_residuo = df_convergence['max_residual'].values
    
    plt.semilogy(iteraciones, max_update, 'b-', linewidth=2, 
                 label='max |ΔV|', alpha=0.8)
    plt.semilogy(iteraciones, max_residuo, 'r-', linewidth=2, 
                 label='max |residuo|', alpha=0.8)
    
    # Línea de tolerancia (asumiendo 1e-6 como en el código)
    tol = 1e-6
    plt.axhline(y=tol, color='green', linestyle='--', linewidth=1.5, 
                alpha=0.6, label=f'Tolerancia = {tol:.0e}')
    
    plt.xlabel('Iteración', fontsize=13)
    plt.ylabel('Error (escala logarítmica)', fontsize=13)
    plt.title('Historial de Convergencia', fontsize=15, fontweight='bold')
    plt.legend(fontsize=12, loc='best')
    plt.grid(True, which='both', alpha=0.3, linestyle=':')
    plt.tight_layout()
    plt.savefig('visualizacion_convergencia.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. MAPA DE ERROR ABSOLUTO
    fig = plt.figure(figsize=(10, 7))
    
    ax = plt.subplot(1, 1, 1)
    error_plot = ax.contourf(Z.T, RHO.T, error_abs.T, levels=30, cmap='hot_r')
    ax.axhline(y=0, color='cyan', linestyle='--', linewidth=1.5, 
               alpha=0.9, label='Interfaz')
    ax.set_xlabel('z', fontsize=12)
    ax.set_ylabel('ρ', fontsize=12)
    ax.set_title('Mapa de Error Absoluto |V_num - V_ana|', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle=':')
    cbar = plt.colorbar(error_plot, ax=ax, label='Error absoluto')
    
    plt.tight_layout()
    plt.savefig('visualizacion_error.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 5. CAMPO ELÉCTRICO (GRADIENTE DEL POTENCIAL)
    print("Calculando y graficando campo eléctrico...")
    
    # Calcular gradiente del potencial numérico
    # E = -∇V
    # En coordenadas cilíndricas: E_ρ = -∂V/∂ρ, E_z = -∂V/∂z
    
    grad_rho, grad_z = np.gradient(V_num, rho_vals, z_vals)
    E_rho = -grad_rho
    E_z = -grad_z
    E_mag = np.sqrt(E_rho**2 + E_z**2)
    
    fig = plt.figure(figsize=(12, 5))
    
    # Magnitud del campo eléctrico
    ax1 = plt.subplot(1, 2, 1)
    contour_E = ax1.contourf(Z.T, RHO.T, E_mag.T, levels=30, cmap='viridis')
    ax1.axhline(y=0, color='white', linestyle='--', linewidth=1.5, 
                alpha=0.8, label='Interfaz')
    ax1.set_xlabel('z', fontsize=12)
    ax1.set_ylabel('ρ', fontsize=12)
    ax1.set_title('Magnitud del Campo Eléctrico |E|', 
                  fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3, linestyle=':')
    plt.colorbar(contour_E, ax=ax1, label='|E|')
    
    # Líneas de campo (streamplot)
    ax2 = plt.subplot(1, 2, 2)
    
    # Submuestrear para mejor visualización
    skip = max(1, N_rho // 40)
    skip_z = max(1, N_z // 40)
    
    stream = ax2.streamplot(z_vals[::skip_z], rho_vals[::skip], 
                           E_z.T[::skip, ::skip_z], E_rho.T[::skip, ::skip_z],
                           color=E_mag.T[::skip, ::skip_z], cmap='plasma',
                           density=1.5, linewidth=1.2, arrowsize=1.2)
    ax2.axhline(y=0, color='white', linestyle='--', linewidth=1.5, 
                alpha=0.8, label='Interfaz')
    ax2.set_xlabel('z', fontsize=12)
    ax2.set_ylabel('ρ', fontsize=12)
    ax2.set_title('Líneas de Campo Eléctrico', 
                  fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3, linestyle=':')
    plt.colorbar(stream.lines, ax=ax2, label='|E|')
    
    plt.tight_layout()
    plt.savefig('visualizacion_campo_electrico.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 6. CORTES TRANSVERSALES
    
    fig = plt.figure(figsize=(14, 5))
    
    # Encontrar índices para diferentes posiciones de z
    z_positions = [z_vals[N_z//4], 0.0, z_vals[3*N_z//4]]
    z_indices = [np.argmin(np.abs(z_vals - zp)) for zp in z_positions]
    
    # Cortes en z constante
    ax1 = plt.subplot(1, 2, 1)
    for idx, zp in zip(z_indices, z_positions):
        ax1.plot(rho_vals, V_num[:, idx], linewidth=2, 
                label=f'z = {zp:.3f}', alpha=0.8)
    ax1.set_xlabel('ρ', fontsize=12)
    ax1.set_ylabel('V(ρ, z)', fontsize=12)
    ax1.set_title('Cortes Transversales (z constante)', 
                  fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle=':')
    
    # Cortes en ρ constante
    ax2 = plt.subplot(1, 2, 2)
    rho_positions = [0.0, rho_vals[N_rho//3], rho_vals[2*N_rho//3]]
    rho_indices = [np.argmin(np.abs(rho_vals - rp)) for rp in rho_positions]
    
    for idx, rp in zip(rho_indices, rho_positions):
        ax2.plot(z_vals, V_num[idx, :], linewidth=2, 
                label=f'ρ = {rp:.3f}', alpha=0.8)
    ax2.axvline(x=0, color='gray', linestyle='--', linewidth=1.5, 
                alpha=0.5, label='Interfaz')
    ax2.set_xlabel('z', fontsize=12)
    ax2.set_ylabel('V(ρ, z)', fontsize=12)
    ax2.set_title('Cortes Transversales (ρ constante)', 
                  fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle=':')
    
    plt.tight_layout()
    plt.savefig('visualizacion_cortes.png', dpi=300, bbox_inches='tight')
    plt.show()



if __name__ == "__main__":
    visualizar_resultados_csv()