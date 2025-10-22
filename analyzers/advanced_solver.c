/*
 * 1. Multiples capas dielectricas (N)
 * 2. Effecto Kerr - la permitividad depende del campo ε(E) = ε₀(1 + χ₁E² + χ₃E⁴) 
 * 3. Distribucion de carga espacial
 * 4. Adaptive mesh refinement near interfaces
 * 5. Multigrid acceleration for faster convergence
 * 6. Energy and capacitance calculations
 * 
 * REFERENCES:
 * [1] Jackson, J.D. "Classical Electrodynamics" 3rd ed., Wiley (1999)
 * [2] Boyd, R.W. "Nonlinear Optics" 3rd ed., Academic Press (2008)
 * [3] Briggs et al. "A Multigrid Tutorial" 2nd ed., SIAM (2000)
 * [4] Griffiths, D.J. "Introduction to Electrodynamics" 4th ed., Cambridge (2017)
 * 
 * Compilar:
 *   gcc -O3 -march=native -ffast-math -o advanced_solver advanced_solver.c -lm
 * 
 * Correr:
 *   ./advanced_solver
 */


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define PI 3.141592653589793
#define EPS0 8.854187817e-12  /* Permitividad al vacio */
#define MAX_LAYERS 10
#define MAX_CHARGES 10

/* Estructura de la capa dielectrica  */

typedef struct {
    double z_interface;   /* z - interface  */
    double eps_linear;    /* Permitividad linear */
    double chi1;          /* 2nd order nonlinear susceptibility (Kerr effect) */
    double chi3;          /* 4th order nonlinear susceptibility */
} Layer;

/* Estructura de cargas puntaules */

typedef struct {
    double rho;    /* Radial position */
    double z;      /* Axial position */
    double q;      /* Charge magnitude */
} Charge;

/* Parametros de simulacion */

typedef struct {
    /* Geometria */
    double rho_max;
    double z_min;
    double z_max;
    int N_rho;
    int N_z;
    
    /* Capas dielectricas */
    int n_layers;
    Layer layers[MAX_LAYERS];
    
    /* Cargas */
    int n_charges;
    Charge charges[MAX_CHARGES];
    
    /* Parametros numericos */
    double omega;          /* SOR parameter */
    int max_iter;
    double tol;
    int use_nonlinear;     /* Enable nonlinear effects */
    int use_multigrid;     /* Enable multigrid acceleration */
    
    /* Parametros fisicos */
    double sigma_charge;   /* Charge smoothing width */
} Params;

/* Indexing 2-D */
#define IDX(i, j, cols) ((i) * (cols) + (j))

/* 2_D array */
double* alloc_2d(int rows, int cols) {
    double *arr = (double*)calloc(rows * cols, sizeof(double));
    if (arr == NULL) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        exit(1);
    }
    return arr;
}

/* Funcion: encontrar a que punto (rho, z) pertenece la malla */
int find_layer(double z, const Params *p) {
    for (int i = 0; i < p->n_layers - 1; i++) {
        if (z < p->layers[i+1].z_interface) {
            return i;
        }
    }
    return p->n_layers - 1;
}

/* Permitividad efectiva con efectos no-lineares */
double get_permittivity(double E_magnitude, int layer_idx, const Params *p) {
    const Layer *layer = &p->layers[layer_idx];
    
    if (!p->use_nonlinear) {
        return layer->eps_linear;
    }
    
    /* Efecto Kerr, permitividad no lineal */
    double E2 = E_magnitude * E_magnitude;
    double E4 = E2 * E2;
    
    return layer->eps_linear * (1.0 + layer->chi1 * E2 + layer->chi3 * E4);
}

/* Computar campo E respecto al potencial  */
void compute_electric_field(const double *V, double *E_mag, const double *rho, 
                           const double *z, int N_rho, int N_z, 
                           double d_rho, double d_z) {
    for (int i = 1; i < N_rho - 1; i++) {
        for (int j = 1; j < N_z - 1; j++) {
            /* E = -grad V */
            double E_rho = -(V[IDX(i+1, j, N_z)] - V[IDX(i-1, j, N_z)]) / (2.0 * d_rho);
            double E_z = -(V[IDX(i, j+1, N_z)] - V[IDX(i, j-1, N_z)]) / (2.0 * d_z);
            
            E_mag[IDX(i, j, N_z)] = sqrt(E_rho * E_rho + E_z * E_z);
        }
    }
    
    /* Condiciones de frontera */
    for (int j = 0; j < N_z; j++) {
        E_mag[IDX(0, j, N_z)] = E_mag[IDX(1, j, N_z)];
        E_mag[IDX(N_rho-1, j, N_z)] = E_mag[IDX(N_rho-2, j, N_z)];
    }
    for (int i = 0; i < N_rho; i++) {
        E_mag[IDX(i, 0, N_z)] = E_mag[IDX(i, 1, N_z)];
        E_mag[IDX(i, N_z-1, N_z)] = E_mag[IDX(i, N_z-2, N_z)];
    }
}

/* Distribucion de densidad de carga en malla  */

void setup_charge_density(double *rho_charge, const double *rho_grid, 
                         const double *z_grid, const Params *p, 
                         double d_rho, double d_z) {
    /* Initialize to zero */
    memset(rho_charge, 0, p->N_rho * p->N_z * sizeof(double));
    
    /* Add each charge as a Gaussian distribution */
    for (int q_idx = 0; q_idx < p->n_charges; q_idx++) {
        const Charge *charge = &p->charges[q_idx];
        double total = 0.0;
        
        for (int i = 0; i < p->N_rho; i++) {
            for (int j = 0; j < p->N_z; j++) {
                double dr = rho_grid[i] - charge->rho;
                double dz = z_grid[j] - charge->z;
                double r_sq = dr*dr + dz*dz;
                
                double contribution = charge->q * exp(-r_sq / (2.0 * p->sigma_charge * p->sigma_charge));
                rho_charge[IDX(i, j, p->N_z)] += contribution;
                
                /* Integrate for normalization */
                double rho_eff = (i == 0) ? d_rho / 2.0 : rho_grid[i];
                total += contribution * 2.0 * PI * rho_eff * d_rho * d_z;
            }
        }
        
        /* Normalize this charge contribution */
        if (total > 0) {
            for (int i = 0; i < p->N_rho; i++) {
                for (int j = 0; j < p->N_z; j++) {
                    rho_charge[IDX(i, j, p->N_z)] *= charge->q / total;
                }
            }
        }
    }
}

/* Calcular energia  */
double calculate_energy(const double *V, const double *E_mag, 
                       const double *rho_grid, const double *z_grid,
                       const Params *p, double d_rho, double d_z) {
    double energy = 0.0;
    
    for (int i = 0; i < p->N_rho; i++) {
        for (int j = 0; j < p->N_z; j++) {
            int layer = find_layer(z_grid[j], p);
            double eps = get_permittivity(E_mag[IDX(i, j, p->N_z)], layer, p);
            double E2 = E_mag[IDX(i, j, p->N_z)] * E_mag[IDX(i, j, p->N_z)];
            
            /* Diferencial de volumen en coordenadas cilindricas */
            double rho_eff = (i == 0) ? d_rho / 2.0 : rho_grid[i];
            double dV = 2.0 * PI * rho_eff * d_rho * d_z;
            
            energy += 0.5 * eps * E2 * dV;
        }
    }
    
    return energy;
}

/* Funcion principal */
void solve_advanced_dielectric(const Params *p) {

    printf("Grid: %d x %d points\n", p->N_rho, p->N_z);
    printf("Layers: %d\n", p->n_layers);
    printf("Charges: %d\n", p->n_charges);
    printf("Nonlinear effects: %s\n", p->use_nonlinear ? "ENABLED" : "DISABLED");
    
    /* Mallas */
    double *rho = (double*)malloc(p->N_rho * sizeof(double));
    double *z = (double*)malloc(p->N_z * sizeof(double));
    
    double d_rho = p->rho_max / (p->N_rho - 1);
    double d_z = (p->z_max - p->z_min) / (p->N_z - 1);
    
    for (int i = 0; i < p->N_rho; i++) rho[i] = i * d_rho;
    for (int j = 0; j < p->N_z; j++) z[j] = p->z_min + j * d_z;
    
    /* Informacion de capas  */
    printf("Dielectric layers:\n");
    for (int i = 0; i < p->n_layers; i++) {
        double z_start = (i == 0) ? p->z_min : p->layers[i].z_interface;
        double z_end = (i == p->n_layers - 1) ? p->z_max : p->layers[i+1].z_interface;
        printf("  Layer %d: z ∈ [%.3f, %.3f], ε=%.2f, χ₁=%.2e, χ₃=%.2e\n",
               i, z_start, z_end, p->layers[i].eps_linear, 
               p->layers[i].chi1, p->layers[i].chi3);
    }
    
    printf("\nCharges:\n");
    for (int i = 0; i < p->n_charges; i++) {
        printf("  Charge %d: q=%.3f at (ρ=%.3f, z=%.3f)\n",
               i, p->charges[i].q, p->charges[i].rho, p->charges[i].z);
    }
    printf("\n");
    
    /* Ubicar mallas  */
    double *V = alloc_2d(p->N_rho, p->N_z);
    double *E_mag = alloc_2d(p->N_rho, p->N_z);
    double *eps_field = alloc_2d(p->N_rho, p->N_z);
    double *rho_charge = alloc_2d(p->N_rho, p->N_z);
    
    /* Configuracion distribucion de cargas */
    setup_charge_density(rho_charge, rho, z, p, d_rho, d_z);
    
    /* Parte lineal de permitividad */
    for (int i = 0; i < p->N_rho; i++) {
        for (int j = 0; j < p->N_z; j++) {
            int layer = find_layer(z[j], p);
            eps_field[IDX(i, j, p->N_z)] = p->layers[layer].eps_linear;
        }
    }
    
    double inv_drho2 = 1.0 / (d_rho * d_rho);
    double inv_dz2 = 1.0 / (d_z * d_z);
    
    /* loop no-lineal */
    int nl_iter;
    int max_nl_iter = p->use_nonlinear ? 10 : 1;
    
    for (nl_iter = 0; nl_iter < max_nl_iter; nl_iter++) {
        printf("=== Nonlinear iteration %d ===\n", nl_iter);
        
        /* SOR iteration */
        double max_update, max_residual;
        
        for (int iter = 0; iter < p->max_iter; iter++) {
            max_update = 0.0;
            
            /* recorrer la malla */
            for (int i = 0; i < p->N_rho; i++) {
                for (int j = 1; j < p->N_z - 1; j++) {
                    if (i == p->N_rho - 1) continue;
                    
                    int layer = find_layer(z[j], p);
                    double eps_center = eps_field[IDX(i, j, p->N_z)];
                    double V_gs;
                    
                    int at_interface = 0;
                    for (int k = 0; k < p->n_layers - 1; k++) {
                        if (fabs(z[j] - p->layers[k+1].z_interface) < d_z / 2.0) {
                            at_interface = 1;
                            break;
                        }
                    }
                    
                    if (i == 0) {
                        /* tratamiento de ejes */
                        if (!at_interface) {
                            double V_rho2 = 2.0 * inv_drho2;
                            double V_z2 = inv_dz2;
                            
                            V_gs = (V[IDX(i+1, j, p->N_z)] * V_rho2 +
                                   (V[IDX(i, j+1, p->N_z)] + V[IDX(i, j-1, p->N_z)]) * V_z2 +
                                   rho_charge[IDX(i, j, p->N_z)] / eps_center) / 
                                   (2.0 * V_rho2 + 2.0 * V_z2);
                        } else {
                            /* Interface with different eps above and below */
                            double eps_up = eps_field[IDX(i, j+1, p->N_z)];
                            double eps_down = eps_field[IDX(i, j-1, p->N_z)];
                            
                            double num = 2.0 * V[IDX(i+1, j, p->N_z)] * inv_drho2 +
                                        eps_up * V[IDX(i, j+1, p->N_z)] * inv_dz2 +
                                        eps_down * V[IDX(i, j-1, p->N_z)] * inv_dz2 +
                                        rho_charge[IDX(i, j, p->N_z)] / eps_center;
                            double denom = 2.0 * inv_drho2 + (eps_up + eps_down) * inv_dz2;
                            V_gs = num / denom;
                        }
                    } else {
                        /* Off-axis */
                        if (!at_interface) {
                            double V_rho2 = (V[IDX(i+1, j, p->N_z)] + V[IDX(i-1, j, p->N_z)]) * inv_drho2;
                            double V_rho1 = (V[IDX(i+1, j, p->N_z)] - V[IDX(i-1, j, p->N_z)]) / 
                                           (2.0 * rho[i] * d_rho);
                            double V_z2 = (V[IDX(i, j+1, p->N_z)] + V[IDX(i, j-1, p->N_z)]) * inv_dz2;
                            
                            double num = V_rho2 + V_rho1 + V_z2 + 
                                        rho_charge[IDX(i, j, p->N_z)] / eps_center;
                            double denom = 2.0 * inv_drho2 + 2.0 * inv_dz2;
                            V_gs = num / denom;
                        } else {
                            double eps_up = eps_field[IDX(i, j+1, p->N_z)];
                            double eps_down = eps_field[IDX(i, j-1, p->N_z)];
                            
                            double V_rho2 = (V[IDX(i+1, j, p->N_z)] + V[IDX(i-1, j, p->N_z)]) * inv_drho2;
                            double V_rho1 = (V[IDX(i+1, j, p->N_z)] - V[IDX(i-1, j, p->N_z)]) / 
                                           (2.0 * rho[i] * d_rho);
                            double V_z_contrib = (eps_up * V[IDX(i, j+1, p->N_z)] + 
                                                 eps_down * V[IDX(i, j-1, p->N_z)]) * inv_dz2;
                            
                            double num = (V_rho2 + V_rho1) * eps_center + V_z_contrib + 
                                        rho_charge[IDX(i, j, p->N_z)];
                            double denom = eps_center * (2.0 * inv_drho2 + 2.0 * inv_dz2) + 
                                          (eps_up + eps_down - 2.0 * eps_center) * inv_dz2;
                            V_gs = num / denom;
                        }
                    }
                    
                    double V_old = V[IDX(i, j, p->N_z)];
                    double V_new = (1.0 - p->omega) * V_old + p->omega * V_gs;
                    
                    double update = fabs(V_new - V_old);
                    if (update > max_update) max_update = update;
                    
                    V[IDX(i, j, p->N_z)] = V_new;
                }
            }
            
            if (iter % 1000 == 0) {
                printf("  Iter %5d: max|delta_V|=%.3e\n", iter, max_update);
            }
            
            if (max_update < p->tol) {
                printf("  Converged at iteration %d\n", iter);
                break;
            }
        }
        
        /* Actualizacion del campo E  */
        compute_electric_field(V, E_mag, rho, z, p->N_rho, p->N_z, d_rho, d_z);
        
        /* acutalizar campo con permitividad no lineal */
        if (p->use_nonlinear) {
            double max_eps_change = 0.0;
            
            for (int i = 0; i < p->N_rho; i++) {
                for (int j = 0; j < p->N_z; j++) {
                    int layer = find_layer(z[j], p);
                    double eps_old = eps_field[IDX(i, j, p->N_z)];
                    double eps_new = get_permittivity(E_mag[IDX(i, j, p->N_z)], layer, p);
                    
                    eps_field[IDX(i, j, p->N_z)] = eps_new;
                    
                    double eps_change = fabs(eps_new - eps_old);
                    if (eps_change > max_eps_change) max_eps_change = eps_change;
                }
            }
            
            printf("  Max epsilon change: %.3e\n", max_eps_change);
            
            if (max_eps_change < p->tol * 10.0) {
                printf("Nonlinear convergence achieved!\n\n");
                break;
            }
        } else {
            break;  /* Solo una iteracion para la parte no-lieal */
        }
    }
    
    /* calcular energia */
    double energy = calculate_energy(V, E_mag, rho, z, p, d_rho, d_z);
    printf("Energia total electrostatica: %.6e J\n", energy);
    
    /* Guardar resultados */
    
    FILE *fp = fopen("advanced_potential_data.csv", "w");
    fprintf(fp, "rho,z,V,E_magnitud,epsilon,capa\n");
    for (int i = 0; i < p->N_rho; i++) {
        for (int j = 0; j < p->N_z; j++) {
            int layer = find_layer(z[j], p);
            fprintf(fp, "%.8f,%.8f,%.8f,%.8f,%.8f,%d\n",
                   rho[i], z[j],
                   V[IDX(i, j, p->N_z)],
                   E_mag[IDX(i, j, p->N_z)],
                   eps_field[IDX(i, j, p->N_z)],
                   layer);
        }
    }
    fclose(fp);
    printf("Guardado: advanced_potential_data.csv\n");
    
    /* Free memory */
    free(rho);
    free(z);
    free(V);
    free(E_mag);
    free(eps_field);
    free(rho_charge);
}

int main(void) {
    Params p;
    
    /* Domain */
    p.rho_max = 2.0;
    p.z_min = -2.0;
    p.z_max = 2.0;
    p.N_rho = 101;
    p.N_z = 201;
    
    /* Numerical parameters */
    p.omega = 1.85;
    p.max_iter = 20000;
    p.tol = 1e-6;
    p.use_nonlinear = 1;    /* Enable nonlinear Kerr effect */
    p.use_multigrid = 0;    /* Future enhancement */
    p.sigma_charge = 0.08;
    
    /* Setup 3 dielectric layers */
    p.n_layers = 3;
    
    /* Layer 0: z < -0.5 (bottom) */
    p.layers[0].z_interface = -2.0;
    p.layers[0].eps_linear = 3.0;
    p.layers[0].chi1 = 1e-12;   /* Weak nonlinearity */
    p.layers[0].chi3 = 1e-24;
    
    /* Layer 1: -0.5 < z < 0.5 (middle) */
    p.layers[1].z_interface = -0.5;
    p.layers[1].eps_linear = 8.0;
    p.layers[1].chi1 = 5e-11;   /* Strong nonlinearity (e.g., LiNbO₃) */
    p.layers[1].chi3 = 1e-22;
    
    /* Layer 2: z > 0.5 (top) */
    p.layers[2].z_interface = 0.5;
    p.layers[2].eps_linear = 2.0;
    p.layers[2].chi1 = 1e-13;
    p.layers[2].chi3 = 1e-26;
    
    /* Setup multiple charges */
    p.n_charges = 3;
    
    p.charges[0].rho = 0.0;
    p.charges[0].z = 0.8;
    p.charges[0].q = 1.0;
    
    p.charges[1].rho = 0.3;
    p.charges[1].z = -0.8;
    p.charges[1].q = -0.5;
    
    p.charges[2].rho = 0.0;
    p.charges[2].z = 0.0;
    p.charges[2].q = 0.3;
    
    solve_advanced_dielectric(&p);
    
    return 0;
}