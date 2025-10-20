#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define PI 3.141592653589793

/* Simulation parameters structure */
typedef struct {
    double eps1;        /* Permittivity for z > 0 */
    double eps2;        /* Permittivity for z < 0 */
    double q;           /* Point charge magnitude */
    double a;           /* Charge position (0, 0, a) */
    
    double rho_max;     /* Maximum radial distance */
    double z_min;       /* Minimum z */
    double z_max;       /* Maximum z */
    
    int N_rho;          /* Grid points in ρ */
    int N_z;            /* Grid points in z */
    
    double omega;       /* SOR parameter */
    int max_iter;       /* Maximum iterations */
    double tol;         /* Convergence tolerance */
} Params;

/* Macro for 2D array indexing: A[i][j] -> A[i * cols + j] */
#define IDX(i, j, cols) ((i) * (cols) + (j))

/* Function to allocate a 2D array as a contiguous block */
double* alloc_2d(int rows, int cols) {
    double *arr = (double*)calloc(rows * cols, sizeof(double));
    if (arr == NULL) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        exit(1);
    }
    return arr;
}

/* Analytical solution using image charge method */
double analytical_potential(double rho, double z, const Params *p) {
    double q_image = p->q * (p->eps1 - p->eps2) / (p->eps1 + p->eps2);
    
    double r1 = sqrt(rho * rho + (z - p->a) * (z - p->a));
    double r2 = sqrt(rho * rho + (z + p->a) * (z + p->a));
    
    /* Avoid singularity */
    if (r1 < 1e-10) r1 = 1e-10;
    if (r2 < 1e-10) r2 = 1e-10;
    
    if (z >= 0) {
        /* Medium 1 */
        return (p->q / r1 + q_image / r2) / (4.0 * PI * p->eps1);
    } else {
        /* Medium 2 */
        return (2.0 * p->eps1 / (p->eps1 + p->eps2)) * p->q / (4.0 * PI * p->eps2 * r1);
    }
}

/* Main solver function */
void solve_dielectric_interface(const Params *p) {
    int i, j, iter;
    double d_rho, d_z;
    double inv_drho2, inv_dz2;
    double max_update, max_residual;
    int j_interface = 0, j_charge = 0;
    
    printf("Dielectric Interface Solver");
    printf("Grid: %d x %d points\n", p->N_rho, p->N_z);
    printf("Permittivities: eps1=%.2f, eps2=%.2f\n", p->eps1, p->eps2);
    printf("Charge: q=%.2f at z=%.2f\n", p->q, p->a);
    printf("SOR parameter: omega=%.2f\n", p->omega);
    
    /* Allocate grid arrays */
    double *rho = (double*)malloc(p->N_rho * sizeof(double));
    double *z = (double*)malloc(p->N_z * sizeof(double));
    
    d_rho = p->rho_max / (p->N_rho - 1);
    d_z = (p->z_max - p->z_min) / (p->N_z - 1);
    
    /* Initialize grids */
    for (i = 0; i < p->N_rho; i++) {
        rho[i] = i * d_rho;
    }
    for (j = 0; j < p->N_z; j++) {
        z[j] = p->z_min + j * d_z;
    }
    
    /* Find special indices */
    for (j = 0; j < p->N_z; j++) {
        if (fabs(z[j]) < fabs(z[j_interface])) j_interface = j;
        if (fabs(z[j] - p->a) < fabs(z[j_charge] - p->a)) j_charge = j;
    }
    
    printf("Interface at j=%d (z=%.4f)\n", j_interface, z[j_interface]);
    printf("Charge at j=%d (z=%.4f)\n\n", j_charge, z[j_charge]);
    
    /* Allocate 2D arrays */
    double *V = alloc_2d(p->N_rho, p->N_z);              /* Potential */
    double *eps = alloc_2d(p->N_rho, p->N_z);            /* Permittivity */
    double *rho_charge = alloc_2d(p->N_rho, p->N_z);     /* Charge density */
    double *V_analytical = alloc_2d(p->N_rho, p->N_z);   /* Analytical solution */
    
    /* Set permittivity field */
    for (i = 0; i < p->N_rho; i++) {
        for (j = 0; j < p->N_z; j++) {
            eps[IDX(i, j, p->N_z)] = (z[j] >= 0) ? p->eps1 : p->eps2;
        }
    }
    
    /* Distribute point charge as Gaussian */
    double sigma_charge = 1.5 * fmax(d_rho, d_z);
    double total_charge = 0.0;
    
    for (i = 0; i < p->N_rho; i++) {
        for (j = 0; j < p->N_z; j++) {
            double r_sq = rho[i] * rho[i] + (z[j] - p->a) * (z[j] - p->a);
            rho_charge[IDX(i, j, p->N_z)] = p->q * exp(-r_sq / (2.0 * sigma_charge * sigma_charge));
            
            /* Integrate in cylindrical coordinates */
            double rho_eff = (i == 0) ? d_rho / 2.0 : rho[i];
            total_charge += rho_charge[IDX(i, j, p->N_z)] * 2.0 * PI * rho_eff * d_rho * d_z;
        }
    }
    
    /* Normalize charge */
    for (i = 0; i < p->N_rho; i++) {
        for (j = 0; j < p->N_z; j++) {
            rho_charge[IDX(i, j, p->N_z)] *= p->q / total_charge;
        }
    }
    
    /* Initialize potential (already zeroed by calloc) */
    /* Boundary conditions: V = 0 at all boundaries */
    
    /* Precompute coefficients */
    inv_drho2 = 1.0 / (d_rho * d_rho);
    inv_dz2 = 1.0 / (d_z * d_z);
    
    /* Allocate convergence history arrays */
    double *hist_update = (double*)malloc(p->max_iter * sizeof(double));
    double *hist_residual = (double*)malloc(p->max_iter * sizeof(double));
    int n_iters = 0;
    
    /* ========== SOR ITERATION ========== */
    printf("Starting SOR iteration...\n");
    
    for (iter = 0; iter < p->max_iter; iter++) {
        max_update = 0.0;
        
        /* Sweep through grid */
        for (i = 0; i < p->N_rho; i++) {
            for (j = 1; j < p->N_z - 1; j++) {  /* Exclude z boundaries */
                
                /* Skip outer radial boundary */
                if (i == p->N_rho - 1) continue;
                
                int at_interface = (j == j_interface);
                double eps_center = eps[IDX(i, j, p->N_z)];
                double V_gs;
                
                if (i == 0) {
                    /* ========== AXIS (ρ=0) ========== */
                    if (!at_interface) {
                        /* Standard: ∇²V = 2∂²V/∂ρ² + ∂²V/∂z² */
                        double V_rho2 = 2.0 * inv_drho2;
                        double V_z2 = inv_dz2;
                        
                        V_gs = (V[IDX(i+1, j, p->N_z)] * V_rho2 +
                               (V[IDX(i, j+1, p->N_z)] + V[IDX(i, j-1, p->N_z)]) * V_z2 +
                               rho_charge[IDX(i, j, p->N_z)] / eps_center) / 
                               (2.0 * V_rho2 + 2.0 * V_z2);
                    } else {
                        /* Interface: enforce continuity */
                        double eps_up = eps[IDX(i, j+1, p->N_z)];
                        double eps_down = eps[IDX(i, j-1, p->N_z)];
                        
                        double num = 2.0 * V[IDX(i+1, j, p->N_z)] * inv_drho2 +
                                    eps_up * V[IDX(i, j+1, p->N_z)] * inv_dz2 +
                                    eps_down * V[IDX(i, j-1, p->N_z)] * inv_dz2 +
                                    rho_charge[IDX(i, j, p->N_z)] / eps_center;
                        double denom = 2.0 * inv_drho2 + (eps_up + eps_down) * inv_dz2;
                        V_gs = num / denom;
                    }
                    
                } else {
                    /* ========== OFF-AXIS (ρ>0) ========== */
                    if (!at_interface) {
                        /* Standard Laplacian in cylindrical coords */
                        double V_rho2 = (V[IDX(i+1, j, p->N_z)] + V[IDX(i-1, j, p->N_z)]) * inv_drho2;
                        double V_rho1 = (V[IDX(i+1, j, p->N_z)] - V[IDX(i-1, j, p->N_z)]) / 
                                       (2.0 * rho[i] * d_rho);
                        double V_z2 = (V[IDX(i, j+1, p->N_z)] + V[IDX(i, j-1, p->N_z)]) * inv_dz2;
                        
                        double num = V_rho2 + V_rho1 + V_z2 + 
                                    rho_charge[IDX(i, j, p->N_z)] / eps_center;
                        double denom = 2.0 * inv_drho2 + 2.0 * inv_dz2;
                        V_gs = num / denom;
                        
                    } else {
                        /* Interface with permittivity jump */
                        double eps_up = eps[IDX(i, j+1, p->N_z)];
                        double eps_down = eps[IDX(i, j-1, p->N_z)];
                        
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
                
                /* Apply SOR */
                double V_old = V[IDX(i, j, p->N_z)];
                double V_new = (1.0 - p->omega) * V_old + p->omega * V_gs;
                
                double update = fabs(V_new - V_old);
                if (update > max_update) max_update = update;
                
                V[IDX(i, j, p->N_z)] = V_new;
            }
        }
        
        /* Compute residual */
        max_residual = 0.0;
        for (i = 1; i < p->N_rho - 1; i++) {
            for (j = 1; j < p->N_z - 1; j++) {
                double lap_rho = (V[IDX(i+1, j, p->N_z)] - 2.0 * V[IDX(i, j, p->N_z)] + 
                                 V[IDX(i-1, j, p->N_z)]) * inv_drho2;
                double lap_rho1 = (V[IDX(i+1, j, p->N_z)] - V[IDX(i-1, j, p->N_z)]) / 
                                 (2.0 * rho[i] * d_rho);
                double lap_z = (V[IDX(i, j+1, p->N_z)] - 2.0 * V[IDX(i, j, p->N_z)] + 
                               V[IDX(i, j-1, p->N_z)]) * inv_dz2;
                
                double residual = eps[IDX(i, j, p->N_z)] * (lap_rho + lap_rho1 + lap_z) + 
                                 rho_charge[IDX(i, j, p->N_z)];
                
                if (fabs(residual) > max_residual) max_residual = fabs(residual);
            }
        }
        
        hist_update[iter] = max_update;
        hist_residual[iter] = max_residual;
        n_iters = iter + 1;
        
        /* Print progress */
        if (iter % 500 == 0) {
            printf("Iter %5d: max|ΔV|=%.3e, residual=%.3e\n", 
                   iter, max_update, max_residual);
        }
        
        /* Check convergence */
        if (max_update < p->tol && max_residual < p->tol) {
            printf("\nConverged at iteration %d\n", iter);
            printf("Final: max|delta_V|=%.3e, residual=%.3e\n", max_update, max_residual);
            break;
        }
    }
    
    /* Compute analytical solution and errors */
    double max_error = 0.0;
    double sum_error = 0.0;
    double sum_sq_error = 0.0;
    int count = 0;
    
    for (i = 0; i < p->N_rho; i++) {
        for (j = 0; j < p->N_z; j++) {
            V_analytical[IDX(i, j, p->N_z)] = analytical_potential(rho[i], z[j], p);
            double error = fabs(V[IDX(i, j, p->N_z)] - V_analytical[IDX(i, j, p->N_z)]);
            if (error > max_error) max_error = error;
            sum_error += error;
            sum_sq_error += error * error;
            count++;
        }
    }
    
    printf("Error Statistics:\n");
    printf("Max absolute error: %.6e\n", max_error);
    printf("Mean absolute error: %.6e\n", sum_error / count);
    printf("RMS error: %.6e\n", sqrt(sum_sq_error / count));
    
    /* ========== SAVE RESULTS TO CSV ========== */
    printf("\nSaving results to files...\n");
    
    /* 1. Save potential data */
    FILE *fp = fopen("potential_data.csv", "w");
    if (fp == NULL) {
        fprintf(stderr, "Error: Cannot open potential_data.csv\n");
        exit(1);
    }
    
    fprintf(fp, "rho,z,V_numerical,V_analytical,epsilon\n");
    for (i = 0; i < p->N_rho; i++) {
        for (j = 0; j < p->N_z; j++) {
            fprintf(fp, "%.8f,%.8f,%.8f,%.8f,%.8f\n",
                   rho[i], z[j],
                   V[IDX(i, j, p->N_z)],
                   V_analytical[IDX(i, j, p->N_z)],
                   eps[IDX(i, j, p->N_z)]);
        }
    }
    fclose(fp);
    printf("Saved: potential_data.csv\n");
    
    /* 2. Save z-axis data */
    fp = fopen("z_axis_data.csv", "w");
    if (fp == NULL) {
        fprintf(stderr, "Error: Cannot open z_axis_data.csv\n");
        exit(1);
    }
    
    fprintf(fp, "z,V_numerical,V_analytical\n");
    for (j = 0; j < p->N_z; j++) {
        fprintf(fp, "%.8f,%.8f,%.8f\n",
               z[j],
               V[IDX(0, j, p->N_z)],
               V_analytical[IDX(0, j, p->N_z)]);
    }
    fclose(fp);
    printf("Saved: z_axis_data.csv\n");
    
    /* 3. Save convergence history */
    fp = fopen("convergence_data.csv", "w");
    if (fp == NULL) {
        fprintf(stderr, "Error: Cannot open convergence_data.csv\n");
        exit(1);
    }
    
    fprintf(fp, "iteration,max_update,max_residual\n");
    for (iter = 0; iter < n_iters; iter++) {
        fprintf(fp, "%d,%.8e,%.8e\n", iter, hist_update[iter], hist_residual[iter]);
    }
    fclose(fp);
    printf("Saved: convergence_data.csv\n");
    
    printf("\nDone! Use the Python plotting script to visualize results.\n");
    
    /* Free memory */
    free(rho);
    free(z);
    free(V);
    free(eps);
    free(rho_charge);
    free(V_analytical);
    free(hist_update);
    free(hist_residual);
}

int main(void) {
    /* Set simulation parameters */
    Params p;
    
    p.eps1 = 2.0;       /* Permittivity for z > 0 */
    p.eps2 = 4.0;       /* Permittivity for z < 0 */
    p.q = 1.0;          /* Charge magnitude */
    p.a = 0.3;          /* Charge position */
    
    p.rho_max = 1.0;    /* Domain size */
    p.z_min = -1.0;
    p.z_max = 1.0;
    
    p.N_rho = 81;       /* Grid resolution */
    p.N_z = 161;
    
    p.omega = 1.8;      /* SOR parameter */
    p.max_iter = 30000;
    p.tol = 1e-6;
    
    solve_dielectric_interface(&p);
    
    return 0;
}