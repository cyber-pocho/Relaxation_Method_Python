# Potencial Electrostático en Interfaces Dieléctricas: Solución Numérica Usando Métodos de Relajación

## Tabla de Contenidos
1. [Visión General](#visión-general)
2. [Fundamentos Teóricos](#fundamentos-teóricos)
3. [Formulación Matemática](#formulación-matemática)
4. [Métodos Numéricos](#métodos-numéricos)
5. [Características Avanzadas](#características-avanzadas)
6. [Detalles de Implementación](#detalles-de-implementación)
7. [Cómo Usar](#cómo-usar)
8. [Resultados y Validación](#resultados-y-validación)
9. [Referencias](#referencias)

---

## Visión General

Este proyecto implementa un resolvedor numérico completo para problemas electrostáticos que involucran interfaces dieléctricas usando **métodos de relajación** (Gauss-Seidel y Sobre-Relajación Sucesiva - SOR). El resolvedor maneja:

- Múltiples capas dieléctricas con interfaces arbitrarias
- Permitividad no lineal (efecto Kerr)
- Múltiples cargas puntuales
- Coordenadas cilíndricas axisimétricas (ρ, z)
- Cálculos de energía y visualización de campos

**Lenguajes:** C puro (para rendimiento) con scripts de visualización en Python

---

## Fundamentos Teóricos

### 1. Electrostática en Medios Dieléctricos

#### 1.1 Ecuaciones de Maxwell en Electrostática

En ausencia de campos que varían con el tiempo, las ecuaciones de Maxwell se reducen a:

```
∇ × E = 0                    (Ley de Faraday)
∇ · D = ρ                    (Ley de Gauss)
```

donde:
- **E** es el campo eléctrico [V/m]
- **D** es el campo de desplazamiento eléctrico [C/m²]
- **ρ** es la densidad de carga [C/m³]

#### 1.2 Relaciones Constitutivas

Para dieléctricos lineales e isotrópicos:

```
D = ε E = ε₀εᵣ E
```

donde:
- **ε** es la permitividad [F/m]
- **ε₀** = 8.854×10⁻¹² F/m (permitividad del vacío)
- **εᵣ** es la permitividad relativa (adimensional)

#### 1.3 Potencial Eléctrico

Dado que ∇ × E = 0, el campo eléctrico es conservativo y puede expresarse como:

```
E = -∇V
```

donde **V** es el potencial eléctrico [V].

#### 1.4 Ecuación de Poisson

Combinando las relaciones anteriores:

```
∇ · (ε∇V) = -ρ
```

Para permitividad uniforme, esto se simplifica a:

```
ε∇²V = -ρ
```

En regiones sin cargas libres (ρ = 0), esto se convierte en la **ecuación de Laplace**:

```
∇²V = 0
```

---

### 2. Interfaces Dieléctricas

#### 2.1 Condiciones de Frontera

En una interfaz entre dos dieléctricos (z = 0), se cumplen las siguientes condiciones:

**Continuidad del campo eléctrico tangencial:**
```
E₁ₜ = E₂ₜ  →  ∂V₁/∂ρ = ∂V₂/∂ρ
```

**Continuidad del desplazamiento normal (sin carga superficial):**
```
D₁ₙ = D₂ₙ  →  ε₁ ∂V₁/∂z = ε₂ ∂V₂/∂z
```

**Continuidad del potencial:**
```
V₁ = V₂  (en la interfaz)
```

#### 2.2 Método de Imágenes

Para una carga puntual q en (0, 0, a) en el medio 1 (z > 0) sobre una interfaz infinita (z = 0) con el medio 2 (z < 0):

**Solución Analítica (z > 0):**
```
V(ρ,z) = (1/4πε₁)[q/r₁ + q'/r₂]
```

donde:
- r₁ = √(ρ² + (z-a)²) - distancia a la carga real
- r₂ = √(ρ² + (z+a)²) - distancia a la carga imagen
- q' = q(ε₁-ε₂)/(ε₁+ε₂) - magnitud de la carga imagen

**Solución Analítica (z < 0):**
```
V(ρ,z) = [2ε₁/(ε₁+ε₂)] · q/(4πε₂r₁)
```

Esta solución analítica sirve como validación para nuestro enfoque numérico.

---

### 3. Coordenadas Cilíndricas

#### 3.1 Laplaciano Axisimétrico

Para problemas con simetría cilíndrica (∂/∂φ = 0):

```
∇²V = ∂²V/∂ρ² + (1/ρ)∂V/∂ρ + ∂²V/∂z²
```

**Caso especial en ρ = 0:** Usando la regla de L'Hôpital:

```
lim(ρ→0) [(1/ρ)∂V/∂ρ] = ∂²V/∂ρ²
```

Por lo tanto, en el eje:

```
∇²V|ₚ₌₀ = 2∂²V/∂ρ² + ∂²V/∂z²
```

#### 3.2 Ecuación de Poisson en Coordenadas Cilíndricas

```
∇ · [ε(z)∇V] = ∂/∂ρ[ε ρ ∂V/∂ρ]/ρ + ∂/∂z[ε ∂V/∂z] = -ρ(r)
```

---

## Formulación Matemática

### 4. Discretización por Diferencias Finitas

#### 4.1 Configuración de la Malla

Discretizar el dominio (ρ, z) ∈ [0, ρₘₐₓ] × [zₘᵢₙ, zₘₐₓ]:

```
ρᵢ = i·Δρ,    i = 0, 1, ..., Nₚ-1
zⱼ = zₘᵢₙ + j·Δz,    j = 0, 1, ..., Nz-1
```

donde:
- Δρ = ρₘₐₓ/(Nₚ-1)
- Δz = (zₘₐₓ - zₘᵢₙ)/(Nz-1)

#### 4.2 Segundas Derivadas (Diferencias Centrales)

```
∂²V/∂ρ²|ᵢ,ⱼ ≈ (Vᵢ₊₁,ⱼ - 2Vᵢ,ⱼ + Vᵢ₋₁,ⱼ)/Δρ²

∂²V/∂z²|ᵢ,ⱼ ≈ (Vᵢ,ⱼ₊₁ - 2Vᵢ,ⱼ + Vᵢ,ⱼ₋₁)/Δz²
```

#### 4.3 Primera Derivada (para el término 1/ρ)

```
∂V/∂ρ|ᵢ,ⱼ ≈ (Vᵢ₊₁,ⱼ - Vᵢ₋₁,ⱼ)/(2Δρ)
```

#### 4.4 Plantilla de Cinco Puntos (Puntos Interiores, ρ > 0)

Para ε uniforme, la ecuación de Poisson discretizada:

```
(Vᵢ₊₁,ⱼ - 2Vᵢ,ⱼ + Vᵢ₋₁,ⱼ)/Δρ² + (Vᵢ₊₁,ⱼ - Vᵢ₋₁,ⱼ)/(2ρᵢΔρ) + (Vᵢ,ⱼ₊₁ - 2Vᵢ,ⱼ + Vᵢ,ⱼ₋₁)/Δz² = -ρᵢ,ⱼ/ε
```

Resolviendo para Vᵢ,ⱼ:

```
Vᵢ,ⱼ = [(Vᵢ₊₁,ⱼ + Vᵢ₋₁,ⱼ)/Δρ² + (Vᵢ₊₁,ⱼ - Vᵢ₋₁,ⱼ)/(2ρᵢΔρ) + (Vᵢ,ⱼ₊₁ + Vᵢ,ⱼ₋₁)/Δz² + ρᵢ,ⱼ/ε] / [2/Δρ² + 2/Δz²]
```

#### 4.5 Puntos en el Eje (ρ = 0)

```
Vᵢ₌₀,ⱼ = [2Vᵢ₊₁,ⱼ/Δρ² + (Vᵢ,ⱼ₊₁ + Vᵢ,ⱼ₋₁)/Δz² + ρ₀,ⱼ/ε] / [2/Δρ² + 2/Δz²]
```

#### 4.6 Tratamiento de Interfaces

En z = z_interfaz donde ε cambia:

```
ε↑∂V/∂z|⁺ = ε↓∂V/∂z|⁻
```

Discretizado:

```
Vᵢ,ⱼ = [términos_radiales + ε↑Vᵢ,ⱼ₊₁/Δz² + ε↓Vᵢ,ⱼ₋₁/Δz² + fuente] / [denom_radial + (ε↑ + ε↓)/Δz²]
```

---

## Métodos Numéricos

### 5. Métodos de Relajación

#### 5.1 Método de Gauss-Seidel

Esquema iterativo que actualiza cada punto de la malla usando los valores más recientes:

```
Vᵢ,ⱼ⁽ᵏ⁺¹⁾ = f(Vᵢ₊₁,ⱼ⁽ᵏ⁺¹⁾, Vᵢ₋₁,ⱼ⁽ᵏ⁺¹⁾, Vᵢ,ⱼ₊₁⁽ᵏ⁾, Vᵢ,ⱼ₋₁⁽ᵏ⁺¹⁾)
```

donde k es el número de iteración.

#### 5.2 Sobre-Relajación Sucesiva (SOR)

Acelera la convergencia mediante sobre-corrección:

```
Vᵢ,ⱼ⁽ᵏ⁺¹⁾ = (1-ω)Vᵢ,ⱼ⁽ᵏ⁾ + ω·Vᵢ,ⱼᴳˢ
```

donde:
- **ω** = 1: Gauss-Seidel puro
- **ω** ∈ (1, 2): sobre-relajación (convergencia más rápida)
- **ω** = 2: valor teórico máximo (a menudo inestable)
- **Óptimo:** ω ∈ [1.8, 1.9] para la mayoría de los problemas

#### 5.3 Criterios de Convergencia

La iteración se detiene cuando:

```
max|Vᵢ,ⱼ⁽ᵏ⁺¹⁾ - Vᵢ,ⱼ⁽ᵏ⁾| < tolerancia   Y
max|residual| < tolerancia
```

donde el residual es:

```
rᵢ,ⱼ = ε∇²Vᵢ,ⱼ + ρᵢ,ⱼ
```

Tolerancia típica: 10⁻⁶

#### 5.4 Complejidad Computacional

- **Memoria:** O(N²) para una malla N×N
- **Tiempo por iteración:** O(N²)
- **Iteraciones para converger:** O(N) para SOR, O(N²) para Jacobi
- **Total:** O(N³) para SOR

---

## Características Avanzadas

### 6. Permitividad No Lineal (Efecto Kerr)

#### 6.1 Mecanismo Físico

En campos eléctricos intensos, la polarización del material se vuelve no lineal:

```
P = ε₀χ⁽¹⁾E + ε₀χ⁽³⁾E³ + ...
```

Esto lleva a una permitividad dependiente del campo:

```
ε(E) = ε₀(1 + χ⁽¹⁾ + χ₁|E|² + χ₃|E|⁴)
```

donde:
- **χ₁**: coeficiente de Kerr [m²/V²]
- **χ₃**: no linealidad de orden superior [m⁴/V⁴]

#### 6.2 Solución Auto-Consistente

Dado que ε depende de E = -∇V, necesitamos acoplamiento iterativo:

```
Algoritmo:
1. Inicializar: ε⁽⁰⁾ = ε_lineal
2. Para n = 0, 1, 2, ... hasta convergencia:
   a. Resolver: ∇·[ε⁽ⁿ⁾∇V⁽ⁿ⁺¹⁾] = -ρ  (SOR)
   b. Calcular: E⁽ⁿ⁺¹⁾ = -∇V⁽ⁿ⁺¹⁾
   c. Actualizar: ε⁽ⁿ⁺¹⁾ = ε(|E⁽ⁿ⁺¹⁾|)
   d. Verificar: |ε⁽ⁿ⁺¹⁾ - ε⁽ⁿ⁾| < tol
```

#### 6.3 Ejemplos Físicos

**Material: LiNbO₃ (Niobato de Litio)**
- ε lineal: ~30-80 (dependiendo de la orientación del cristal)
- χ₁ ≈ 10⁻¹¹ m²/V²
- Aplicaciones: Moduladores electro-ópticos, óptica no lineal

**Material: CS₂ (Disulfuro de Carbono)**
- ε lineal: ~2.6
- Gran efecto Kerr
- Aplicaciones: Interruptores ópticos, autoenfoque

### 7. Cálculos de Energía

#### 7.1 Energía Electrostática

La energía total almacenada en el campo eléctrico:

```
U = (1/2) ∫ ε(r)|E(r)|² dV
```

En coordenadas cilíndricas:

```
U = (1/2) ∫∫∫ ε(ρ,z)|E(ρ,z)|² · 2πρ dρ dz
```

Discretizada:

```
U ≈ π Σᵢ Σⱼ εᵢ,ⱼ |Eᵢ,ⱼ|² · ρᵢ · Δρ · Δz
```

#### 7.2 Capacitancia

Para un sistema de dos conductores:

```
C = Q/V = 2U/V²
```

donde Q es la carga total y V es la diferencia de potencial.

---

## Detalles de Implementación

### 8. Estructura del Código

#### 8.1 Resolvedor Básico (`dielectric_solver.c`)
- Interfaz única
- Permitividad lineal
- Iteración SOR
- ~400 líneas

#### 8.2 Resolvedor Avanzado (`advanced_solver.c`)
- Múltiples capas (hasta 10)
- Permitividad no lineal
- Múltiples cargas (hasta 10)
- Cálculo de energía
- ~700 líneas

#### 8.3 Gestión de Memoria

Los arreglos se almacenan como 1D con macro de indexación 2D:
```c
#define IDX(i, j, cols) ((i) * (cols) + (j))
```

Ventajas:
- Amigable con caché (memoria contigua)
- Fácil de asignar/liberar
- Acceso más rápido

#### 8.4 Banderas de Optimización

```bash
-O3              # Optimización máxima
-march=native    # Instrucciones específicas de CPU
-ffast-math      # Punto flotante rápido (ligeramente menos preciso)
-lm              # Vincular biblioteca matemática
```

Aceleración esperada: 10-50× vs Python

---

## Cómo Usar

### 9. Compilación y Ejecución

#### 9.1 Resolvedor Básico
```bash
# Compilar
gcc -O3 -march=native -o dielectric_solver dielectric_solver.c -lm

# Ejecutar
./dielectric_solver

# Graficar
python plot_results.py
```

#### 9.2 Resolvedor Avanzado
```bash
# Compilar
gcc -O3 -march=native -ffast-math -o advanced_solver advanced_solver.c -lm

# Ejecutar
./advanced_solver

# Graficar
python advanced_plotter.py
```

#### 9.3 Modificar Parámetros

Editar la función `main()` en el archivo C:

```c
// Dominio
p.rho_max = 2.0;
p.z_min = -2.0;
p.z_max = 2.0;
p.N_rho = 101;
p.N_z = 201;

// Numérico
p.omega = 1.85;
p.tol = 1e-6;
p.use_nonlinear = 1;

// Agregar capas
p.layers[0].eps_linear = 3.0;
p.layers[0].chi1 = 1e-12;

// Agregar cargas
p.charges[0].q = 1.0;
p.charges[0].z = 0.8;
```

---

## Referencias

### 11. Literatura Clave

#### Libros de Texto
1. **Jackson, J.D.** (1999). *Classical Electrodynamics*, 3ra ed. Wiley.
   - Capítulos 4-5: Problemas de frontera dieléctrica
   
2. **Griffiths, D.J.** (2017). *Introduction to Electrodynamics*, 4ta ed. Cambridge.
   - Capítulo 4: Campos eléctricos en la materia

3. **Boyd, R.W.** (2008). *Nonlinear Optics*, 3ra ed. Academic Press.
   - Capítulo 4: El índice de refracción dependiente de la intensidad
   
4. **Briggs, W.L., Henson, V.E., McCormick, S.F.** (2000). *A Multigrid Tutorial*, 2da ed. SIAM.
   - Métodos iterativos para EDPs

#### Artículos
5. **Kerr, J.** (1875). "A new relation between electricity and light". *Phil. Mag.* 50, 337-348.

6. **Young, D.M.** (1954). "Iterative methods for solving partial difference equations of elliptic type". *Trans. Amer. Math. Soc.* 76, 92-111.
   - Teoría del método SOR

7. **Courant, R., Friedrichs, K., Lewy, H.** (1928). "Über die partiellen Differenzengleichungen der mathematischen Physik". *Math. Ann.* 100, 32-74.
   - Métodos de diferencias finitas

#### Recursos en Línea
8. MIT OpenCourseWare: 6.013 Electromagnetismo y Aplicaciones
9. Electromagnetismo Computacional (ETH Zürich)
10. Manual NIST de Funciones Matemáticas

---

## Licencia

Licencia MIT - Libre para uso académico y comercial.

## Contacto

Para preguntas o contribuciones, por favor abra un issue en el repositorio.

## Agradecimientos

Basado en métodos clásicos de:
- Classical Electrodynamics de J.D. Jackson
- Numerical Recipes in C
- Cursos de Física Computacional del MIT

---

**Última Actualización:** Octubre 2025
**Versión:** 2.0 (Resolvedor Avanzado No Lineal)