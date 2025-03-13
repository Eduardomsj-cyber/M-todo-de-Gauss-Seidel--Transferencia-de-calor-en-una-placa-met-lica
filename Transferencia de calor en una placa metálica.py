import numpy as np

# Coeficientes del sistema de ecuaciones (matriz A)
A = np.array([
    [20, -5, -3, 0, 0],
    [-4, 18, -2, -1, 0],
    [-3, -1, 22, -5, 0],
    [0, -2, -4, 25, -1]
])

# Términos independientes (vector b)
b = np.array([100, 120, 130, 150])

# Inicialización de las soluciones iniciales (por ejemplo, cero)
x = np.zeros_like(b)

# Parámetros de iteración
iterations = 25
tolerance = 1e-6

# Método de Jacobi
for k in range(iterations):
    x_new = np.zeros_like(x)
    
    for i in range(len(b)):
        # La fórmula de Jacobi
        sum_ax = np.dot(A[i, :], x) - A[i, i] * x[i]
        x_new[i] = (b[i] - sum_ax) / A[i, i]

    # Verificar si la solución ha convergido
    if np.linalg.norm(x_new - x, ord=np.inf) < tolerance:
        print(f"Solución convergente después de {k+1} iteraciones")
        break

    x = x_new

# Resultado final
print("Solución del sistema:")
print(x)
