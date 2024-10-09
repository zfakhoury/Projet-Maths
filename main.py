import math
import numpy as np

# Constants
pi = 3.14159
v0 = 1.0

# Helper functions
def square(x):
    return x * x

def cube(x):
    return x * x * x

# Lagrangien function
def Lagrangien(r, h, lambda_):
    return square(pi) * square(r) * (square(r) + square(h)) + lambda_ * (pi / 3 * square(r) * h - v0)

# Gradient function
def Gradient(r, h, lambda_):
    grad = [0.0, 0.0, 0.0]
    
    # par rapport a r
    grad[0] = 4 * cube(r) * square(pi) + 2 * square(pi) * square(h) * r + 2 * r * lambda_ * (pi / 3) * h
    # par rapport a h
    grad[1] = 2 * h * square(pi) * square(r) + (pi / 3) * square(r) * lambda_
    # par rapport a lambda
    grad[2] = (pi / 3) * square(r) * h - v0

    return grad

# NormeGrad function
def NormeGrad(grad):
    return math.sqrt(square(grad[0]) + square(grad[1]) + square(grad[2]))

# Gradient descent with fixed step size
def GradPasFixe(x0, pas, epsilon):
    x = np.array(x0, dtype=float)
    gradient = np.zeros(3)
    norme = 0.0
    i = 0
    
    while True:
        i += 1
        # Update gradient
        gradient = Gradient(x[0], x[1], x[2])
        
        # Update x values
        x[0] = x[0] - pas * gradient[0]
        print(f"i={i}: x[0]={x[0]:.3f}")
        
        x[1] = x[1] - pas * gradient[1]
        print(f"i={i}: x[1]={x[1]:.3f}")
        
        x[2] = x[2] - pas * gradient[2]
        print(f"i={i}: x[2]={x[2]:.3f}")
        
        norme = NormeGrad(gradient)
        
        if norme <= epsilon:
            break
    
    return x.tolist()

# Gradient descent with optimal step size
def GradPasOptimal(x0, pas, epsilon):
    x = np.array(x0, dtype=float)
    gradient = np.zeros(3)
    norme = 0.0
    i = 0
    
    while True:
        i += 1
        # Update gradient
        gradient = Gradient(x[0], x[1], x[2])
        
        # Calculate optimal step size (pas optimal)
        pas_optimal = pas / (i + 1)
        
        # Update x values
        x[0] = x[0] - pas_optimal * gradient[0]
        print(f"i={i}: x[0]={x[0]:.3f}")
        
        x[1] = x[1] - pas_optimal * gradient[1]
        print(f"i={i}: x[1]={x[1]:.3f}")
        
        x[2] = x[2] - pas_optimal * gradient[2]
        print(f"i={i}: x[2]={x[2]:.3f}")
        
        norme = NormeGrad(gradient)
        
        if norme <= epsilon:
            break
    
    return x.tolist()

# Main function
if __name__ == "__main__":
    x0 = [1, 1, 1]
    
    # Call GradPasFixe
    result_fixe = GradPasFixe(x0, 1, 0.001)
    print(f"Result of GradPasFixe: {result_fixe}")
    
    # Call GradPasOptimal
    result_optimal = GradPasOptimal(x0, 1, 0.001)
    print(f"Result of GradPasOptimal: {result_optimal}")