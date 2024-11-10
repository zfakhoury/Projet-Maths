import math
import numpy as np

pi = math.pi
v_0 = 100

# ---------------------------------------------------------------------------------------
#                                   LAGRANGIEN
# ---------------------------------------------------------------------------------------

def lagrangian(r, h, lambd):
    return pi**2 * r**2 * (r**2 + h**2) + lambd * (pi/3 * r**2 * h - v_0)

# ---------------------------------------------------------------------------------------
#                              GRADIENT DU LAGRANGIEN
# ---------------------------------------------------------------------------------------

def grad_lagrangien(r, h, lambd):
    grad = [0, 0, 0]
    grad[0] = 4 * r**3 * pi**2 + 2 * pi**2 * h**2 * r - 2 * r * lambd * (pi/3) * h  # Par rapport à r
    grad[1] = 2 * h * pi**2 * r**2 + (pi/3) * r**2 * lambd  # Par rapport à h
    grad[2] = (pi/3) * r**2 * h - v_0  # Par rapport à lambda
    return grad

# ---------------------------------------------------------------------------------------
#                                  NORME DU GRADIENT
# ---------------------------------------------------------------------------------------

def norme(grad):
    return math.sqrt(grad[0]**2 + grad[1]**2 + grad[2]**2)

# ---------------------------------------------------------------------------------------
#                               GRADIENT À PAS FIXE
# ---------------------------------------------------------------------------------------

def gradient_pas_fixe(x_0, pas, tolerance, max_iterations):
    """
    Méthode du gradient à pas fixe pour minimiser le Lagrangien
    
    Args:
        x_0: Point initial (r_0, h_0, lambda_0)
        pas: Taille du pas fixe (alpha)
        tolerance: Critère d'arrêt pour la norme du gradient
        max_iterations: Nombre maximum d'itérations
    
    Returns:
        x: Solution optimale trouvée (r*, h*, lambda*)
        iterations: Nombre d'itérations effectuées
        historique: Liste des points visités et des normes du gradient
    """

    # Point initial
    x = x_0.copy()
    gradX = grad_lagrangien(x[0], x[1], x[2])
    historique = {'points': [x_0.copy()], 'normes': [norme(gradX)]}

    iterations = 0
    
    while norme(gradX) > tolerance and iterations < max_iterations:
        # Mise à jour: x_{k+1} = x_k - alpha * gradient
        x[0] = x[0] - pas * gradX[0]
        x[1] = x[1] - pas * gradX[1]
        x[2] = x[2] - pas * gradX[2]

        # Direction de descente d_k = -gradient
        gradX = grad_lagrangien(-x[0], -x[1], -x[2])
        
        iterations += 1
        # Stocker une copie du point courant
        historique['points'].append(x)
        historique['normes'].append(norme(gradX))
    
    return x, iterations, historique

# ---------------------------------------------------------------------------------------
#                               RECHERCHE LINÉAIRE
# ---------------------------------------------------------------------------------------

def recherche_lineaire(x, p, c1=0.0001, rho=0.5):
    """
    Recherche linéaire avec règle d'Armijo (backtracking)
    
    Args:
        x: Point courant (r, h, lambda)
        p: Direction de descente
        c1: Paramètre de la condition d'Armijo (entre 0 et 1)
        rho: Facteur de réduction du pas (entre 0 et 1)
    
    Returns:
        alpha: Pas acceptable
    """
    alpha = 1.0  # Commencer avec un pas unitaire
    
    # Valeur et gradient au point courant
    f_k = lagrangian(*x)
    grad_k = grad_lagrangien(*x)
    
    # Produit scalaire grad_k^T * p
    direction_derivee = sum(g * d for g, d in zip(grad_k, p))
    
    while True:
        # Nouveau point candidat
        x_new = [x[i] + alpha * p[i] for i in range(3)]
        f_new = lagrangian(*x_new)
        
        # Condition d'Armijo
        if f_new <= f_k + c1 * alpha * direction_derivee:
            break
        
        # Réduire le pas
        alpha *= rho
        
        # Protection contre les pas trop petits
        if alpha < 1e-10:
            break
    
    return alpha

# ---------------------------------------------------------------------------------------
#                               GRADIENT À PAS OPTIMAL
# ---------------------------------------------------------------------------------------

def gradient_pas_optimal(x_0, tolerance, max_iterations):
    """
    Méthode du gradient à pas optimal avec recherche linéaire d'Armijo:
    1. k = 0, choisir x_0
    2. Calculer direction de descente d_k = -grad_lagrangien(x_k)
    3. Trouver le pas optimal alpha_k par recherche linéaire (règle d'Armijo)
    4. X_{k+1} = X_k - alpha_k * d_k
    5. Si norme(d_k) <= tolerance: STOP, sinon k = k+1 et retour à 2.
    """

    # Point initial
    x = x_0.copy()
    gradX = grad_lagrangien(x[0], x[1], x[2])
    historique = {'points': [x_0.copy()], 'normes': [norme(gradX)]}
    iterations = 0
    
    while norme(gradX) > tolerance and iterations < max_iterations:
        # Direction de descente d_k = -gradient
        gradX = grad_lagrangien(-x[0], -x[1], -x[2])

        # Recherche du pas optimal
        pas = recherche_lineaire(x, gradX)
        
        # Mise à jour
        x[0] = x[0] - pas * gradX[0]
        x[1] = x[1] - pas * gradX[1]
        x[2] = x[2] - pas * gradX[2]
        
        iterations += 1
        historique['points'].append(x)
        historique['normes'].append(norme(gradX))
    
    return x, iterations, historique


# ---------------------------------------------------------------------------------------
#                                      WOLFE
# ---------------------------------------------------------------------------------------

def wolfe(x_k, d_k, max_iterations):
    cond1 = 0
    cond2 = 0

    gradX_k = grad_lagrangien(x_k[0], x_k[1], x_k[2])
    lagX_k = lagrangian(x_k[0], x_k[1], x_k[2])

    Psk = d_k[0] * gradX_k[0] + d_k[1] * gradX_k[1] + d_k[2] * gradX_k[2]

    alpha_min = 0
    alpha_max = 100
    alpha_k = (alpha_min + alpha_max) / 2

    iterations = 0

    while ((cond1 + cond2) < 3) and (iterations < max_iterations):
        x_k[0] = x_k[0] + alpha_k * d_k[0]
        x_k[1] = x_k[1] + alpha_k * d_k[1]
        x_k[2] = x_k[2] + alpha_k * d_k[2]

        gradX_k = grad_lagrangien(x_k[0], x_k[1], x_k[2])

        Psk = d_k[0] * gradX_k[0] + d_k[1] * gradX_k[1] + d_k[2] * gradX_k[2]

        if lagX_k > (lagX_k + cond1 * alpha_k * Psk):
            cond1 = 0
            alpha_max = alpha_k
            alpha_k = (alpha_min + alpha_max) / 2
        else:
            cond1 = 1
        
        if (-Psk > -cond2 * Psk):
            cond2 = 0
            alpha_min = alpha_k
            alpha_k = (alpha_min + alpha_max) / 2
        else:
            cond2 = 1
        
    return x_k, iterations

# ---------------------------------------------------------------------------------------
#                                      TESTS
# ---------------------------------------------------------------------------------------

def test_gradient_pas_fixe():
    x_0 = [1, 1, 1]  # (r_0, h_0, lambda_0)
    pas = 1e-5
    tolerance = 1e-3
    max_iterations = 1000
    
    solution, nb_iterations, historique = gradient_pas_fixe(x_0, pas, tolerance, max_iterations)
    
    print("\nRésultats de l'optimisation:")
    print("-" * 50)
    print(f"Point initial (r₀, h₀, λ₀) = ({x_0[0]:.6f}, {x_0[1]:.6f}, {x_0[2]:.6f})")
    print(f"Point final (r*, h*, λ*) = ({solution[0]:.6f}, {solution[1]:.6f}, {solution[2]:.6f})")
    print(f"Nombre d'itérations: {nb_iterations}")
    print(f"Norme finale du gradient: {historique['normes'][-1]:.8f}")
    
    print("\nHistorique de convergence:")
    print("-" * 50)

    steps_to_show = [0, nb_iterations//4, nb_iterations//2, 3*nb_iterations//4, nb_iterations-1]

    for i in steps_to_show:
        point = historique['points'][i]
        norm = historique['normes'][i]
        print(f"Iteration {i:4d}: (r, h, λ) = ({point[0]:10.6f}, {point[1]:10.6f}, {point[2]:10.6f}), |∇L| = {norm:.8f}")
    
def test_gradient_pas_optimal():
    x_0 = [1, 1, 1]  # (r_0, h_0, lambda_0)
    tolerance = 1e-3
    max_iterations = 1000
    
    solution, nb_iterations, historique = gradient_pas_optimal(x_0, tolerance, max_iterations)
    
    print("\nRésultats de l'optimisation (Pas Optimal):")
    print("-" * 50)
    print(f"Point initial (r₀, h₀, λ₀) = ({x_0[0]:.6f}, {x_0[1]:.6f}, {x_0[2]:.6f})")
    print(f"Point final (r*, h*, λ*) = ({solution[0]:.6f}, {solution[1]:.6f}, {solution[2]:.6f})")
    print(f"Nombre d'itérations: {nb_iterations}")
    print(f"Norme finale du gradient: {historique['normes'][-1]:.8f}")
    
    print("\nHistorique de convergence:")
    print("-" * 50)
    steps_to_show = [0, nb_iterations//4, nb_iterations//2, 3*nb_iterations//4, nb_iterations-1]
    
    for i in steps_to_show:
        point = historique['points'][i]
        norm = historique['normes'][i]
        print(f"Iteration {i:4d}: (r, h, λ) = ({point[0]:10.6f}, {point[1]:10.6f}, {point[2]:10.6f}), |∇L| = {norm:.8f}")

if __name__ == "__main__":
    test_gradient_pas_fixe()
    test_gradient_pas_optimal()

