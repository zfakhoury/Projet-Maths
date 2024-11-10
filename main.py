import numpy as np
import matplotlib.pyplot as plt

pi = np.pi
v_0 = 300 # 300 cm^3 = 0.3 L

# ---------------------------------------------------------------------------------------
#                                   LAGRANGIEN
# ---------------------------------------------------------------------------------------

def lagrangien(r, h, lambd):
    return pi**2 * r**2 * (r**2 + h**2) + lambd * (pi/3 * r**2 * h - v_0)

# ---------------------------------------------------------------------------------------
#                              GRADIENT DU LAGRANGIEN
# ---------------------------------------------------------------------------------------

def grad_lagrangien(r, h, lambd):
    grad = [0, 0, 0]
    # Par rapport à r
    grad[0] = 4 * r**3 * pi**2 + 2 * pi**2 * h**2 * r + 2 * r * lambd * (pi/3) * h  
    # Par rapport à h
    grad[1] = 2 * h * pi**2 * r**2 + (pi/3) * r**2 * lambd  
    # Par rapport à lambda
    grad[2] = (pi/3) * r**2 * h - v_0  
    return grad

# ---------------------------------------------------------------------------------------
#                                  NORME DU GRADIENT
# ---------------------------------------------------------------------------------------

def norme(grad):
    return np.sqrt(grad[0]**2 + grad[1]**2 + grad[2]**2)

# ---------------------------------------------------------------------------------------
#                                      HESSIENNE
# ---------------------------------------------------------------------------------------

def hessienne(r, h, lambd):
    """
    Calcule la matrice hessienne du Lagrangien
    """
    H = np.zeros((3, 3))
    
    # Dérivées secondes par rapport à r
    H[0,0] = 12 * r**2 * pi**2 + 2 * pi**2 * h**2 + 2 * lambd * (pi/3) * h
    
    # Dérivées croisées r,h
    H[0,1] = H[1,0] = 4 * pi**2 * r * h + 2 * r * lambd * (pi/3)
    
    # Dérivées croisées r,lambda
    H[0,2] = H[2,0] = 2 * r * (pi/3) * h
    
    # Dérivées secondes par rapport à h
    H[1,1] = 2 * pi**2 * r**2
    
    # Dérivées croisées h,lambda
    H[1,2] = H[2,1] = (pi/3) * r**2
    
    # Dérivées secondes par rapport à lambda
    H[2,2] = 0
    
    return H


# ---------------------------------------------------------------------------------------
#                               GRADIENT À PAS FIXE
# ---------------------------------------------------------------------------------------

def gradient_pas_fixe(x_0, pas, tolerance, it_max):
    """
    Méthode du gradient à pas fixe pour minimiser le Lagrangien
    
    Args:
        x_0: Point initial (r_0, h_0, lambda_0)
        pas: Taille du pas fixe (alpha)
        tolerance: Critère d'arrêt pour la norme du gradient
        it_max: Nombre maximum d'itérations
    
    Returns:
        x: Solution optimale trouvée (r*, h*, lambda*)
        it: Nombre d'itérations effectuées
        historique: Liste des points visités et des normes du gradient
    """

    # Point initial
    x = x_0.copy()
    gradX = grad_lagrangien(x[0], x[1], x[2])
    historique = {'points': [x_0.copy()], 'normes': [norme(gradX)]}

    it = 0
    
    while norme(gradX) > tolerance and it < it_max:
        # Mise à jour: x_{k+1} = x_k - alpha * gradient
        # Si dérivée > 0, x avance. Sinon, x recule.
        # Le pas est toujours positif.
        x[0] = x[0] - pas * gradX[0]
        x[1] = x[1] - pas * gradX[1]
        x[2] = x[2] - pas * gradX[2]

        # Direction de descente d_k = -gradient
        gradX = grad_lagrangien(x[0], x[1], x[2])
        
        it += 1
        # Stocker une copie du point courant
        historique['points'].append(x.copy())
        historique['normes'].append(norme(gradX))
    
    return x, it, historique


# ---------------------------------------------------------------------------------------
#                               RECHERCHE LINÉAIRE
# ---------------------------------------------------------------------------------------

def recherche_lineaire(x, d_k, c1=0.5, rho=0.5):
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
    alpha = 0.001  # pas initial
    
    # Valeur et gradient au point courant
    lag_k = lagrangien(*x)
    grad_k = grad_lagrangien(*x)
    
    # Produit scalaire grad_k^T * p
    PS_k = grad_k[0] * d_k[0] + grad_k[1] * d_k[1] + grad_k[2] * d_k[2]
    
    while True:
        x_k1 = [
            x[0] + alpha * d_k[0], 
            x[1] + alpha * d_k[1], 
            x[2] + alpha * d_k[2]
        ]

        lag_k1 = lagrangien(x_k1[0], x_k1[1], x_k1[2])
        
        # Condition d'Armijo
        if lag_k1 <= lag_k + c1 * alpha * PS_k:
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

def gradient_pas_optimal(x_0, tolerance, it_max):
    """
    Méthode du gradient à pas optimal avec recherche linéaire d'Armijo
    
    Args:
        x_0: Point initial (r_0, h_0, lambda_0)
        tolerance: Critère d'arrêt pour la norme du gradient
        it_max: Nombre maximum d'itérations
    
    Returns:
        x: Solution optimale trouvée (r*, h*, lambda*)
        it: Nombre d'itérations effectuées
        historique: Historique des points visités et des normes du gradient
    """

    # Point initial
    x = x_0.copy()
    gradX = grad_lagrangien(x[0], x[1], x[2])
    historique = {'points': [x_0.copy()], 'normes': [norme(gradX)]}
    it = 0
    
    while norme(gradX) > tolerance and it < it_max:
        # Direction de descente d_k = -gradient
        direction = [-g for g in gradX]  # Create negative gradient direction
        
        # Recherche du pas optimal
        pas = recherche_lineaire(x, direction)
        
        # Mise à jour
        x[0] = x[0] - pas * gradX[0]
        x[1] = x[1] - pas * gradX[1]
        x[2] = x[2] - pas * gradX[2]
        
        # Direction de descente d_k = -gradient
        gradX = grad_lagrangien(x[0], x[1], x[2])

        it += 1
        historique['points'].append(x.copy())
        historique['normes'].append(norme(gradX))
    
    return x, it, historique


# ---------------------------------------------------------------------------------------
#                                      WOLFE
# ---------------------------------------------------------------------------------------

def wolfe(x_k, d_k, it_max, c1=0.5, c2=0.9):
    """
    Recherche linéaire avec conditions de Wolfe
    
    Args:
        x_k: Point courant (r, h, lambda)
        d_k: Direction de descente
        it_max: Nombre maximum d'itérations
        c1: Premier paramètre de Wolfe (condition d'Armijo) (entre 0 et 1)
        c2: Second paramètre de Wolfe (condition de courbure) (entre c1 et 1)
    
    Returns:
        a_k: Pas optimal trouvé
        x_k1: Nouveau point optimal
        it: Nombre d'itérations effectuées
        historique: Historique des points et des normes du gradient
    """

    cond1 = 0
    cond2 = 0

    x_k1 = [0, 0, 0]

    gradX_k = grad_lagrangien(x_k[0], x_k[1], x_k[2])
    lagX_k = lagrangien(x_k[0], x_k[1], x_k[2])

    # Calcul du produit scalaire initial
    PS_k = d_k[0]*gradX_k[0] + d_k[1]*gradX_k[1] + d_k[2]*gradX_k[2]

    a_min = 0
    a_max = 100
    a_k = (a_min + a_max) / 2

    it = 0
    historique = {'points': [x_k.copy()], 'normes': [norme(gradX_k)]}

    while ((cond1 + cond2) < 2) and (it < it_max):
        # Calcul du nouveau point
        x_k1 = [x + a_k * d for x, d in zip(x_k, d_k)]

        lagX_k1 = lagrangien(x_k1[0], x_k1[1], x_k1[2])

        gradX_k1 = grad_lagrangien(x_k1[0], x_k1[1], x_k1[2])

        # Calcul du nouveau produit scalaire
        PS_k1 = d_k[0]*gradX_k1[0] + d_k[1]*gradX_k1[1] + d_k[2]*gradX_k1[2]

        if lagX_k1 > (lagX_k + c1 * a_k * PS_k):
            cond1 = 0
            a_max = a_k
        else:
            cond1 = 1

        if -PS_k1 > -c2 * PS_k:
            cond2 = 0
            a_min = a_k
        else:
            cond2 = 1
            
        a_k = (a_min + a_max) / 2
        
        it += 1
        historique['points'].append(x_k1.copy())
        historique['normes'].append(norme(gradX_k1))
    
    return a_k, x_k1, it, historique


# ---------------------------------------------------------------------------------------
#                                      NEWTON
# ---------------------------------------------------------------------------------------

def newton(x_0, tolerance, it_max):
    """
    Méthode de Newton pour minimiser le Lagrangien
    
    Args:
        x_0: Point initial (r_0, h_0, lambda_0)
        tolerance: Critère d'arrêt pour la norme du gradient
        it_max: Nombre maximum d'itérations
    
    Returns:
        x: Solution optimale trouvée (r*, h*, lambda*)
        it: Nombre d'itérations effectuées
        historique: Historique des points visités et des normes du gradient
    """
    # Point initial
    x = x_0.copy()
    gradX = grad_lagrangien(x[0], x[1], x[2])
    historique = {'points': [x_0.copy()], 'normes': [norme(gradX)]}
    
    it = 0
    
    while norme(gradX) > tolerance and it < it_max:
        # Calcul de la hessienne
        H = hessienne(x[0], x[1], x[2])
        
        # Convertir le gradient en array numpy et le rendre négatif
        minus_grad = -np.array(gradX)

        # Résoudre le système H * d = -grad pour trouver la direction d
        # C'est comme résoudre l'équation H * d = -grad
        direction = np.linalg.solve(H, minus_grad)

        # Mise à jour: x_{k+1} = x_k + d
        x[0] = x[0] + direction[0]
        x[1] = x[1] + direction[1]
        x[2] = x[2] + direction[2]
        
        # Nouveau gradient
        gradX = grad_lagrangien(x[0], x[1], x[2])
        
        it += 1
        historique['points'].append(x.copy())
        historique['normes'].append(norme(gradX))
            
    return x, it, historique

# ---------------------------------------------------------------------------------------
#                                      TESTS
# ---------------------------------------------------------------------------------------

def test_gradient_pas_fixe():
    x_0 = [4.5, 12, 1]  # (r_0, h_0, lambda_0)
    pas = 0.00001
    tolerance = 0.001
    it_max = 1000
    
    solution, nb_iterations, historique = gradient_pas_fixe(x_0, pas, tolerance, it_max)
    
    print("\nRésultats de l'optimisation (Pas Fixe):")
    print("-" * 90)
    print(f"Point initial (r₀, h₀, λ₀) = ({x_0[0]:.6f}, {x_0[1]:.6f}, {x_0[2]:.6f})")
    print(f"Point final (r*, h*, λ*) = ({solution[0]:.6f}, {solution[1]:.6f}, {solution[2]:.6f})")
    print(f"Nombre d'itérations: {nb_iterations}")
    print(f"Norme finale du gradient: {historique['normes'][-1]:.8f}")
    
    print("\nHistorique de convergence:")
    print("-" * 90)

    steps_to_show = [0, nb_iterations//4, nb_iterations//2, 3*nb_iterations//4, nb_iterations-1]

    for i in steps_to_show:
        point = historique['points'][i]
        norm = historique['normes'][i]
        print(f"Iteration {i:4d}: (r, h, λ) = ({point[0]:10.6f}, {point[1]:10.6f}, {point[2]:10.6f}), |∇L| = {norm:.8f}")

    
def test_gradient_pas_optimal():
    x_0 = [5, 7, 1]  # (r_0, h_0, lambda_0)
    tolerance = 0.0001
    it_max = 1000
    
    solution, nb_iterations, historique = gradient_pas_optimal(x_0, tolerance, it_max)
    
    print("\nRésultats de l'optimisation (Pas Optimal):")
    print("-" * 90)
    print(f"Point initial (r₀, h₀, λ₀) = ({x_0[0]:.6f}, {x_0[1]:.6f}, {x_0[2]:.6f})")
    print(f"Point final (r*, h*, λ*) = ({solution[0]:.6f}, {solution[1]:.6f}, {solution[2]:.6f})")
    print(f"Nombre d'itérations: {nb_iterations}")
    print(f"Norme finale du gradient: {historique['normes'][-1]:.8f}")
    
    print("\nHistorique de convergence:")
    print("-" * 90)
    steps_to_show = [0, nb_iterations//4, nb_iterations//2, 3*nb_iterations//4, nb_iterations-1]
    
    for i in steps_to_show:
        point = historique['points'][i]
        norm = historique['normes'][i]
        print(f"Iteration {i:4d}: (r, h, λ) = ({point[0]:10.6f}, {point[1]:10.6f}, {point[2]:10.6f}), |∇L| = {norm:.8f}")


def test_wolfe():
    x_0 = [4.5, 12, 1]  # (r_0, h_0, lambda_0)
    d_k = [-1, -1, -1]  # direction de descente
    it_max = 1000
    
    a_k, solution, nb_iterations, historique = wolfe(x_0, d_k, it_max)
    
    print("\nRésultats de l'optimisation (Wolfe):")
    print("-" * 90)
    print(f"Point initial (r₀, h₀, λ₀) = ({x_0[0]:.6f}, {x_0[1]:.6f}, {x_0[2]:.6f})")
    print(f"Point final (r*, h*, λ*) = ({solution[0]:.6f}, {solution[1]:.6f}, {solution[2]:.6f})")
    print(f"Pas optimal (α*) = {a_k:.6f}")
    print(f"Nombre d'itérations: {nb_iterations}")
    print(f"Norme finale du gradient: {historique['normes'][-1]:.8f}")
    
    print("\nHistorique de convergence:")
    print("-" * 90)
    steps_to_show = [0, nb_iterations//4, nb_iterations//2, 3*nb_iterations//4, nb_iterations-1]
    
    for i in steps_to_show:
        point = historique['points'][i]
        norm = historique['normes'][i]
        print(f"Iteration {i:4d}: (r, h, λ) = ({point[0]:10.6f}, {point[1]:10.6f}, {point[2]:10.6f}), |∇L| = {norm:.8f}")


def test_newton():
    x_0 = [4.5, 12, 1]  # (r_0, h_0, lambda_0)
    tolerance = 0.0001
    it_max = 1000
    
    solution, nb_iterations, historique = newton(x_0, tolerance, it_max)
    
    print("\nRésultats de l'optimisation (Newton):")
    print("-" * 90)
    print(f"Point initial (r₀, h₀, λ₀) = ({x_0[0]:.6f}, {x_0[1]:.6f}, {x_0[2]:.6f})")
    print(f"Point final (r*, h*, λ*) = ({solution[0]:.6f}, {solution[1]:.6f}, {solution[2]:.6f})")
    print(f"Nombre d'itérations: {nb_iterations}")
    print(f"Norme finale du gradient: {historique['normes'][-1]:.8f}")
    
    print("\nHistorique de convergence:")
    print("-" * 90)
    steps_to_show = [0, nb_iterations//4, nb_iterations//2, 3*nb_iterations//4, nb_iterations-1]
    
    for i in steps_to_show:
        point = historique['points'][i]
        norm = historique['normes'][i]
        print(f"Iteration {i:4d}: (r, h, λ) = ({point[0]:10.6f}, {point[1]:10.6f}, {point[2]:10.6f}), |∇L| = {norm:.8f}")


if __name__ == "__main__":
    test_gradient_pas_fixe()
    test_gradient_pas_optimal()
    test_wolfe()
    test_newton()
