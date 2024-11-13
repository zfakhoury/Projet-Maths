import numpy as np
import matplotlib.pyplot as plt

pi = np.pi
v_0 = 300 # 300 cm^3 = 0.3 L


def surface(r, h):
    return pi**2 * r**2 * (r**2 + h**2)


def volume(r, h):
    return pi/3 * r**2 * h

# ---------------------------------------------------------------------------------------
#                                   LAGRANGIEN
# ---------------------------------------------------------------------------------------

def lagrangien(r, h, lambd):
    return surface(r, h) + lambd * (volume(r, h) - v_0)

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
        # Direction de descente d_k = -gradient
        direction = [-gradX[0], -gradX[1], -gradX[2]]
        
        # Mise à jour: x_{k+1} = x_k + alpha * d_k
        #                      = x_k - alpha * gradient
        x[0] = x[0] + pas * direction[0]  # équivalent à x[0] - pas * gradX[0]
        x[1] = x[1] + pas * direction[1]  # équivalent à x[1] - pas * gradX[1]
        x[2] = x[2] + pas * direction[2]  # équivalent à x[2] - pas * gradX[2]
        
        # Calcul du nouveau gradient
        gradX = grad_lagrangien(x[0], x[1], x[2])
        
        it += 1
        historique['points'].append(x.copy())
        historique['normes'].append(norme(gradX))
    
    return x, it, historique


# ---------------------------------------------------------------------------------------
#                               RECHERCHE LINÉAIRE
# ---------------------------------------------------------------------------------------

def recherche_lineaire(x, direction, c1=0.5, rho=0.5):
    """
    Recherche linéaire avec règle d'Armijo (backtracking)
    
    Args:
        x: Point courant (r, h, lambda)
        direction: Direction de descente
        c1: Paramètre de la condition d'Armijo (entre 0 et 1)
        rho: Facteur de réduction du pas (entre 0 et 1)
    
    Returns:
        alpha: Pas acceptable
    """

    x_new = [0, 0, 0]
    alpha = 1.0  # Commencer avec un pas unitaire
    
    # Valeur et gradient au point courant
    lag_k = lagrangien(x[0], x[1], x[2])
    grad_k = grad_lagrangien(x[0], x[1], x[2])
    
    # Produit scalaire grad_k^T * d_k
    PS_k = grad_k[0] * direction[0] + grad_k[1] * direction[1] + grad_k[2] * direction[2]
    
    while True:
        # Calcul du nouveau point x_{k+1} = x_k + alpha * d_k
        x_new[0] = x[0] + alpha * direction[0]
        x_new[1] = x[1] + alpha * direction[1]
        x_new[2] = x[2] + alpha * direction[2]
        
        # Valeur du Lagrangien au nouveau point
        lag_new = lagrangien(x_new[0], x_new[1], x_new[2])
        
        # Condition d'Armijo: f(x + alpha*d) ≤ f(x) + c1*alpha*grad_f(x)^T*d
        if lag_new <= lag_k + c1 * alpha * PS_k:
            break
        
        # Réduire le pas par le facteur rho
        alpha = alpha * rho
        
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
        direction = [-gradX[0], -gradX[1], -gradX[2]]
        
        # Recherche du pas optimal par la méthode d'Armijo
        pas = recherche_lineaire(x, direction)
        
        # Mise à jour: x_{k+1} = x_k + alpha * d_k
        x[0] = x[0] + pas * direction[0]
        x[1] = x[1] + pas * direction[1]
        x[2] = x[2] + pas * direction[2]
        
        # Calcul du nouveau gradient
        gradX = grad_lagrangien(x[0], x[1], x[2])
        
        it += 1
        historique['points'].append(x.copy())
        historique['normes'].append(norme(gradX))
    
    return x, it, historique

# ---------------------------------------------------------------------------------------
#                                    QUASI-NEWTON
# ---------------------------------------------------------------------------------------

def quasi_newton(x_0, tolerance, it_max):
    """
    Méthode Quasi-Newton BFGS pour minimiser le Lagrangien
    
    Args:
        x_0: Point initial (r_0, h_0, lambda_0)
        tolerance: Critère d'arrêt pour la norme du gradient
        it_max: Nombre maximum d'itérations
    
    Returns:
        x: Solution optimale trouvée (r*, h*, lambda*)
        it: Nombre d'itérations effectuées
        historique: Historique des points visités et des normes du gradient
    """
    
    n = len(x_0)
    x = x_0.copy()
    
    # Initialisation de l'approximation de l'inverse du Hessien
    H = np.eye(n) # Matrice identité (diagonale de 1)
    
    gradX = grad_lagrangien(x[0], x[1], x[2])
    historique = {'points': [x_0.copy()], 'normes': [norme(gradX)]}
    
    it = 0
    
    while norme(gradX) > tolerance and it < it_max:
        # Direction de descente: d_k = -H_k * grad f(x_k)
        direction = np.zeros(len(gradX))
        for i in range(len(gradX)):
            for j in range(len(gradX)):
                direction[i] -= H[i,j] * gradX[j]
        
        # Recherche linéaire pour trouver le pas optimal
        alpha = recherche_lineaire(x, direction.tolist())
        
        # Ancien gradient
        grad_old = np.array(gradX)
        
        # Mise à jour du point
        s = alpha * direction  # s_k = x_{k+1} - x_k = alpha_k * d_k
        x_new = x + s
        
        # Nouveau gradient
        gradX = grad_lagrangien(x_new[0], x_new[1], x_new[2])
        y = np.array(gradX) - grad_old  # y_k = grad f(x_{k+1}) - grad f(x_k)
        
        # Mise à jour de l'approximation de l'inverse du Hessien (formule BFGS)
        ys = y[0]*s[0] + y[1]*s[1] + y[2]*s[2]
        
        if abs(ys) > 1e-10:
            rho = 1.0 / ys
        else:
            rho = 1.0

        I = np.eye(n)
        # Calcul des termes intermédiaires
        sy_outer = np.outer(s, y)
        ys_outer = np.outer(y, s)
        ss_outer = np.outer(s, s)
        
        # Calcul des termes (I - rho * s*y^T) et (I - rho * y*s^T)
        term1 = I - rho * sy_outer
        term2 = I - rho * ys_outer
        
        # Multiplication par H
        temp = term1 @ H
        
        # Multiplication par le second terme
        result = temp @ term2
        
        # Addition du dernier terme
        H = result + rho * ss_outer
        
        # Mise à jour du point courant
        x = x_new
        
        it += 1
        historique['points'].append(x.copy())
        historique['normes'].append(norme(gradX))
    
    return x, it, historique


# ---------------------------------------------------------------------------------------
#                                      WOLFE
# ---------------------------------------------------------------------------------------

def wolfe(x_k, d_k, it_max, c1=0.3, c2=0.6):
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

def plot_optimization_results(historique, method_name):
    """Helper function to plot surface and volume evolution for any optimization method"""
    # Calculate surface and volume values for each point
    surface_values = [surface(point[0], point[1]) for point in historique['points']]
    volume_values = [volume(point[0], point[1]) for point in historique['points']]
    
    # Get initial and final points
    initial_point = historique['points'][0]
    final_point = historique['points'][-1]
    
    # Create two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Surface evolution plot
    ax1.plot(range(len(surface_values)), surface_values, 'b-', linewidth=2)
    ax1.scatter(range(len(surface_values)), surface_values, color='red', s=30)
    ax1.set_title(f"Évolution de la surface pendant l'optimisation ({method_name})")
    ax1.set_xlabel("Itération")
    ax1.set_ylabel("Surface")
    ax1.grid(True)
    
    # Annotate initial surface values and point
    ax1.annotate(f'Initial:\nS = {surface_values[0]:.2f}\nr = {initial_point[0]:.2f}\nh = {initial_point[1]:.2f}', 
                xy=(0, surface_values[0]), 
                xytext=(10, 30), 
                textcoords='offset points',
                ha='left',
                va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # Annotate final surface values and point
    ax1.annotate(f'Final:\nS = {surface_values[-1]:.2f}\nr = {final_point[0]:.2f}\nh = {final_point[1]:.2f}', 
                xy=(len(surface_values)-1, surface_values[-1]), 
                xytext=(-10, -30), 
                textcoords='offset points',
                ha='right',
                va='top',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # Volume evolution plot
    ax2.plot(range(len(volume_values)), volume_values, 'g-', linewidth=2)
    ax2.scatter(range(len(volume_values)), volume_values, color='red', s=30)
    ax2.axhline(y=v_0, color='r', linestyle='--', label=f'Contrainte v_0 = {v_0}')
    ax2.set_title(f"Évolution du volume pendant l'optimisation ({method_name})")
    ax2.set_xlabel("Itération")
    ax2.set_ylabel("Volume")
    ax2.grid(True)
    ax2.legend()
    
    # Annotate initial volume values and point
    ax2.annotate(f'Initial:\nV = {volume_values[0]:.2f}\nr = {initial_point[0]:.2f}\nh = {initial_point[1]:.2f}', 
                xy=(0, volume_values[0]), 
                xytext=(10, 30), 
                textcoords='offset points',
                ha='left',
                va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # Annotate final volume values and point
    ax2.annotate(f'Final:\nV = {volume_values[-1]:.2f}\nr = {final_point[0]:.2f}\nh = {final_point[1]:.2f}', 
                xy=(len(volume_values)-1, volume_values[-1]), 
                xytext=(-10, -30), 
                textcoords='offset points',
                ha='right',
                va='top',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.tight_layout()
    plt.show()


def test_gradient_pas_fixe():
    x_0 = [5, 10, 1]  # (r_0, h_0, lambda_0)
    pas = 1e-7
    tolerance = 0.1
    it_max = 500
    
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
    
    plot_optimization_results(historique, "Gradient à pas fixe")


def test_gradient_pas_optimal():
    x_0 = [5, 10, 1]  # (r_0, h_0, lambda_0)
    tolerance = 0.1
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
    
    plot_optimization_results(historique, "Gradient à pas optimal")


def test_wolfe():
    x_0 = [5, 10, 1]  # (r_0, h_0, lambda_0)
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
    
    plot_optimization_results(historique, "Wolfe")


def test_newton():
    x_0 = [5, 10, 1]  # (r_0, h_0, lambda_0)
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
    
    plot_optimization_results(historique, "Newton")


def test_bfgs():
    """Test de la méthode Quasi-Newton BFGS"""
    x_0 = [5, 10, 1]  # (r_0, h_0, lambda_0)
    tolerance = 0.0001
    it_max = 1000
    
    solution, nb_iterations, historique = quasi_newton(x_0, tolerance, it_max)
    
    print("\nRésultats de l'optimisation (BFGS):")
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
    
    plot_optimization_results(historique, "BFGS (Quasi-Newton)")


if __name__ == "__main__":
    test_gradient_pas_fixe()
    test_gradient_pas_optimal()
    test_wolfe()
    test_newton()
    test_bfgs()
