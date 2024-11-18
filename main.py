import numpy as np
import matplotlib.pyplot as plt

pi = np.pi
v_0 = 300 # 300 cm^3 = 0.3 L (contrainte de volume modifiable)

# ---------------------------------------------------------------------------------------
#                                      SURFACE
# ---------------------------------------------------------------------------------------

def surface(r, h):
    """
    Calcule la surface latérale du cône.
    
    Args:
        r (float): Rayon du cône
        h (float): Hauteur du cône
    
    Returns:
        float: Surface latérale du cône
    """
    return pi**2 * r**2 * (r**2 + h**2)

# ---------------------------------------------------------------------------------------
#                                      VOLUME
# ---------------------------------------------------------------------------------------

def volume(r, h):
    """
    Calcule le volume du cône.
    
    Args:
        r (float): Rayon du cône
        h (float): Hauteur du cône
    
    Returns:
        float: Volume du cône
    """
    return pi/3 * r**2 * h

# ---------------------------------------------------------------------------------------
#                                   LAGRANGIEN
# ---------------------------------------------------------------------------------------

def lagrangien(r, h, lambd):
    """
    Calcule le Lagrangien du problème d'optimisation.
    
    Args:
        r (float): Rayon du cône
        h (float): Hauteur du cône
        lambd (float): Multiplicateur de Lagrange
    
    Returns:
        float: Valeur du Lagrangien
    """
    return surface(r, h) + lambd * (volume(r, h) - v_0)

# ---------------------------------------------------------------------------------------
#                              GRADIENT DU LAGRANGIEN
# ---------------------------------------------------------------------------------------

def grad_lagrangien(r, h, lambd):
    """
    Calcule le gradient du Lagrangien.
    
    Args:
        r (float): Rayon du cône
        h (float): Hauteur du cône
        lambd (float): Multiplicateur de Lagrange
    
    Returns:
        list: Gradient [∂L/∂r, ∂L/∂h, ∂L/∂λ]
    """
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
    """
    Calcule la norme euclidienne d'un vecteur.
    
    Args:
        grad (list): Vecteur dont on veut calculer la norme
    
    Returns:
        float: Norme euclidienne du vecteur
    """
    return np.sqrt(grad[0]**2 + grad[1]**2 + grad[2]**2)

# ---------------------------------------------------------------------------------------
#                                      HESSIENNE
# ---------------------------------------------------------------------------------------

def hessienne(r, h, lambd): 
    """
    Calcule la matrice hessienne du Lagrangien.
    
    Args:
        r (float): Rayon du cône
        h (float): Hauteur du cône
        lambd (float): Multiplicateur de Lagrange
    
    Returns:
        numpy.ndarray: Matrice hessienne 3x3
    """
    H = [[0, 0, 0], 
         [0, 0, 0],
         [0, 0, 0]]
    
    # Dérivées secondes par rapport à r
    H[0][0] = 12 * pi**2 * r**2 + 2 * pi**2 * h**2 + (2/3) * pi * h * lambd
    
    # Dérivées croisées r,h
    H[0][1] = H[1][0] = 4 * pi**2 * r * h + (2/3) * pi * r * lambd
    
    # Dérivées croisées r, lambda
    H[0][2] = H[2][0] = (2/3) * pi * r * h
    
    # Dérivées secondes par rapport à h
    H[1][1] = 2 * pi**2 * r**2
    
    # Dérivées croisées h, lambda
    H[1][2] = H[2][1] = (1/3) * pi * r**2
    
    # Dérivées secondes par rapport à lambda
    H[2][2] = 0
    
    return np.array(H)  # Convertir en array numpy pour faciliter la manipulation


# ---------------------------------------------------------------------------------------
#                                      WOLFE
# ---------------------------------------------------------------------------------------

def wolfe(x_k, d_k):
    """
    Recherche du pas optimal avec les conditions de Wolfe.
    
    Args:
        x_k (list): Point courant [r, h, λ]
        d_k (list): Direction de descente
    
    Returns:
        float: Pas optimal satisfaisant les conditions de Wolfe
    
    Note:
        Utilise les paramètres c1=0.01 et c2=0.99 pour les conditions de Wolfe
        et un maximum de 100 itérations
    """

    cond1 = 0
    cond2 = 0

    c1 = 0.01
    c2 = 0.99

    it_max = 100

    x_k1 = [0, 0, 0]

    gradX_k = grad_lagrangien(x_k[0], x_k[1], x_k[2])
    lagX_k = lagrangien(x_k[0], x_k[1], x_k[2])

    # Calcul du produit scalaire initial
    PS_k = d_k[0]*gradX_k[0] + d_k[1]*gradX_k[1] + d_k[2]*gradX_k[2]

    a_min = 0
    a_max = 100

    # a_k = (a_min + a_max) / 2
    a_k = 1e-6

    it = 0
    historique = {'points': [x_k.copy()], 'normes': [norme(gradX_k)]}

    while ((cond1 + cond2) < 2) and (it < it_max):
        # Calcul du nouveau point
        x_k1[0] = x_k[0] + a_k * d_k[0]
        x_k1[1] = x_k[1] + a_k * d_k[1]
        x_k1[2] = x_k[2] + a_k * d_k[2]

        lagX_k1 = lagrangien(x_k1[0], x_k1[1], x_k1[2])

        gradX_k1 = grad_lagrangien(x_k1[0], x_k1[1], x_k1[2])

        # Calcul du nouveau produit scalaire
        PS_k1 = d_k[0]*gradX_k1[0] + d_k[1]*gradX_k1[1] + d_k[2]*gradX_k1[2]

        if lagX_k1 > (lagX_k + c1 * a_k * PS_k):
            cond1 = 0
            a_max = a_k
            a_k = (a_min + a_max) / 2
        else:
            cond1 = 1

        if -PS_k1 > -c2 * PS_k:
            cond2 = 0
            if a_max < 100:
                a_min = a_k
                a_k = (a_min + a_max) / 2
            else:
                a_k = a_k * 2
        else:
            cond2 = 1
        
        it += 1
        historique['points'].append(x_k1.copy())
        historique['normes'].append(norme(gradX_k1))
    
    return a_k


# ---------------------------------------------------------------------------------------
#                               GRADIENT À PAS FIXE
# ---------------------------------------------------------------------------------------

def gradient_pas_fixe(x_0, pas, tolerance, it_max):
    """
    Méthode du gradient à pas fixe pour minimiser le Lagrangien.
    
    Args:
        x_0 (list): Point initial [r₀, h₀, λ₀]
        pas (float): Taille du pas fixe (alpha)
        tolerance (float): Critère d'arrêt pour la norme du gradient
        it_max (int): Nombre maximum d'itérations
    
    Returns:
        tuple: (
            list: Solution optimale [r*, h*, λ*],
            int: Nombre d'itérations effectuées,
            dict: Historique des points et des normes du gradient
        )
    """

    # Point initial
    x = x_0.copy()
    gradX = grad_lagrangien(x[0], x[1], x[2])
    historique = {'points': [x_0.copy()], 'normes': [norme(gradX)]}
    it = 0
    
    while norme(gradX) > tolerance and it < it_max:
        # Direction de descente d_k = -gradient
        d_k = [-gradX[0], -gradX[1], -gradX[2]]
        
        # Mise à jour: x_{k+1} = x_k + alpha * d_k
        #                      = x_k - alpha * gradient
        x[0] = x[0] + pas * d_k[0]
        x[1] = x[1] + pas * d_k[1]
        x[2] = x[2] + pas * d_k[2]
        
        # Calcul du nouveau gradient
        gradX = grad_lagrangien(x[0], x[1], x[2])
        
        it += 1
        historique['points'].append(x.copy())
        historique['normes'].append(norme(gradX))
    
    return x, it, historique


# ---------------------------------------------------------------------------------------
#                               GRADIENT À PAS OPTIMAL
# ---------------------------------------------------------------------------------------

def gradient_pas_optimal(x_0, tolerance, it_max):
    """
    Méthode du gradient à pas optimal avec recherche linéaire de Wolfe.
    
    Args:
        x_0 (list): Point initial [r₀, h₀, λ₀]
        tolerance (float): Critère d'arrêt pour la norme du gradient
        it_max (int): Nombre maximum d'itérations
    
    Returns:
        tuple: (
            list: Solution optimale [r*, h*, λ*],
            int: Nombre d'itérations effectuées,
            dict: Historique des points et des normes du gradient
        )
    """

    # Point initial
    x = x_0.copy()
    gradX = grad_lagrangien(x[0], x[1], x[2])
    historique = {'points': [x_0.copy()], 'normes': [norme(gradX)]}
    it = 0

    while norme(gradX) > tolerance and it < it_max:
        # Direction de descente d_k = -gradient
        direction = [-gradX[0], -gradX[1], -gradX[2]]
        
        pas = wolfe(x, direction)
        
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
#                                      NEWTON
# ---------------------------------------------------------------------------------------

def newton(x_0, tolerance, it_max):
    """
    Méthode de Newton pour minimiser le Lagrangien.
    
    Args:
        x_0 (list): Point initial [r₀, h₀, λ₀]
        tolerance (float): Critère d'arrêt pour la norme du gradient
        it_max (int): Nombre maximum d'itérations
    
    Returns:
        tuple: (
            list: Solution optimale [r*, h*, λ*],
            int: Nombre d'itérations effectuées,
            dict: Historique des points et des normes du gradient
        )
    """

    # Point initial
    x = x_0.copy()
    gradX = grad_lagrangien(x[0], x[1], x[2])
    historique = {'points': [x_0.copy()], 'normes': [norme(gradX)]}
    
    it = 0
    
    while norme(gradX) > tolerance and it < it_max:
        # Calcul de la hessienne
        H = hessienne(x[0], x[1], x[2])
        H = np.linalg.inv(H) # matrice inverse

        # Direction de descente
        d_k = -H @ gradX

        # Mise à jour: x_{k+1} = x_k + d_k
        x[0] = x[0] + d_k[0]
        x[1] = x[1] + d_k[1]
        x[2] = x[2] + d_k[2]
        
        # Nouveau gradient
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
    Méthode de quasi-Newton BFGS pour minimiser le Lagrangien.
    
    Args:
        x_0 (list): Point initial [r₀, h₀, λ₀]
        tolerance (float): Critère d'arrêt pour la norme du gradient
        it_max (int): Nombre maximum d'itérations
    
    Returns:
        tuple: (
            numpy.ndarray: Solution optimale [r*, h*, λ*],
            int: Nombre d'itérations effectuées,
            dict: Historique des points et des normes du gradient
        )
    """
    x = np.array(x_0, dtype=float)
    n = len(x)
    
    # Gradient initial
    gradX = np.array(grad_lagrangien(x[0], x[1], x[2]))
    
    # Initialisation de l'inverse de la hessienne
    inv_hessienne = np.linalg.inv(hessienne(x[0], x[1], x[2]))
    
    historique = {'points': [x.tolist()], 'normes': [norme(gradX)]}
    it = 0
    
    while norme(gradX) > tolerance and it < it_max:
        # Calcul de la direction de recherche
        d_k = -np.dot(inv_hessienne, gradX)
        
        # Recherche linéaire
        alpha = wolfe(x, d_k)
        
        x_old = x.copy()
        grad_old = gradX.copy()
        
        # Mise à jour: x_{k+1} = x_k + alpha * d_k
        x = x + alpha * d_k
        
        # Nouveau gradient
        gradX = np.array(grad_lagrangien(x[0], x[1], x[2]))
        
        # Mise à jour de l'inverse de la hessienne avec la formule de BFGS
        s_k = x - x_old
        y_k = gradX - grad_old
        
        rho_k = 1.0 / np.dot(y_k, s_k)

        v1 = np.eye(n) - rho_k * np.outer(s_k, y_k)
        v2 = np.eye(n) - rho_k * np.outer(y_k, s_k)
        
        inv_hessienne = np.dot(v1, np.dot(inv_hessienne, v2)) + rho_k * np.outer(s_k, s_k)
        
        it += 1
        historique['points'].append(x.tolist())
        historique['normes'].append(norme(gradX))
    
    return x, it, historique


# ---------------------------------------------------------------------------------------
#                                     GRAPHIQUES
# ---------------------------------------------------------------------------------------

def plot_optimization_results(historique, method_name):
    """
    Trace les graphiques d'évolution de la surface et du volume pendant l'optimisation.
    
    Args:
        historique (dict): Dictionnaire contenant l'historique des points et des normes
        method_name (str): Nom de la méthode d'optimisation utilisée
    
    Returns:
        None: Affiche les graphiques avec matplotlib
    """

    # Calculer les valeurs de surface et de volume pour chaque point
    surface_values = [surface(point[0], point[1]) for point in historique['points']]
    volume_values = [volume(point[0], point[1]) for point in historique['points']]
    
    # Point initial (donné en paramètres) et point final (solution retournée par la méthode choisie)
    initial_point = historique['points'][0]
    final_point = historique['points'][-1]
    
    # Créer deux sous-graphiques
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Créer des valeurs entières pour l'axe x 
    # (pour n'afficher que des entiers sur l'axe des itérations)
    iterations = list(range(len(surface_values)))
    
    # Graphique de l'évolution de la surface en fonction du nombre d'itérations
    ax1.plot(iterations, surface_values, 'b-', linewidth=2)
    ax1.scatter(iterations, surface_values, color='red', s=30)
    ax1.set_title(f"Évolution de la surface pendant l'optimisation ({method_name})")
    ax1.set_xlabel("Itération")
    ax1.set_ylabel("Surface")
    ax1.grid(True)
    
    # Annotations pour le graphique de la surface
    # Point initial
    ax1.annotate(f'Initial:\nS = {surface_values[0]:.2f}\nr = {initial_point[0]:.2f}\nh = {initial_point[1]:.2f}', 
                xy=(0, surface_values[0]), 
                xytext=(10, 30), 
                textcoords='offset points',
                ha='left',
                va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # Point final
    ax1.annotate(f'Final:\nS = {surface_values[-1]:.2f}\nr = {final_point[0]:.2f}\nh = {final_point[1]:.2f}', 
                xy=(len(surface_values)-1, surface_values[-1]), 
                xytext=(-10, -30), 
                textcoords='offset points',
                ha='right',
                va='top',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # Graphique de l'évolution du volume en fonction du nombre d'itérations
    ax2.plot(iterations, volume_values, 'g-', linewidth=2)
    ax2.scatter(iterations, volume_values, color='red', s=30)
    ax2.axhline(y=v_0, color='r', linestyle='--', label=f'Contrainte v_0 = {v_0}')
    ax2.set_title(f"Évolution du volume pendant l'optimisation ({method_name})")
    ax2.set_xlabel("Itération")
    ax2.set_ylabel("Volume")
    ax2.grid(True)
    ax2.legend()
    
    # Annotations pour le graphique du volume
    # Point initial
    ax2.annotate(f'Initial:\nV = {volume_values[0]:.2f}\nr = {initial_point[0]:.2f}\nh = {initial_point[1]:.2f}', 
                xy=(0, volume_values[0]), 
                xytext=(10, 30), 
                textcoords='offset points',
                ha='left',
                va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # Point final
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


# ---------------------------------------------------------------------------------------
#                                      TESTS
# ---------------------------------------------------------------------------------------

def test_gradient_pas_fixe():
    x_0 = [10, 10, 1]  # (r_0, h_0, lambda_0)
    pas = 1E-7
    tolerance = 0.1
    it_max = 2000
    
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
    x_0 = [10, 10, 1]  # (r_0, h_0, lambda_0)
    tolerance = 0.1
    it_max = 10000
    
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


def test_newton():
    x_0 = [10, 10, 1]  # (r_0, h_0, lambda_0)
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


def test_quasi_newton():
    x_0 = [10, 10, 1]  # (r_0, h_0, lambda_0)
    tolerance = 0.0001
    it_max = 500
    
    solution, nb_iterations, historique = quasi_newton(x_0, tolerance, it_max)
    
    print("\nRésultats de l'optimisation (Quasi-Newton):")
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
    
    plot_optimization_results(historique, "Quasi-Newton")


# ---------------------------------------------------------------------------------------
#                                      MAIN
# ---------------------------------------------------------------------------------------

if __name__ == "__main__":
    # test_gradient_pas_fixe()
    test_gradient_pas_optimal()
    test_newton()
    test_quasi_newton()
    