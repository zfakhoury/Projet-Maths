# Projet-Maths

Ce projet implémente différentes méthodes d'optimisation numérique pour trouver les dimensions optimales d'un cône sous contrainte de volume. Les méthodes implémentées sont :

- Gradient à pas fixe
- Gradient à pas optimal (avec recherche linéaire par la règle de Wolfe)
- Méthode de Newton
- Méthode Quasi-Newton (BFGS)

## Dépendances

Le projet nécessite les bibliothèques Python suivantes :

- Python >= 3.8
- NumPy >= 1.20.0
- Matplotlib >= 3.4.0

## Installation et configuration

1. Assurez-vous d'avoir la version minimale de Python installée sur votre système
2. Installez les dépendances avec `pip install numpy matplotlib`
3. Naviguez vers le répértoire du projet dans le terminal
4. Exécutez le programme avec `python main.py`

## Sortie du programme

Le programme va :

1. Exécuter les 4 méthodes d'optimisation
2. Afficher dans le terminal les résultats de l'optimisation pour chaque méthode
3. Afficher dans une fenêtre des graphiques montrant pour chaque méthode :
   - L'évolution de la surface du cône en fonction du nombre d'itérations
   - L'évolution du volume du cône en fonction du nombre d'itérations
   - Une visualisation 3D du cône avec les dimensions optimales
4. Afficher dans le terminal un tableau comparatif des performances des différentes méthodes

## Structure du projet

- `main.py` : Code source contenant l'implémentation des méthodes d'optimisation
- `README.md` : Documentation du projet
- `.gitignore`: Fichier ignorant les fichiers intermédiaires/inutiles au programme

## Fonctions implémentées

### Calculs géométriques

- `surface_au_carre(r, h)` : Calcule le carré de la surface latérale du cône en fonction du rayon et de la hauteur
- `volume(r, h)` : Calcule le volume du cône en fonction du rayon et de la hauteur

### Fonctions d'optimisation

- `lagrangien(r, h, lambd)` : Calcule le Lagrangien du problème d'optimisation
- `grad_lagrangien(r, h, lambd)` : Calcule le gradient du Lagrangien
- `hessienne(r, h, lambd)` : Calcule la matrice hessienne du Lagrangien
- `norme(grad)` : Calcule la norme euclidienne d'un vecteur

### Recherche linéaire

- `wolfe(x_k, d_k)` : Implémente la recherche linéaire avec les conditions de Wolfe pour trouver le pas optimal
  - Utilise les paramètres $c_1=0.01$ et $c_2=0.99$
  - Retourne le pas optimal qui satisfait les conditions de Wolfe

## Fonctions d'algorithme d'optimisation

### Gradient à pas fixe

`gradient_pas_fixe(x_0, pas, tolerance, it_max)`

- Utilise un pas constant pour la descente
- Plus simple mais convergence plus lente
- Paramètres par défaut : `pas=1E-7`, `tolerance=0.001`, `it_max=2000`
- Arguments :
  - `x_0` : Point initial [$r_0$, $h_0$, $\lambda_0$]
  - `pas` : Taille du pas fixe (alpha)
  - `tolerance` : Critère d'arrêt pour la norme du gradient
  - `it_max` : Nombre maximum d'itérations
- À chaque itération :
  1. Calcule le gradient au point courant
  2. Met à jour la position avec un pas fixe dans la direction opposée au gradient

### Gradient à pas optimal

`gradient_pas_optimal(x_0, tolerance, it_max)`

- Utilise la recherche linéaire par la règle de Wolfe pour déterminer le pas optimal
- Meilleure convergence que le pas fixe
- Paramètres par défaut : `tolerance=0.001`, `it_max=1000`
- Arguments :
  - `x_0` : Point initial [$r_0$, $h_0$, $\lambda_0$]
  - `tolerance` : Critère d'arrêt pour la norme du gradient
  - `it_max` : Nombre maximum d'itérations
- À chaque itération :
  1. Calcule le gradient au point courant
  2. Détermine le pas optimal avec la méthode de Wolfe
  3. Met à jour la position avec le pas optimal

### Newton

`newton(x_0, tolerance, it_max)`

- Utilise la matrice hessienne pour une convergence quadratique
- Convergence rapide mais calculs plus complexes
- Paramètres par défaut : `tolerance=0.001`, `it_max=100`
- Arguments :
  - `x_0` : Point initial [$r_0$, $h_0$, $\lambda_0$]
  - `tolerance` : Critère d'arrêt pour la norme du gradient
  - `it_max` : Nombre maximum d'itérations
- À chaque itération :
  1. Calcule le gradient et la hessienne au point courant
  2. Résout le système linéaire pour obtenir la direction de descente
  3. Met à jour la position dans cette direction

### Quasi-Newton BFGS

`quasi_newton(x_0, tolerance, it_max)`

- Approxime la matrice hessienne pour réduire le coût de calcul
- Bon compromis entre vitesse et précision
- Paramètres par défaut : `tolerance=0.001`, `it_max=500`
- Arguments :
  - `x_0` : Point initial [$r_0$, $h_0$, $\lambda_0$]
  - `tolerance` : Critère d'arrêt pour la norme du gradient
  - `it_max` : Nombre maximum d'itérations
- À chaque itération :
  1. Calcule le gradient au point courant
  2. Met à jour l'approximation de l'inverse de la hessienne avec la formule BFGS
  3. Détermine la direction de descente et le pas optimal
  4. Met à jour la position

## Fonctions de visualisation

- `plot_optimization_results(historique, method_name)` : Génère les graphiques montrant l'évolution de la surface et du volume, ainsi qu'une visualisation 3D du cône optimal
- `compare_methods()` : Compare les performances des différentes méthodes d'optimisation et affiche un tableau récapitulatif

## Fonctions de tests individuels des méthodes

Chaque fonction de test affiche les résultats détaillés dans le terminal, montre l'évolution des paramètres à différentes étapes et génère les graphiques associés.

- `test_gradient_pas_fixe()` : Teste la méthode du gradient à pas fixe

- `test_gradient_pas_optimal()` : Teste la méthode du gradient à pas optimal

- `test_newton()` : Teste la méthode de Newton

- `test_quasi_newton()` : Teste la méthode Quasi-Newton BFGS

Pour chaque test, les informations affichées comprennent :

- Le point initial et final
- Le nombre d'itérations effectuées
- La norme finale du gradient
- L'historique de convergence à 5 étapes clés (0%, 25%, 50%, 75%, 100%). Par exemple, si le programme fait 200 itérations, il affichera le résultat des itérations 0, 50, 100, 150 et 200.
