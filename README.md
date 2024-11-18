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
2. Afficher les résultats de l'optimisation pour chaque méthode
3. Afficher des graphiques montrant pour chaque méthode :
   - L'évolution de la surface du cône en fonction du nombre d'itérations
   - L'évolution du volume du cône en fonction du nombre d'itérations
   - Une visualisation 3D du cône optimal
4. Afficher un tableau comparatif des performances des différentes méthodes

## Structure du projet

- `main.py` : Contient l'implémentation des méthodes d'optimisation et les fonctions de visualisation
- `README.md` : Documentation du projet

## Méthodes implémentées

### Gradient à pas fixe

- Utilise un pas constant pour la descente
- Plus simple mais convergence plus lente
- Paramètres par défaut : `pas=1E-7`, `tolerance=0.1`, `it_max=2000`

### Gradient à pas optimal

- Utilise la recherche linéaire de Wolfe pour déterminer le pas optimal
- Meilleure convergence que le pas fixe
- Paramètres par défaut : `tolerance=0.1`, `it_max=10000`

### Newton

- Utilise la matrice hessienne pour une convergence quadratique
- Convergence rapide mais calculs plus complexes
- Paramètres par défaut : `tolerance=0.0001`, `it_max=1000`

### Quasi-Newton (BFGS)

- Approxime la matrice hessienne pour réduire le coût de calcul
- Bon compromis entre vitesse et précision
- Paramètres par défaut : `tolerance=0.0001`, `it_max=500`
