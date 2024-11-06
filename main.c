#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define square(x) ((x)*(x))
#define cube(x) ((x)*(x)*(x))

const double pi = 3.14159;
const double v0 = 1.0;

double Lagrangien(double r, double h, double lambda) {    
	return square(pi) * square(r) * (square(r) + square(h)) + lambda * (pi/3 * square(r) * h - v0);
}

//---------------------------------------------------------------------------------------
//                                Calc de gradient
//---------------------------------------------------------------------------------------

double* GradLag(double r, double h, double lambda) {
	double *grad;
	
	grad = (double*) malloc(3 * sizeof(double));

	// par rapport a r
	grad[0] = 4*cube(r)*square(pi) + 2*square(pi)*square(h)*r + 2*r*lambda*(pi/3)*h;
	// par rapport a h
	grad[1] = 2*h*square(pi)*square(r) + (pi/3)*square(r)*lambda;
	// par rapport a lambda
	grad[2] = (pi/3)*square(r)*h - v0;

	return grad;
	free(grad);
} 

//---------------------------------------------------------------------------------------
//                                Calc de la norme du gradient
//---------------------------------------------------------------------------------------

double NormeGrad(double *grad) {
	return sqrt(square(grad[0]) + square(grad[1]) + square(grad[2]));
}

//---------------------------------------------------------------------------------------
//                                Methode de gradient à pas fixe
//---------------------------------------------------------------------------------------

double* GradPasFixe(double* x0, double pas, double epsilon) {
	double* x = (double*) malloc(3 *sizeof(double));
	x[0] = x0[0];
	x[1] = x0[1];
	x[2] = x0[2];
	
	double* gradient = (double*) malloc(3 *sizeof(double));
	double norme;
	int i = 0;
	// "do" pour executer le bloc au moins 1 fois (permet d'assigner a la premiere iteration les valeurs initiales du gradient et de la norme)
	do {
		i = i + 1;
		// Mise à jour du gradient
		gradient = GradLag(x[0], x[1], x[2]);
		
		// Mise à jour des valeurs du gradient
		x[0] = x[0] - pas * gradient[0];
		printf("i=%d: x[%d]=%.3f \n", i, 0, x[0]);
		
		x[1] = x[1] - pas * gradient[1];
		printf("i=%d: x[%d]=%.3f \n", i, 1, x[1]);
		
		x[2] = x[2] - pas * gradient[2];	
		printf("i=%d: x[%d]=%.3f \n", i, 2, x[2]);
		
		norme = NormeGrad(gradient);

		// appeler le lagrangien puis afficher les valeurs de r, h et lambda

		// free(gradient);  // Libérer la mémoire allouée pour le gradient
	} while ((norme > epsilon) && (i < 1000));
	// pas fixe ne converge pas

	return x;
	free(x);
}

//---------------------------------------------------------------------------------------
//                                Methode de Wolfe
//---------------------------------------------------------------------------------------

// ... et grad(xk) sont des vecteurs qu'il fqut initialiser et allouer de la memoire
double Wolfe(double* xk, double* dk) {
	int cond1 = 0;
	int cond2 = 0;
	int i = 0;
	int i_max = 1000;

	double Lag_xk = Lagrangien(xk[0], xk[1], xk[2]);
	double* Grad_xk = GradLag(xk[0], xk[1], xk[2]);
	double Ps_k = dk[0] * Grad_xk[0] + dk[1] * Grad_xk[1] + dk[2] * Grad_xk[2];

	double* xk1 = xk;
	double* Grad_xk1;
	double Lag_xk1;
	double Ps_k1;
	

	int alpha_min = 0;
	int alpha_max = 100;
	double alpha_k = (alpha_min + alpha_max)/2;

	while (((cond1 + cond2) < 2) && (i < i_max)) {
		xk1[0] = xk[0] + alpha_k * dk[0];
		xk1[1] = xk[1] + alpha_k * dk[1];
		xk1[2] = xk[2] + alpha_k * dk[2];

		Lag_xk1 = Lagrangien(xk1[0], xk1[1], xk1[2]);
		Grad_xk1 = GradLag(xk1[0], xk1[1], xk1[2]);
		Ps_k1 = dk[0] * Grad_xk1[0] + dk[1] * Grad_xk1[1] + dk[2] * Grad_xk1[2];

		if (Lag_xk1 > (Lag_xk + 0.1 * alpha_k * Ps_k)) {
			cond1 = 0;
			alpha_max = alpha_k;
			alpha_k = (alpha_min + alpha_max)/2;
		} else {
			cond1 = 1;
		} 

		if (-Ps_k1 > -0.99 * Ps_k) {
			cond2 = 0;
			alpha_min = alpha_k;
			alpha_k = (alpha_min + alpha_max)/2;
		} else {
			cond2 = 0;
		}

		
	}

	return alpha_k;
}

//---------------------------------------------------------------------------------------
//                                Methode de gradient à pas optimal
//---------------------------------------------------------------------------------------

double* GradPasOptimal(double* x0, double pas, double epsilon) {
	double* x = (double*) malloc(3 *sizeof(double));
	x[0] = x0[0];
	x[1] = x0[1];
	x[2] = x0[2];
	
	double* gradient = (double*) malloc(3 *sizeof(double));
	double* u = (double*) malloc(3 *sizeof(double));
	double norme;
	int i = 0;

	do {
		i = i + 1;
		// Mise à jour du gradient negatif
		u = GradLag(-x[0], -x[1], -x[2]);
		
		// Mise à jour des valeurs du gradient
		x[0] = x[0] + pas * u[0];
		printf("i=%d: x[%d]=%.3f \n", i, 0, x[0]);

		x[1] = x[1] + pas * u[1];
		printf("i=%d: x[%d]=%.3f \n", i, 1, x[1]);

		x[2] = x[2] + pas * u[2];	
		printf("i=%d: x[%d]=%.3f \n", i, 2, x[2]);

		norme = NormeGrad(gradient);

		// free(gradient);  // Libérer la mémoire allouée pour le gradient
	} while (norme > epsilon);
	
	return gradient;
}


int main() {
	double* x0 = (double*)malloc(3 * sizeof(double));
	x0[0] = 1;
	x0[1] = 1;
	x0[2] = 1;

	double* dk = (double*) malloc(3 * sizeof(double));
	dk[0] = 1;
	dk[1] = 1;
	dk[2] = 1;
	
	// GradPasFixe(x0, 0.001, 0.001);
	// GradPasOptimal(x0, 1, 0.001);

	Wolfe(x0, dk);
	return 0;
}


// NOTES
// double a; 16 chiffres apres la virgule
// float b; 8 chiffres apres la virgule
// *b pointeur (vecteur)
// **c matrice

// meilleure methode parmi les 4
// methode de wolfe pour calculer alpha (aka le pas)

// 

