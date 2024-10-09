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

double* Gradient(double r, double h, double lambda) {
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
		gradient = Gradient(x[0], x[1], x[2]);
		
		// Mise à jour des valeurs du gradient
		x[0] = x[0] - pas * gradient[0];
		printf("i=%d: x[%d]=%.3f \n", i, 0, x[0]);
		
		x[1] = x[1] - pas * gradient[1];
		printf("i=%d: x[%d]=%.3f \n", i, 1, x[1]);
		
		x[2] = x[2] - pas * gradient[2];	
		printf("i=%d: x[%d]=%.3f \n", i, 2, x[2]);
		
		norme = NormeGrad(gradient);

		free(gradient);  // Libérer la mémoire allouée pour le gradient
	} while (norme > epsilon);

	return x;
	free(x);
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
	
	
	return gradient;
}



int main() {
	double* x0 = (double*)malloc(3 * sizeof(double));
	x0[0] = 1;
	x0[1] = 1;
	x0[2] = 1;
	
	GradPasFixe(x0, 1, 0.001);
	return 0;
}



// NOTES
// double a; 16 chiffres apres la virgule
// float b; 8 chiffres apres la virgule
// *b pointeur (vecteur)
// **c matrice