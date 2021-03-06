// Author: Mingcheng Chen (linyufly@gmail.com)

#ifndef MATH_H_
#define MATH_H_

// Matrix A is m * n. Matrix B is n * p.
double **matrix_matrix_multiplication(double **a, double **b,
                                      int m, int n, int p);

// Matrix A is n * m.
double **transpose(double **a, int n, int m);

// vectors has n rows and k columns. Each row is a k-dimension vector.
// The result is a k-dimension vector.
double *principal_component(double **vectors, int n, int k);

double determinant_3(double **matrix);

double interpolate(double start, double finish, double lambda);

#endif  // MATH_H_
