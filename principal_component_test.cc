// Author: Mingcheng Chen (linyufly@gmail.com)

#include "math.h"
#include "util.h"

#include <cstdio>

int main() {
  double **org_vec = NULL;
  double *result = NULL;

  double vectors_1[3][2] = {{0.0, 0.0},
                            {1.0, -1.0},
                            {1.0, 1.0}};

  org_vec = create_matrix<double>(3, 2);

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 2; j++) {
      org_vec[i][j] = vectors_1[i][j];
    }
  }

  result = principal_component(org_vec, 3, 2);
  printf("%lf, %lf\n", result[0], result[1]);

  delete [] result;
  delete_matrix(org_vec);

  double vectors_2[3][3] = {{0.0, 0.0, 0.0},
                            {10.0, 1.0, 0.0},
                            {10.0, -1.0, 0.0}};

  org_vec = create_matrix<double>(3, 3);

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      org_vec[i][j] = vectors_2[i][j];
    }
  }

  result = principal_component(org_vec, 3, 3);
  printf("%lf, %lf, %lf\n", result[0], result[1], result[2]);

  delete [] result;
  delete_matrix(org_vec);

  return 0;
}
