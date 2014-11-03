// Author: Mingcheng Chen (linyufly@gmail.com)

#include "c_ridge_extractor.h"
#include "util.h"
#include "math.h"
#include "marchingCubesTable.h"

#include <vtkStructuredPoints.h>
#include <vtkPolyData.h>
#include <vtkGradientFilter.h>
#include <vtkSmartPointer.h>
#include <vtkPointData.h>
#include <vtkDataArray.h>
#include <vtkCellArray.h>
#include <vtkPoints.h>
#include <vtkMath.h>
#include <vtkDoubleArray.h>

#include <cstring>
#include <cmath>

#include <algorithm>

namespace {

const int kVertexList[8][3] = {
    {0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
    {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}
};

const int kEdgeList[12][2] = {
    {0, 1}, {1, 2}, {2, 3}, {3, 0},
    {4, 5}, {5, 6}, {6, 7}, {7, 4},
    {0, 4}, {1, 5}, {2, 6}, {3, 7}
};

double dot_product_3d(double *a, double *b) {
  double result = 0.0;
  for (int i = 0; i < 3; i++) {
    result += a[i] * b[i];
  }
  return result;
}

// eigen_vector[0..2] stores the first eigen-vector.
// eigen_vector[3..5] stores the second eigen-vector.
// eigen_vector[6..8] stores the third eigen-vector.
void get_eigen_vector(double **cauchy_green, double *eigen_vector, double *eigen_value) {
  double **f_trans = transpose(cauchy_green, 3, 3);
  double **tensor = matrix_matrix_multiplication(f_trans, cauchy_green, 3, 3, 3);

  double **eigen_vectors = create_matrix<double>(3, 3);
  vtkMath::Jacobi(tensor, eigen_value, eigen_vectors);

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      eigen_vector[i * 3 + j] = eigen_vectors[j][i];
    }
  }

  delete_matrix(eigen_vectors);
  delete_matrix(f_trans);
  delete_matrix(tensor);
}

// e1 is the eigen vector of the largest eigen value.
// l1 is the largest eigen value.
void get_e1(double **cauchy_green, double *e1, double *l1) {
  double **f_trans = transpose(cauchy_green, 3, 3);
  double **tensor = matrix_matrix_multiplication(f_trans, cauchy_green, 3, 3, 3);

  double *eigen_values = new double[3];
  double **eigen_vectors = create_matrix<double>(3, 3);
  vtkMath::Jacobi(tensor, eigen_values, eigen_vectors);

  *l1 = eigen_values[0];

  for (int i = 0; i < 3; i++) {
    e1[i] = eigen_vectors[i][0];
  }

  delete [] eigen_values;
  delete_matrix(eigen_vectors);
  delete_matrix(f_trans);
  delete_matrix(tensor);
}

// e3 is the eigen vector of the smallest eigen value.
// l3 is the smallest eigen value.
void get_e3(double **hessian, double *e3, double *l3) {
  double *eigen_values = new double[3];
  double **eigen_vectors = create_matrix<double>(3, 3);
  vtkMath::Jacobi(hessian, eigen_values, eigen_vectors);

  *l3 = eigen_values[2];

  for (int i = 0; i < 3; i++) {
    e3[i] = eigen_vectors[i][2];
  }

  delete [] eigen_values;
  delete_matrix(eigen_vectors);
}

bool check_consistency(double *vec_1, double *vec_2) {
  double standard = fabs(dot_product_3d(vec_1, vec_2));

  for (int i = 1; i < 3; i++) {
    if (fabs(dot_product_3d(vec_1, vec_2 + i * 3)) > standard ||
        fabs(dot_product_3d(vec_2, vec_1 + i * 3)) > standard) {
      return false;
    }
  }

  return true;
}

bool check_consistency(double eigen_vectors[2][2][2][9]) {
  for (int x = 0; x < 2; x++) {
    for (int y = 0; y < 2; y++) {
      for (int z = 0; z < 2; z++) {
        if (x == 0 && !check_consistency(eigen_vectors[x][y][z], eigen_vectors[x + 1][y][z])) {
          return false;
        }
        if (y == 0 && !check_consistency(eigen_vectors[x][y][z], eigen_vectors[x][y + 1][z])) {
          return false;
        }
        if (z == 0 && !check_consistency(eigen_vectors[x][y][z], eigen_vectors[x][y][z + 1])) {
          return false;
        }
      }
    }
  }
  return true;
}

void get_major_eigen_vectors(double eigen_vectors[2][2][2][9], double eigen_values[2][2][2][3],
                             double e1[2][2][2][3]) {
  double **vectors = create_matrix<double>(24, 3);
  int count = 0;

  for (int x = 0; x < 2; x++) {
    for (int y = 0; y < 2; y++) {
      for (int z = 0; z < 2; z++) {
        for (int e = 0; e < 3; e++) {
          for (int d = 0; d < 3; d++) {
            vectors[count][d] = eigen_vectors[x][y][z][e * 3 + d] * eigen_values[x][y][z][e];
          }
          count++;
        }
      }
    }
  }

  double *principal = principal_component(vectors, 24, 3);

  for (int x = 0; x < 2; x++) {
    for (int y = 0; y < 2; y++) {
      for (int z = 0; z < 2; z++) {
        for (int d = 0; d < 3; d++) {
          e1[x][y][z][d] = principal[d];
        }
      }
    }
  }

  delete [] principal;
  delete_matrix(vectors);
}

/*
void get_major_eigen_vectors(double eigen_vectors[2][2][2][9], double eigen_values[2][2][2][3],
                             double e1[2][2][2][3]) {
  int best_p, best_e;
  double best_value = -1.0;

  double std_vec[3], tmp_vec[3];
  for (int p = 0; p < 8; p++) {
    for (int e = 0; e < 2; e++) {
      for (int i = 0; i < 3; i++) {
        std_vec[i] = eigen_vectors[!!(p & 1)][!!(p & 2)][!!(p & 4)][e * 3 + i];
      }
      double sum = 0.0;
      for (int p2 = 0; p2 < 8; p2++) {
        int x = !!(p2 & 1);
        int y = !!(p2 & 2);
        int z = !!(p2 & 4);
        for (int e2 = 0; e2 < 3; e2++) {
          for (int i = 0; i < 3; i++) {
            tmp_vec[i] = eigen_vectors[x][y][z][e2 * 3 + i];
          }
          sum += fabs(dot_product_3d(std_vec, tmp_vec)) * eigen_values[x][y][z][e2];
        }
      }
      if (sum > best_value) {
        best_p = p;
        best_e = e;
        best_value = sum;
      }
    }
  }

  for (int i = 0; i < 3; i++) {
    std_vec[i] = eigen_vectors[!!(best_p & 1)][!!(best_p & 2)][!!(best_p & 4)][best_e * 3 + i];
  }

  for (int p = 0; p < 8; p++) {
    int x = !!(p & 1);
    int y = !!(p & 2);
    int z = !!(p & 4);
    int be;
    double ms = -1.0;
    for (int e = 0; e < 3; e++) {
      double sim = fabs(dot_product_3d(std_vec, eigen_vectors[x][y][z] + e * 3));
      if (ms < sim) {
        be = e;
        ms = sim;
      }
    }
    for (int i = 0; i < 3; i++) {
      e1[x][y][z][i] = eigen_vectors[x][y][z][be * 3 + i];
    }
  }
}
*/

/*
struct Edge {
  int x1, y1, z1, e1, x2, y2, z2, e2;
  double similarity;

  Edge() {
  }

  Edge(int x1, int y1, int z1, int e1, int x2, int y2, int z2, int e2, double similarity) :
      x1(x1), y1(y1), z1(z1), e1(e1), x2(x2), y2(y2), z2(z2), e2(e2), similarity(similarity) {
  }
};

bool operator < (const Edge &a, const Edge &b) {
  return a.similarity > b.similarity;
}

void add_edges(double vectors[2][2][2][9], int x1, int y1, int z1, int x2, int y2, int z2, Edge *edges, int *num_edges) {
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      /// DEBUG ///
      printf("i, j = %d, %d\n", i, j);
      double *vec_1 = vectors[x1][y1][z1] + i * 3;
      double *vec_2 = vectors[x2][y2][z2] + j * 3;
      Edge edge(x1, y1, z1, i, x2, y2, z2, j, fabs(dot_product_3d(vec_1, vec_2)));
      edges[*num_edges] = edge;
      (*num_edges)++;
      printf("--*num_edges = %d\n", *num_edges);
    }
  }
}

void find_father(int father[2][2][2][3][4], int x, int y, int z, int e, int *fx, int *fy, int *fz, int *fe) {
  int *imm_fa = father[x][y][z][e];
  if (imm_fa[0] == x && imm_fa[1] == y && imm_fa[2] == z && imm_fa[3] == e) {
    *fx = x;
    *fy = y;
    *fz = z;
    *fe = e;
    return;
  }
  find_father(father, imm_fa[0], imm_fa[1], imm_fa[2], imm_fa[3], fx, fy, fz, fe);
  imm_fa[0] = *fx;
  imm_fa[1] = *fy;
  imm_fa[2] = *fz;
  imm_fa[3] = *fe;
}

void get_major_eigen_vectors(double eigen_vectors[2][2][2][9], double eigen_values[2][2][2][3],
                             double e1[2][2][2][3]) {
  static Edge edges[8 * 3 * 3 * 3 / 2];
  int num_edges = 0;

  /// DEBUG ///
  printf("before collecting edges\n");

  for (int x = 0; x < 2; x++) {
    for (int y = 0; y < 2; y++) {
      for (int z = 0; z < 2; z++) {
        if (x == 0) {
          add_edges(eigen_vectors, x, y, z, x + 1, y, z, edges, &num_edges);
        }
        if (y == 0) {
          add_edges(eigen_vectors, x, y, z, x, y + 1, z, edges, &num_edges);
        }
        if (z == 0) {
          add_edges(eigen_vectors, x, y, z, x, y, z + 1, edges, &num_edges);
        }

        printf("- num_edges = %d\n", num_edges);
      }
    }
  }

  /// DEBUG ///
  printf("num_edges = %d\n", num_edges);

  std::sort(edges, edges + num_edges);
  
  static int father[2][2][2][3][4];  // x, y, z, e, dim
  static int color[2][2][2][3];
  int color_cnt = 0;
  for (int x = 0; x < 2; x++) {
    for (int y = 0; y < 2; y++) {
      for (int z = 0; z < 2; z++) {
        for (int e = 0; e < 3; e++) {
          father[x][y][z][e][0] = x;
          father[x][y][z][e][1] = y;
          father[x][y][z][e][2] = z;
          father[x][y][z][e][3] = e;

          color[x][y][z][e] = 1 << color_cnt;
        }
        color_cnt++;
      }
    }
  }

  /// DEBUG ///
  printf("after init father\n");

  for (int edge_idx = 0; edge_idx < num_edges; edge_idx++) {
    int fx1, fy1, fz1, fe1;
    int fx2, fy2, fz2, fe2;
    find_father(father, edges[edge_idx].x1, edges[edge_idx].y1, edges[edge_idx].z1, edges[edge_idx].e1, &fx1, &fy1, &fz1, &fe1);
    find_father(father, edges[edge_idx].x2, edges[edge_idx].y2, edges[edge_idx].z2, edges[edge_idx].e2, &fx2, &fy2, &fz2, &fe2);
    int color1 = color[fx1][fy1][fz1][fe1];
    int color2 = color[fx2][fy2][fz2][fe2];
    if (color1 & color2) {
      continue;
    }
    father[fx2][fy2][fz2][fe2][0] = fx1;
    father[fx2][fy2][fz2][fe2][1] = fy1;
    father[fx2][fy2][fz2][fe2][2] = fz1;
    father[fx2][fy2][fz2][fe2][3] = fe1;
    color[fx1][fy1][fz1][fe1] = color1 | color2;
  }

  /// DEBUG ///
  for (int x = 0; x < 2; x++) {
    for (int y = 0; y < 2; y++) {
      for (int z = 0; z < 2; z++) {
        for (int e = 0; e < 3; e++) {
          if (father[x][y][z][e][0] != x ||
              father[x][y][z][e][1] != y ||
              father[x][y][z][e][2] != z ||
              father[x][y][z][e][3] != e) {
            continue;
          }
          printf("color: %d\n", color[x][y][z][e]);
        }
      }
    }
  }

  static double sum_eigenvalue[2][2][2][3];
  for (int x = 0; x < 2; x++) {
    for (int y = 0; y < 2; y++) {
      for (int z = 0; z < 2; z++) {
        for (int e = 0; e < 3; e++) {
          sum_eigenvalue[x][y][z][e] = 0.0;
        }
      }
    }
  }

  for (int x = 0; x < 2; x++) {
    for (int y = 0; y < 2; y++) {
      for (int z = 0; z < 2; z++) {
        for (int e = 0; e < 3; e++) {
          int fx, fy, fz, fe;
          find_father(father, x, y, z, e, &fx, &fy, &fz, &fe);

          if (color[fx][fy][fz][fe] != (1 << 8) - 1) {
            printf("invalid code: %d\n", color[fx][fy][fz][fe]);
            report_error("Invalid set\n");
          }

          sum_eigenvalue[fx][fy][fz][fe] += eigen_values[x][y][z][e];
        }
      }
    }
  }

  int bx = 0, by = 0, bz = 0, be = 0;  // best x, y, z, and e
  for (int x = 0; x < 2; x++) {
    for (int y = 0; y < 2; y++) {
      for (int z = 0; z < 2; z++) {
        for (int e = 0; e < 3; e++) {
          if (sum_eigenvalue[x][y][z][e] > sum_eigenvalue[bx][by][bz][be]) {
            bx = x;
            by = y;
            bz = z;
            be = e;
          }
        }
      }
    }
  }

  for (int x = 0; x < 2; x++) {
    for (int y = 0; y < 2; y++) {
      for (int z = 0; z < 2; z++) {
        for (int e = 0; e < 3; e++) {
          int fx, fy, fz, fe;
          find_father(father, x, y, z, e, &fx, &fy, &fz, &fe);
          if (fx != bx || fy != by || fz != bz || fe != be) {
            continue;
          }
          for (int idx = 0; idx < 3; idx++) {
            e1[x][y][z][idx] = eigen_vectors[x][y][z][e * 3 + idx];
          }
        }
      }
    }
  } 
}
*/

/*
void get_major_eigen_vectors(double eigen_vectors[2][2][2][9], double eigen_value[2][2][2][3]) {
  const double kInfinity = 1e100;

  static const int kCodeList[8][3] = {{0, 0, 0}, {0, 0, 1}, {0, 1, 1}, {0, 1, 0},
                                      {1, 1, 0}, {1, 1, 1}, {1, 0, 1}, {1, 0, 0}};
  static const int kPermutationList[6][3] = {{0, 1, 2}, {0, 2, 1},
                                             {1, 0, 2}, {1, 2, 0},
                                             {2, 0, 1}, {2, 1, 0}};
  
  static double opt[8][6];
  static int prev[8][6];

  for (int i = 0; i < 8; i++) {
    for (int j = 0; j < 6; j++) {
      opt[i][j] = kInfinity;
    }
  }
  
  opt[0][0] = 0.0;

  for (int i = 1; i < 8; i++) {
    double best[3] = {0.0, 0.0, 0.0};
    for (int a = 0; a < 3; a++) {
      double *vec_a = eigen_vectors[kCodeList[i - 1][0]]
                                   [kCodeList[i - 1][1]]
                                   [kCodeList[i - 1][2]] + a * 3;
      for (int b = 0; b < 3; b++) {
        double *vec_b = eigen_vectors[kCodeList[i][0]]
                                     [kCodeList[i][1]]
                                     [kCodeList[i][2]] + b * 3;
        double cos_ab = fabs(dot_product_3d(vec_a, vec_b));
        if (cos_ab > best[b]) {
          best[b] = cos_ab;
        }
      }
    }

    int best_prev_code = 0;
    for (int code = 1; code < 6; code++) {
      if (opt[i - 1][code] < opt[i - 1][best_prev_code]) {
        best_prev_code = code;
      }
    }

    for (int curr_code = 0; curr_code < 6; curr_code++) {
      double similarity[3];
      int prev_idx[3] = {kPermutationList[curr_code][0],
                         kPermutationList[curr_code][1],
                         kPermutationList[curr_code][2]};
      for (int idx = 0; idx < 3; idx++) {
        double *vec_curr = eigen_vectors[kCodeList[i][0]]
                                        [kCodeList[i][1]]
                                        [kCodeList[i][2]] + idx * 3;
        double *vec_prev = eigen_vectors[kCodeList[i - 1][0]]
                                        [kCodeList[i - 1][1]]
                                        [kCodeList[i - 1][2]] + prev_idx[idx] * 3;
        similarity[idx] = fabs(dot_product_3d(vec_curr, vec_prev));
      }
     
      prev[i][curr_code] = best_prev_code;
      opt[i][curr_code] = opt[i - 1][best_prev_code];
      for (int idx = 0; idx < 3; idx++) {
        opt[i][curr_code] += best[idx] - similarity[idx];
      }
    }
  }
}
*/

}

void CRidgeExtractor::get_gradient(
    vtkStructuredPoints *scalar_field,
    vtkStructuredPoints **gradient) {
  vtkSmartPointer<vtkGradientFilter> grad_filter =
      vtkSmartPointer<vtkGradientFilter>::New();
  grad_filter->SetInputData(scalar_field);
  grad_filter->Update();

  *gradient = vtkStructuredPoints::New();
  (*gradient)->ShallowCopy(grad_filter->GetOutput());

  (*gradient)->GetPointData()->SetActiveScalars("Gradients");
  (*gradient)->GetPointData()->GetScalars()->SetName("gradient");
}

void CRidgeExtractor::get_gradient_and_hessian(
    vtkStructuredPoints *scalar_field,
    vtkStructuredPoints **gradient,
    vtkStructuredPoints **hessian) {
  // Calculate the gradient
  vtkSmartPointer<vtkGradientFilter> grad_filter =
      vtkSmartPointer<vtkGradientFilter>::New();
  grad_filter->SetInputData(scalar_field);
  grad_filter->Update();

  *gradient = vtkStructuredPoints::New();
  (*gradient)->ShallowCopy(grad_filter->GetOutput());

  (*gradient)->GetPointData()->SetActiveScalars("Gradients");
  (*gradient)->GetPointData()->GetScalars()->SetName("gradient");

  // Calculate the hessian
  grad_filter = vtkSmartPointer<vtkGradientFilter>::New();
  grad_filter->SetInputData(*gradient);
  grad_filter->Update();

  *hessian = vtkStructuredPoints::New();
  (*hessian)->ShallowCopy(grad_filter->GetOutput());

  (*hessian)->GetPointData()->SetActiveScalars("Gradients");
  (*hessian)->GetPointData()->GetScalars()->SetName("hessian");
}

void CRidgeExtractor::get_cauchy_green_tensor(
    vtkStructuredPoints *flow_map, vtkStructuredPoints **cauchy_green) {
  get_gradient(flow_map, cauchy_green);
  (*cauchy_green)->GetPointData()->GetScalars()->SetName("cauchy_green");
}

void CRidgeExtractor::get_ftle(vtkStructuredPoints *cauchy_green,
                      vtkStructuredPoints **ftle) {
  int dimensions[3];
  double spacing[3], origin[3];

  cauchy_green->GetDimensions(dimensions);
  cauchy_green->GetSpacing(spacing);
  cauchy_green->GetOrigin(origin);

  vtkSmartPointer<vtkDoubleArray> ftle_data =
      vtkSmartPointer<vtkDoubleArray>::New();

  ftle_data->SetName("ftle");
  ftle_data->SetNumberOfComponents(1);

  int index = 0;
  for (int z = 0; z < dimensions[2]; z++) {
    for (int y = 0; y < dimensions[1]; y++) {
      for (int x = 0; x < dimensions[0]; x++) {
        double tensor[9];
        cauchy_green->GetPointData()->GetScalars()->GetTuple(index, tensor);
        index++;

        double **f_tensor = create_matrix<double>(3, 3);
        for (int row = 0; row < 3; row++) {
          for (int col = 0; col < 3; col++) {
            f_tensor[row][col] = tensor[row * 3 + col];
          }
        }

        double **f_transpose = transpose(f_tensor, 3, 3);

        double **c_tensor = matrix_matrix_multiplication(
            f_transpose, f_tensor, 3, 3, 3);

        double **eigen_vectors = create_matrix<double>(3, 3);
        double eigen_values[3];

        vtkMath::Jacobi(c_tensor, eigen_values, eigen_vectors);

        ftle_data->InsertNextTuple1(log(sqrt(eigen_values[0])));

        delete_matrix(eigen_vectors);
        delete_matrix(c_tensor);
        delete_matrix(f_transpose);
        delete_matrix(f_tensor);
      }
    }
  }

  *ftle = vtkStructuredPoints::New();
  (*ftle)->SetDimensions(dimensions);
  (*ftle)->SetOrigin(origin);
  (*ftle)->SetSpacing(spacing);
  (*ftle)->GetPointData()->SetScalars(ftle_data);
}

vtkPolyData *CRidgeExtractor::extract_ridges(
    vtkStructuredPoints *flow_map) {
  int dimensions[3];
  double spacing[3], origin[3];
  flow_map->GetDimensions(dimensions);
  flow_map->GetSpacing(spacing);
  flow_map->GetOrigin(origin);

  // Get Cauchy-Green tensor
  vtkStructuredPoints *cauchy_green = NULL;
  get_cauchy_green_tensor(flow_map, &cauchy_green);

  // Get FTLE
  vtkStructuredPoints *ftle = NULL;
  get_ftle(cauchy_green, &ftle);

  // Get gradient of FTLE
  vtkStructuredPoints *grad_ftle = NULL;
  get_gradient(ftle, &grad_ftle);

  int nx = dimensions[0];
  int ny = dimensions[1];
  int nz = dimensions[2];

  /*
  int ****edge_mark = create_4d_array<int>(nx, ny, nz, 3);
  for (int x = 0; x < nx; x++) {
    for (int y = 0; y < ny; y++) {
      for (int z = 0; z < nz; z++) {
        for (int d = 0; d < 3; d++) {
          edge_mark[x][y][z][d] = -1;
        }
      }
    }
  }
  */

  vtkSmartPointer<vtkPoints> mesh_points =
      vtkSmartPointer<vtkPoints>::New();
  vtkSmartPointer<vtkCellArray> mesh_cells =
      vtkSmartPointer<vtkCellArray>::New();

  int total = (nx - 1) * (ny - 1) * (nz - 1);
  int num_invalid_cells = 0;
  int count_cells = 0;

  for (int x = 0; x + 1 < nx; x++) {
    for (int y = 0; y + 1 < ny; y++) {
      for (int z = 0; z + 1 < nz; z++) {
        double dot_prod[2][2][2], eigen_vector[2][2][2][9], eigen_value[2][2][2][3], grad[2][2][2][3];
        double e1[2][2][2][3];

        // Collect eigen_vector and grad
        for (int dx = 0; dx < 2; dx++) {
          for (int dy = 0; dy < 2; dy++) {
            for (int dz = 0; dz < 2; dz++) {
              int curr_x = x + dx;
              int curr_y = y + dy;
              int curr_z = z + dz;

              double **cg = create_matrix<double>(3, 3);
              int point_id = (curr_z * ny + curr_y) * nx + curr_x;

              double tensor[9];
              cauchy_green->GetPointData()->GetScalars()
                                          ->GetTuple(point_id, tensor);
              for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                  cg[i][j] = tensor[i * 3 + j];
                }
              }

              get_eigen_vector(cg, eigen_vector[dx][dy][dz], eigen_value[dx][dy][dz]);
              for (int idx = 0; idx < 3; idx++) {
                e1[dx][dy][dz][idx] = eigen_vector[dx][dy][dz][idx];
              }

              delete_matrix(cg);
              
              grad_ftle->GetPointData()->GetScalars()
                                       ->GetTuple(point_id, tensor);
              for (int i = 0; i < 3; i++) {
                grad[dx][dy][dz][i] = tensor[i];
              }
            }
          }
        }

        /// DEBUG ///
        bool invalid_cell = false;

        if (!check_consistency(eigen_vector)) {
          num_invalid_cells++;
          invalid_cell = true;
        }
        count_cells++;
        // printf("%d, %d, %lf\n", num_invalid_cells, count_cells, static_cast<double>(num_invalid_cells) / count_cells);

        if (invalid_cell) {
          get_major_eigen_vectors(eigen_vector, eigen_value, e1);
          // continue;
        }
        
        // Re-orientate e3
        double **vectors = create_matrix<double>(8, 3);
        int num_vectors = 0;
        for (int dx = 0; dx < 2; dx++) {
          for (int dy = 0; dy < 2; dy++) {
            for (int dz = 0; dz < 2; dz++) {
              for (int i = 0; i < 3; i++) {
                vectors[num_vectors][i] = e1[dx][dy][dz][i];
              }
              num_vectors++;
            }
          }
        }

        double *pivot = principal_component(vectors, 8, 3);

        delete_matrix(vectors);

        for (int dx = 0; dx < 2; dx++) {
          for (int dy = 0; dy < 2; dy++) {
            for (int dz = 0; dz < 2; dz++) {
              if (dot_product_3d(pivot, e1[dx][dy][dz]) < 0.0) {
                for (int i = 0; i < 3; i++) {
                  e1[dx][dy][dz][i] *= -1.0;
                }
              }
              dot_prod[dx][dy][dz] = dot_product_3d(grad[dx][dy][dz],
                                                    e1[dx][dy][dz]);
            }
          }
        }

        delete [] pivot;

        // Identify iso-surfaces
        int cube_code = 0;
        for (int i = 0; i < 8; i++) {
          int dx = kVertexList[i][0];
          int dy = kVertexList[i][1];
          int dz = kVertexList[i][2];

          if (dot_prod[dx][dy][dz] <= 0.0) {
            cube_code |= (1 << i);
          }
        }

        for (int i = 0; i < numVertsTable[cube_code]; i += 3) {
          mesh_cells->InsertNextCell(3);

          for (int j = 0; j < 3; j++) {
            int edge_idx = triTable[cube_code][i + j];
            int vtx_1 = kEdgeList[edge_idx][0];
            int vtx_2 = kEdgeList[edge_idx][1];

            int dim;
            for (dim = 0; dim < 3; dim++) {
              if (kVertexList[vtx_1][dim] != kVertexList[vtx_2][dim]) {
                break;
              }
            }

            if (kVertexList[vtx_1][dim] > kVertexList[vtx_2][dim]) {
              std::swap(vtx_1, vtx_2);
            }

            int start_x = kVertexList[vtx_1][0];
            int start_y = kVertexList[vtx_1][1];
            int start_z = kVertexList[vtx_1][2];

            int finish_x = kVertexList[vtx_2][0];
            int finish_y = kVertexList[vtx_2][1];
            int finish_z = kVertexList[vtx_2][2];

            // Insert a new point to the mesh (always)
            // if (edge_mark[x + start_x][y + start_y][z + start_z][dim] == -1)
            {
              // edge_mark[x + start_x][y + start_y][z + start_z][dim] =
              //     mesh_points->GetNumberOfPoints();

              double dot_prod_1 = dot_prod[start_x][start_y][start_z];
              double dot_prod_2 = dot_prod[finish_x][finish_y][finish_z];

              if (dot_prod_1 * dot_prod_2 > 0.0) {
                report_error("Same sign in marching cubes");
              }

              double lambda = -dot_prod_1 / (dot_prod_2 - dot_prod_1);

              double aug_x = (finish_x - start_x) * spacing[0] * lambda;
              double aug_y = (finish_y - start_y) * spacing[1] * lambda;
              double aug_z = (finish_z - start_z) * spacing[2] * lambda;

              double point_x = origin[0] + spacing[0] * (x + start_x) + aug_x;
              double point_y = origin[1] + spacing[1] * (y + start_y) + aug_y;
              double point_z = origin[2] + spacing[2] * (z + start_z) + aug_z;

              mesh_points->InsertNextPoint(point_x, point_y, point_z);
            }

            mesh_cells->InsertCellPoint(mesh_points->GetNumberOfPoints() - 1);
            // mesh_cells->InsertCellPoint(
            //     edge_mark[x + start_x][y + start_y][z + start_z][dim]);
          }
        }
      }
    }
  }

  /// DEBUG ///
  printf("%d, %d, %lf\n", num_invalid_cells, total, static_cast<double>(num_invalid_cells) / total);

  vtkPolyData *mesh = vtkPolyData::New();
  mesh->SetPoints(mesh_points);
  mesh->SetPolys(mesh_cells);

  // delete_4d_array(edge_mark);

  cauchy_green->Delete();
  ftle->Delete();
  grad_ftle->Delete();

  return mesh;
}
