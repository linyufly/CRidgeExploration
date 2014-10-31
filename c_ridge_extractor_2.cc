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

void get_ek(double **cauchy_green, int k, double *e1, double *l1) {
  double **f_trans = transpose(cauchy_green, 3, 3);
  double **tensor = matrix_matrix_multiplication(f_trans, cauchy_green, 3, 3, 3);

  double *eigen_values = new double[3];
  double **eigen_vectors = create_matrix<double>(3, 3);
  vtkMath::Jacobi(tensor, eigen_values, eigen_vectors);

  *l1 = eigen_values[k];

  for (int i = 0; i < 3; i++) {
    e1[i] = eigen_vectors[i][k];
  }

  delete [] eigen_values;
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

  // Get determinant
  vtkSmartPointer<vtkStructuredPoints> scalar_field =
      vtkSmartPointer<vtkStructuredPoints>::New();
  scalar_field->SetDimensions(dimensions);
  scalar_field->SetOrigin(origin);
  scalar_field->SetSpacing(spacing);

  vtkSmartPointer<vtkDoubleArray> determinant =
      vtkSmartPointer<vtkDoubleArray>::New();
  determinant->SetName("determinant");
  determinant->SetNumberOfComponents(1);
  determinant->SetNumberOfTuples(nx * ny * nz);

  for (int x = 0; x < nx; x++) {
    for (int y = 0; y < ny; y++) {
      for (int z = 0; z < nz; z++) {
        int point_id = (z * ny + y) * nx + x;
        double tensor[9];

        // Get Cauchy-Green tensor
        cauchy_green->GetPointData()->GetScalars()->GetTuple(point_id, tensor);

        double **f_tensor = create_matrix<double>(3, 3);
        for (int row = 0; row < 3; row++) {
          for (int col = 0; col < 3; col++) {
            f_tensor[row][col] = tensor[row * 3 + col];
          }
        }

        double **ft = transpose(f_tensor, 3, 3);
        double **cg = matrix_matrix_multiplication(ft, f_tensor, 3, 3, 3);

        // Get gradient of FTLE
        grad_ftle->GetPointData()->GetScalars()->GetTuple(point_id, tensor);

        double **g = create_matrix<double>(3, 1);
        for (int row = 0; row < 3; row++) {
          g[row][0] = tensor[row];
        }

        double **cg_g = matrix_matrix_multiplication(cg, g, 3, 3, 1);
        double **cg_cg_g = matrix_matrix_multiplication(cg, cg_g, 3, 3, 1);        

        double **det_mat = create_matrix<double>(3, 3);
        for (int row = 0; row < 3; row++) {
          det_mat[row][0] = g[row][0];
          det_mat[row][1] = cg_g[row][0];
          det_mat[row][2] = cg_cg_g[row][0];
        }

        /// DEBUG ///
        if (x == 1 && y == 0 && z == 3 || x == 1 && y == 1 && z == 3) {
          printf("x, y, z: %d, %d, %d\n", x, y, z);
          double **eigen_vectors = create_matrix<double>(3, 3);
          double eigen_values[3];

          vtkMath::Jacobi(cg, eigen_values, eigen_vectors);
          printf("eigen values: %lf, %lf, %lf\n", eigen_values[0], eigen_values[1], eigen_values[2]);
          printf("eigen vectors:\n");
          for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
              printf(" %lf", eigen_vectors[i][j]);
            }
            printf("\n");
          }

          delete_matrix(eigen_vectors);
        }

        determinant->SetTuple1(point_id, determinant_3(det_mat));

        delete_matrix(det_mat);
        delete_matrix(cg_cg_g);
        delete_matrix(cg_g);
        delete_matrix(g);
        delete_matrix(cg);
        delete_matrix(ft);
        delete_matrix(f_tensor);
      }
    }
  }
  scalar_field->GetPointData()->SetScalars(determinant);

  int ****edge_mark = create_4d_array<int>(nx, ny, nz, 3);
  bool ****edge_valid = create_4d_array<bool>(nx, ny, nz, 3);
  for (int x = 0; x < nx; x++) {
    for (int y = 0; y < ny; y++) {
      for (int z = 0; z < nz; z++) {
        for (int d = 0; d < 3; d++) {
          edge_mark[x][y][z][d] = -1;
          edge_valid[x][y][z][d] = false;
        }
      }
    }
  }

  vtkSmartPointer<vtkPoints> mesh_points =
      vtkSmartPointer<vtkPoints>::New();
  vtkSmartPointer<vtkCellArray> mesh_cells =
      vtkSmartPointer<vtkCellArray>::New();

  for (int x = 0; x + 1 < nx; x++) {
    for (int y = 0; y + 1 < ny; y++) {
      for (int z = 0; z + 1 < nz; z++) {
        double dot_prod[2][2][2], e1[2][2][2][3], grad[2][2][2][3];

        /// DEBUG ///
        double scalar[2][2][2];

        // Collect e3 and grad
        for (int dx = 0; dx < 2; dx++) {
          for (int dy = 0; dy < 2; dy++) {
            for (int dz = 0; dz < 2; dz++) {
              int curr_x = x + dx;
              int curr_y = y + dy;
              int curr_z = z + dz;

              double **cg = create_matrix<double>(3, 3);
              int point_id = (curr_z * ny + curr_y) * nx + curr_x;

              /// DEBUG ///
              scalar[dx][dy][dz] = determinant->GetTuple1(point_id);

              double tensor[9];
              cauchy_green->GetPointData()->GetScalars()
                                          ->GetTuple(point_id, tensor);
              for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                  cg[i][j] = tensor[i * 3 + j];
                }
              }

              double l1;

              /// DEBUG ///
              get_ek(cg, 2, e1[dx][dy][dz], &l1);

              delete_matrix(cg);
              
              grad_ftle->GetPointData()->GetScalars()
                                       ->GetTuple(point_id, tensor);
              for (int i = 0; i < 3; i++) {
                grad[dx][dy][dz][i] = tensor[i];
              }
            }
          }
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

              /// DEBUG ///
              if (x == 0 && y == 0 && z == 2) {
                if (dx == 1 && dz == 1) {
                  printf("%d, %d, %d:", dx, dy, dz);
                  for (int i = 0; i < 3; i++) {
                    printf(" %lf", vectors[num_vectors][i]);
                  }
                  printf(" |");
                  for (int i = 0; i < 3; i++) {
                    printf(" %lf", grad[dx][dy][dz][i]);
                  }
                  printf("\n");
                }
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
          // mesh_cells->InsertNextCell(3);

          int num_invalid = 0;
          int point_list[3];

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

            // Insert a new point to the mesh if necessary
            if (edge_mark[x + start_x][y + start_y][z + start_z][dim] == -1) {
              edge_mark[x + start_x][y + start_y][z + start_z][dim] =
                  mesh_points->GetNumberOfPoints();

              /// DEBUG ///
              double dot_prod_1 = dot_prod[start_x][start_y][start_z];
              double dot_prod_2 = dot_prod[finish_x][finish_y][finish_z];
              double _dot_prod_1 = scalar[start_x][start_y][start_z];
              double _dot_prod_2 = scalar[finish_x][finish_y][finish_z];

              if (dot_prod_1 * dot_prod_2 > 0.0) {
                // printf("origin: %lf, %lf\n", _dot_prod_1, _dot_prod_2);
                // printf("deter: %lf, %lf\n", dot_prod_1, dot_prod_2);
                // printf("x, y, z: %d, %d, %d\n", x, y, z);
                // printf("start: %d, %d, %d\n", start_x, start_y, start_z);
                // printf("finish: %d, %d, %d\n", finish_x, finish_y, finish_z);
                // report_error("Same sign in marching cubes\n");
              } else {
                edge_valid[x + start_x][y + start_y][z + start_z][dim] = true;
              }

              double lambda = -dot_prod_1 / (dot_prod_2 - dot_prod_1);
              // double lambda = -_dot_prod_1 / (_dot_prod_2 - _dot_prod_1);

              double aug_x = (finish_x - start_x) * spacing[0] * lambda;
              double aug_y = (finish_y - start_y) * spacing[1] * lambda;
              double aug_z = (finish_z - start_z) * spacing[2] * lambda;

              double point_x = origin[0] + spacing[0] * (x + start_x) + aug_x;
              double point_y = origin[1] + spacing[1] * (y + start_y) + aug_y;
              double point_z = origin[2] + spacing[2] * (z + start_z) + aug_z;

              mesh_points->InsertNextPoint(point_x, point_y, point_z);
            }

            if (!edge_valid[x + start_x][y + start_y][z + start_z][dim]) {
              num_invalid++;
            }

            point_list[j] = edge_mark[x + start_x][y + start_y][z + start_z][dim];

            // mesh_cells->InsertCellPoint(
            //     edge_mark[x + start_x][y + start_y][z + start_z][dim]);
          }

          if (num_invalid == 0) {
            mesh_cells->InsertNextCell(3);
            for (int idx = 0; idx < 3; idx++) {
              mesh_cells->InsertCellPoint(point_list[idx]);
            }
          }
        }
      }
    }
  }

  vtkPolyData *mesh = vtkPolyData::New();
  mesh->SetPoints(mesh_points);
  mesh->SetPolys(mesh_cells);

  delete_4d_array(edge_mark);
  delete_4d_array(edge_valid);

  cauchy_green->Delete();
  ftle->Delete();
  grad_ftle->Delete();

  return mesh;
}
