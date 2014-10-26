// Author: Mingcheng Chen (linyufly@gmail.com)

#include <vtkStructuredPoints.h>
#include <vtkSmartPointer.h>
#include <vtkStructuredPointsReader.h>
#include <vtkGradientFilter.h>
#include <vtkIndent.h>
#include <vtkPointData.h>
#include <vtkDataArray.h>

#include <cstdio>

#include <iostream>

const char *kDataFile = "data/gyre_half.vtk";
const int kNumberOfSamples = 10;

vtkStructuredPoints *read_grid() {
  vtkSmartPointer<vtkStructuredPointsReader> reader =
      vtkSmartPointer<vtkStructuredPointsReader>::New();

  reader->SetFileName(kDataFile);
  reader->Update();

  vtkStructuredPoints *grid = vtkStructuredPoints::New();
  grid->ShallowCopy(reader->GetOutput());

  return grid;
}

void check_grid(vtkStructuredPoints *grid) {
  double origin[3], spacing[3];
  int dimensions[3];
  grid->GetOrigin(origin);
  grid->GetSpacing(spacing);
  grid->GetDimensions(dimensions);

  printf("origin: %lf, %lf, %lf\n", origin[0], origin[1], origin[2]);
  printf("spacing: %lf, %lf, %lf\n", spacing[0], spacing[1], spacing[2]);
  printf("dimensions: %d, %d, %d\n", dimensions[0], dimensions[1], dimensions[2]);

  grid->PrintSelf(std::cout, vtkIndent(0));
}

vtkStructuredPoints *get_gradient(vtkStructuredPoints *grid) {
  vtkSmartPointer<vtkGradientFilter> grad_filter =
      vtkSmartPointer<vtkGradientFilter>::New();

  grad_filter->SetInputData(grid);
  grad_filter->Update();
  
  vtkStructuredPoints *gradient = vtkStructuredPoints::New();
  gradient->ShallowCopy(grad_filter->GetOutput());

  gradient->GetPointData()->SetActiveScalars("Gradients");

  return gradient;
}

int main() {
  vtkStructuredPoints *grid = read_grid();
  printf("Check the original grid\n");
  check_grid(grid);
  printf("\n");

  vtkStructuredPoints *gradient = get_gradient(grid);
  printf("Active scalar is %s.\n", gradient->GetPointData()->GetScalars()->GetName());
  gradient->GetPointData()->GetScalars()->SetName("gradient");
  printf("New active scalar is %s.\n", gradient->GetPointData()->GetScalars()->GetName());
  printf("Check the gradient grid\n");
  check_grid(gradient);
  printf("\n");

  vtkStructuredPoints *hessian = get_gradient(gradient);
  printf("Check the hessian grid\n");
  check_grid(hessian);
  printf("\n");

  double tensor[9];
  for (int i = 0; i < kNumberOfSamples; i++) {
    hessian->GetPointData()->GetScalars()->GetTuple(i, tensor);
    for (int row = 0; row < 3; row++) {
      for (int col = 0; col < 3; col++) {
        if (col) printf(" ");
        printf("%lf", tensor[row * 3 + col]);
      }
      printf("\n");
    }
    printf("\n");
  }

  return 0;
}
