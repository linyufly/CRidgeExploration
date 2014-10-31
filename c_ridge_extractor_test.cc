// Author: Mingcheng Chen (linyufly@gmail.com)

#include "c_ridge_extractor.h"

#include <vtkStructuredPointsReader.h>
#include <vtkStructuredPointsWriter.h>
#include <vtkStructuredPoints.h>
#include <vtkSmartPointer.h>
#include <vtkPointData.h>
#include <vtkDataArray.h>
#include <vtkIndent.h>
#include <vtkPolyData.h>
#include <vtkPolyDataWriter.h>

#include <cstdio>
#include <cstdlib>

#include <iostream>

const char *kDataFile = "data/flow_map.vtk";
const char *kFTLEFile = "ftle.vtk";
const char *kPolyDataFile = "poly_mesh.vtk";
const int kNumberOfSamples = 10;

void get_cauchy_green_tensor_and_get_ftle_test() {
  printf("get_cauchy_green_tensor_and_get_ftle_test {\n");

  vtkSmartPointer<vtkStructuredPointsReader> reader =
      vtkSmartPointer<vtkStructuredPointsReader>::New();

  reader->SetFileName(kDataFile);
  reader->Update();

  vtkStructuredPoints *cauchy_green = NULL, *ftle = NULL;
  CRidgeExtractor::get_cauchy_green_tensor(reader->GetOutput(),
                                           &cauchy_green);
  CRidgeExtractor::get_ftle(cauchy_green, &ftle);

  cauchy_green->PrintSelf(std::cout, vtkIndent(0));
  ftle->PrintSelf(std::cout, vtkIndent(0));

  vtkSmartPointer<vtkStructuredPointsWriter> writer =
      vtkSmartPointer<vtkStructuredPointsWriter>::New();

  writer->SetFileName(kFTLEFile);
  writer->SetInputData(ftle);
  writer->Write();

  cauchy_green->Delete();
  ftle->Delete();

  printf("} get_cauchy_green_tensor_and_get_ftle_test\n");
}

void extract_ridges_test() {
  printf("extract_ridges_test {\n");

  vtkSmartPointer<vtkStructuredPointsReader> reader =
      vtkSmartPointer<vtkStructuredPointsReader>::New();

  reader->SetFileName(kDataFile);
  reader->Update();

  CRidgeExtractor extractor;
  vtkPolyData *ridges = extractor.extract_ridges(reader->GetOutput());

  int num_points = ridges->GetNumberOfPoints();
  int num_cells = ridges->GetNumberOfCells();

  printf("num_points = %d\n", num_points);
  printf("num_cells = %d\n", num_cells);

  vtkSmartPointer<vtkPolyDataWriter> writer =
    vtkSmartPointer<vtkPolyDataWriter>::New();

  writer->SetFileName(kPolyDataFile);
  writer->SetInputData(ridges);
  writer->Write();

  if (ridges) {
    ridges->Delete();
  }

  printf("} extract_ridges_test\n\n");
}

int main() {
  // get_cauchy_green_tensor_and_get_ftle_test();
  extract_ridges_test();

  return 0;
}
