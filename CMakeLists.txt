cmake_minimum_required(VERSION 2.8)

project(CRidgeExtractor)

find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

add_executable(GradientFilterTest gradient_filter_test.cc)
add_executable(PrincipalComponentTest principal_component_test.cc
                                      math.cc)
add_executable(CRidgeExtractorTest c_ridge_extractor_test.cc
                                   c_ridge_extractor.cc
                                   math.cc
                                   util.cc)

target_link_libraries(GradientFilterTest ${VTK_LIBRARIES})
target_link_libraries(PrincipalComponentTest ${VTK_LIBRARIES})
target_link_libraries(CRidgeExtractorTest ${VTK_LIBRARIES})
