cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(axt_spl CXX CUDA)

set(NUM_ITE 1) # Number of iterations to run kernels
option(USE_OPENMP "Use OpenMP" TRUE)

find_package(CUDAToolkit REQUIRED)

add_executable(cudaSpmv src/cudaSpmv.cu)
target_include_directories(cudaSpmv PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(cudaSpmv PRIVATE CUDA::cusparse)

if (USE_OPENMP)
message(STATUS "Using OpenMP")
find_package(OpenMP REQUIRED)
target_compile_definitions(cudaSpmv PRIVATE _OMP_)
target_link_libraries(cudaSpmv PUBLIC OpenMP::OpenMP_CXX)
endif()

set_property(TARGET cudaSpmv PROPERTY CUDA_ARCHITECTURES 60 70 86)

target_compile_definitions(cudaSpmv PRIVATE NUM_ITE=${NUM_ITE})
target_compile_definitions(cudaSpmv PRIVATE FP_TYPE=FP_DOUBLE)

