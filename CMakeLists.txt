cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(axt_spl CXX CUDA)

set(NUM_ITE 1) # Number of iterations to run kernels
option(USE_OPENMP "Use OpenMP" TRUE)

find_package(CUDAToolkit REQUIRED)

add_executable(test_gpu_spmv src/test_gpu_spmv.cu
	src/test_gaxtuh1hw08w.cu
	src/test_gaxtuh.cu
	src/test_gaxtuh1hw04w.cu
	src/test_gcucsr.cu
	src/test_ncsr.cu
	src/test_gcsr.cu
	src/test_gaxtuh1hw16w.cu
	src/test_gaxtch.cu
	src/test_gaxc.cu
	src/test_gaxtch1.cu
	src/test_gk1.cu)
target_include_directories(test_gpu_spmv PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_gpu_spmv PRIVATE CUDA::cusparse)

if (USE_OPENMP)
message(STATUS "Using OpenMP")
find_package(OpenMP REQUIRED)
target_compile_definitions(test_gpu_spmv PRIVATE _OMP_)
target_link_libraries(test_gpu_spmv PUBLIC OpenMP::OpenMP_CXX)
endif()

set_property(TARGET test_gpu_spmv PROPERTY CUDA_ARCHITECTURES 60 70 86)

target_compile_definitions(test_gpu_spmv PRIVATE NUM_ITE=${NUM_ITE})
target_compile_definitions(test_gpu_spmv PRIVATE FP_TYPE=FP_DOUBLE)

