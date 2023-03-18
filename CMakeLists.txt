cmake_minimum_required(VERSION 3.14)
project(wavernn_c_mb)

set(CMAKE_CXX_STANDARD 11)

set(CMAKE_BUILD_TYPE Debug)  # Debug  Release

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/")

include_directories(/home/wqt/software/eigen-3.3.7/)

include_directories(/home/wqt/intel/mkl/include/)
link_directories(/home/wqt/intel/mkl/lib/intel64 /home/wqt/intel/lib/intel64)


SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -ffast-math -march=native -fopenmp -funroll-all-loops")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ffast-math -march=native -fopenmp -funroll-all-loops")

add_executable (vocoder vocoder.cpp)
add_library(wavernn wavernn.cpp net_impl.cpp)
#target_link_libraries(vocoder wavernn )

target_link_libraries(vocoder wavernn -lpthread -lm -ldl -lmkl_rt -m64 -L${MKLROOT}/lib/intel64
        -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl)


add_subdirectory(pybind11)
pybind11_add_module(WaveRNNVocoder WaveRNNVocoder.cpp wavernn.cpp net_impl.cpp)

target_link_libraries(WaveRNNVocoder PRIVATE mkl_rt -fopenmp)


