cmake_minimum_required(VERSION 3.15)
project(kmp_mpi_cuda)

find_package(MPI REQUIRED)

include_directories(${MPI_INCLUDE_PATH})

set(CMAKE_CXX_STANDARD 14)
add_executable(kmp_mpi_cuda src/kmp_mpi.cpp)

target_link_libraries(kmp_mpi_cuda ${MPI_LIBRARIES})

if(MPI_COMPILE_FLAGS)
    set_target_properties(kmp_mpi_cuda PROPERTIES
            COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()

if(MPI_LINK_FLAGS)
    set_target_properties(kmp_mpi_cuda PROPERTIES
            LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()

