cmake_minimum_required(VERSION 3.10)

project(MyProject)

# Find MPI
find_package(MPI REQUIRED COMPONENTS CXX)

# Add MPI compiler flags
add_definitions(${MPI_CXX_COMPILE_FLAGS})

# Specify the path to the external libraries
set(KOKKOS_PATH "${PROJECT_SOURCE_DIR}/vendor/kokkos")
# Path to the installed PDI
set(PDI_INSTALL_PATH "${PROJECT_SOURCE_DIR}/vendor/install_pdi")

# Add Kokkos
add_subdirectory(${KOKKOS_PATH})

# Comment out these lines as PDI is already installed
# set(BUILD_SHARED_LIBS ON)
# add_subdirectory(${PDI_PATH})

# Create executable
add_executable(my_app src/main.cpp)

# Include MPI headers
include_directories(${MPI_CXX_INCLUDE_PATH})

# Include PDI headers
include_directories(${PDI_INSTALL_PATH}/include)

# Find the PDI library
find_library(PDI_LIBRARY NAMES pdi PATHS ${PDI_INSTALL_PATH}/lib)

# Link libraries with MPI and PDI
target_link_libraries(my_app ${MPI_CXX_LIBRARIES} ${MPI_CXX_LINK_FLAGS} Kokkos::kokkos ${PDI_LIBRARY})
