#include <mpi.h>
#include <pdi.h>
#include <Kokkos_Core.hpp>

#include <iostream>


int main(int argc, char** argv) 
{
  // Initialize MPI and Kokkos
  MPI_Init(&argc, &argv);
  Kokkos::initialize(argc, argv);

  // Print Kokkos info
  Kokkos::print_configuration(std::cout);
  
  //Get and print MPI values
  int mpi_rank, mpi_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  printf("Rank %d out of %d\n", mpi_rank, mpi_size);
  fflush(stdout); // Flush the output buffer

  //Finalize Kokkos and MPI
  Kokkos::finalize();
  MPI_Finalize();

  return 0;
}
