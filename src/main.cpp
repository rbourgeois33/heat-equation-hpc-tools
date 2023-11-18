// Include MPI, Kokkos and PDI-related headers
#include <mpi.h>
#include <Kokkos_Core.hpp>
#include <pdi.h>
#include <yaml.h>
#include <paraconf.h>

// Include std headers
#include <iostream>

//Include src headers
#include "heat_equation.hpp"


int main(int argc, char** argv) 
{
  // Check the yaml file
  if ( argc != 2 ) 
  {
		fprintf(stderr, "Usage: %s <config_file>\n", argv[0]);
		exit(1);
	}

  // Initialize MPI, PDI and Kokkos
  MPI_Init(&argc, &argv);
  PC_tree_t conf = PC_parse_path(argv[1]);
  PDI_init(PC_get(conf, ".pdi"));
  Kokkos::initialize(argc, argv);

  //Get MPI values and send to PDI
	MPI_Comm main_comm = MPI_COMM_WORLD;
  int mpi_rank; MPI_Comm_rank(main_comm, &mpi_rank);
	int mpi_size; MPI_Comm_size(main_comm, &mpi_size);

  // Print Kokkos and MPI ranks
  if (mpi_rank == 0)
  {
    Kokkos::print_configuration(std::cout);
  }

  //Run simulation
  heat_equation(argc, argv, main_comm, conf);

  //Finalize Kokkos, PDI and MPI
  Kokkos::finalize();
  PC_tree_destroy(&conf);
  PDI_finalize();
  MPI_Finalize();

  return 0;
}
