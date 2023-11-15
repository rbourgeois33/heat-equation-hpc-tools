#include <mpi.h>
#include <pdi.h>
#include <Kokkos_Core.hpp>

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        // Your Kokkos code here
    }
    Kokkos::finalize();

    return 0;
}
