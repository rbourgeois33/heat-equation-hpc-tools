#include "mpi_decomposition.hpp"

KOKKOS_INLINE_FUNCTION
double initial_condition(const double x, const double y) {
    
    const double rmax = 0.2;
    const double r = Kokkos::sqrt((x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5));
    return r < rmax ? 1 : 0;
}

void Initialisation(Kokkos::View<double**>& U, const double dx, const double dy, const MPI_DECOMPOSITION& mpi_decomposition, const Kokkos::MDRangePolicy<Kokkos::Rank<2>> &policy) {
    
    int nx = mpi_decomposition.nx;
    int ny = mpi_decomposition.ny;
    Coordinates mpi_coords = mpi_decomposition.coords;

    Kokkos::parallel_for("Initialisation", 
                        policy, 
                        KOKKOS_LAMBDA(const int i, const int j) {
                            const double x = (i + mpi_coords.x * nx) * dx - 0.5 * dx;
                            const double y = (j + mpi_coords.y * ny) * dy - 0.5 * dy;
                            U(i, j) = initial_condition(x, y);
    });
}

void stencil_kernel(Kokkos::View<double**>& U, const Kokkos::View<double**>& U_, const double dx, const double dy, const double dt, const double kappa, const Kokkos::MDRangePolicy<Kokkos::Rank<2>> &policy) {
    
    Kokkos::parallel_for("heat_equation_kernel", 
                        policy, 
                        KOKKOS_LAMBDA(const int i, const int j) {
                            U(i, j) = U_(i, j) + dt * kappa * ((U_(i + 1, j) - 2.0 * U_(i, j) + U_(i - 1, j)) / (dx * dx) + (U_(i, j + 1) - 2.0 * U_(i, j) + U_(i, j - 1)) / (dy * dy));
    });
}

void print_perf(const double elapsed_time, const int nx, const int ny, const int nstep) {
    
    int num_threads = Kokkos::DefaultExecutionSpace::concurrency();
    double MCellUpdate = double(nx) * double(ny) * double(nstep) / 1e6;
    double MCellUpdatePerSec = MCellUpdate / elapsed_time;
    printf("Elapsed time: %f\nNumber of threads: %d\nMCell update per second: %f\n\n", elapsed_time, num_threads, MCellUpdatePerSec);
}

void MPIBoundaryCondition(Kokkos::View<double**>& U, MPI_DECOMPOSITION& mpi_decomposition) {
    
    mpi_decomposition.fill_buffers_from_U(U, mpi_decomposition.up);
    mpi_decomposition.send_recv_buffers(mpi_decomposition.up);
    mpi_decomposition.fill_U_from_buffers(U, mpi_decomposition.up);

    mpi_decomposition.fill_buffers_from_U(U, mpi_decomposition.down);
    mpi_decomposition.send_recv_buffers(mpi_decomposition.down);
    mpi_decomposition.fill_U_from_buffers(U, mpi_decomposition.down);
}

void compute_total_temperature(const Kokkos::View<double**>& U, const double dx, const double dy, const Kokkos::MDRangePolicy<Kokkos::Rank<2>> &policy, const MPI_DECOMPOSITION& mpi_decomposition) {
    
    double local_sum; //Sum local to the current MPI process

    //Compute the local sum
    // Kokkos::parallel_reduce("total temperature reduction kernel", 
    //                         policy, 
    //                         KOKKOS_LAMBDA(const int& i, const int& j, double& lsum) {
    //                             lsum += U(i, j)*dx*dy;},
    //                         local_sum
    // );

    //Wait until the local reduction is over
    Kokkos::fence();
    MPI_Barrier(mpi_decomposition.comm);
    //Reduce the local sums into the global sum
    double global_sum; 

    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, mpi_decomposition.comm);

    printf("Total sum = %f\n", global_sum/mpi_decomposition.size);

}

void write_solution_to_file(const Kokkos::View<double**>::HostMirror& U_host, const Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& U_IO, const Kokkos::View<double**>& U, int& nwrite, double time) {
    
    // Copy the view to the host mirror
    Kokkos::deep_copy(U_host, U);
    //Transpose via deepcopy if U and U'IO layouts are differents
    Kokkos::deep_copy(U_IO, U_host);

    // Expose the solution
    PDI_multi_expose("write_data",
                    "nwrite", &nwrite, PDI_OUT,
                    "time", &time, PDI_OUT,
                    "main_field", U_IO.data(), PDI_OUT,
                    NULL);
    
    //Increment writing counter
    nwrite=nwrite+1;
}

void heat_equation(int argc, char* argv[], const MPI_Comm main_comm, const PC_tree_t conf) {    
    
    //Domain lenght, final time and diffusion coefficient 
    const double Lx = 1.0, Ly = 1.0, Tend = 1, kappa = 0.1;

    //CFL number
    const double cfl = 0.9;

    //Resolution per MPI sub-domain
    const int nx = 128, ny = 128;

    //MPI cartesian decomposition
    Coordinates mpi_max_coords = {2, 2};

    //Max number of iteration
    const int nmax = 9999999;

    //Don't change parameters from here (they are deduced)

    //Total resolution
    const int Nx = nx * mpi_max_coords.x, Ny = ny * mpi_max_coords.y;

    //Space step and time step computed from CFL condition
    const double dx = Lx / Nx, dy = Ly / Ny;
    const double inv_dt_x = 1.0 / (0.5 * dx * dx / kappa), inv_dt_y = 1.0 / (0.5 * dy * dy / kappa);
    double dt = cfl / (inv_dt_x + inv_dt_y);
    
    //Writing frequency
    const int freq_write = std::floor(Tend / (10 * dt));

    // Size of the arrays
    const int size_x = nx + 2, size_y = ny + 2;
    const Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy({1, 1}, {nx + 1, ny + 1});

    // Get MPI info and compute rank info
    int mpi_rank, mpi_size;
    MPI_Comm_rank(main_comm, &mpi_rank);
    MPI_Comm_size(main_comm, &mpi_size);
    MPI_DECOMPOSITION mpi_decomposition(mpi_rank, mpi_size, mpi_max_coords, main_comm, nx, ny);

    // Send meta-data to PDI
    PDI_multi_expose("init_PDI", 
                    "mpi_coords_x", &mpi_decomposition.coords.x, PDI_OUT,
                    "mpi_coords_y", &mpi_decomposition.coords.y, PDI_OUT,
                    "mpi_max_coords_x", &mpi_decomposition.max_coords.x, PDI_OUT,
                    "mpi_max_coords_y", &mpi_decomposition.max_coords.y, PDI_OUT,
                    "nx", &nx, PDI_OUT,
                    "ny", &ny, PDI_OUT,
                    NULL);

    // Print simulation information
    if (mpi_rank == 0) {
        Kokkos::print_configuration(std::cout);
        printf("------------ Simulation Information --------------\n");
        printf("Diffusion coefficient: %f\n", kappa);
        printf("Cell size: %f\n", dx);
        printf("Time step: %f\n", dt);
        printf("Number of cells per proc: %d , %d\n", nx, ny);
        printf("Total of cells per proc: %d , %d\n", Nx, Ny);
        printf("Number of iterations: %d\n", nmax);
        printf("CFL number: %f\n", cfl);
        printf("Final time: %f\n", Tend);
        printf("Domain size: %f , %f\n", Lx, Ly);
        printf("--------------------------------------------------\n");
    }

    // Print MPI info
    mpi_decomposition.printDetails();

    // Allocate the views (Kokkos's arrays)
    Kokkos::View<double**> U("Solution U on device", size_x, size_y);
    Kokkos::View<double**> U_("Intermediate Solution U on device", size_x, size_y);
    auto U_host = Kokkos::create_mirror(U);

    //We force the layout on the view used for I/O for compatibility with the host default layout.
    //Indeed, the default view on the device may change depending on the backend.
    Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> U_IO("I/O array for PDI", size_x, size_y);

    // Initialization
    Initialisation(U, dx, dy, mpi_decomposition, policy);

    // Set time and indexes to 0
    double time = 0.0;
    int nstep = 0, nwrite = 0;
    write_solution_to_file(U_host, U_IO, U, nwrite, time);

    // Main simulation loop
    Kokkos::Timer timer;
    while (time < Tend && nstep < nmax) {
        Kokkos::fence(); // Wait until all previous Kokkos kernels are completed
        MPI_Barrier(mpi_decomposition.comm); // Synchronize all MPI processes

        // Adjust the time step to precisely reach Tend
        if (time + dt > Tend) {
            dt = Tend - time;
        }

        // Apply MPI boundary conditions
        MPIBoundaryCondition(U, mpi_decomposition);

        // Copy U to U_ to preserve the current state for the stencil operation
        Kokkos::deep_copy(U_, U);

        // Update the solution U using the stencil kernel
        stencil_kernel(U, U_, dx, dy, dt, kappa, policy);

        // Update the time and iteration counters
        time += dt;
        nstep++;

        // Periodic progress information
        if (nstep % 500 == 0 && mpi_rank == 0) {
            printf("Time step: %d, Time: %f\n", nstep, time);
        }

        // Write solution to file at specified intervals
        if (nstep % freq_write == 0) {
            write_solution_to_file(U_host, U_IO, U, nwrite, time);
        }
    }
    double elapsed_time = timer.seconds();

    // Final solution write and performance print
    write_solution_to_file(U_host, U_IO, U, nwrite, time);
    if (mpi_rank == 0) {
        // Final simulation status and performance details
        print_perf(elapsed_time, nx, ny, nstep);
    }
}
