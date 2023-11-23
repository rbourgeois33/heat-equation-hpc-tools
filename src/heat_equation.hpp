#include "mpi_decomposition.hpp"

KOKKOS_INLINE_FUNCTION
double initial_condition(const double x, const double y)
{
    const double rmax=0.2;
    const double r = Kokkos::sqrt((x-0.5)*(x-0.5)+(y-0.5)*(y-0.5));
    return r<rmax ? 1:0;
}

//Initialize the view U with the initial condition function above
void Initialisation(Kokkos::View<double**>& U, const double dx, const double dy, const MPI_DECOMPOSITION mpi_decomposition, const Kokkos::MDRangePolicy<Kokkos::Rank<2>> &policy)
{
    int nx = mpi_decomposition.nx;
    int ny = mpi_decomposition.ny;
    Coordinates mpi_coords = {mpi_decomposition.coords.x, mpi_decomposition.coords.y};

    Kokkos::parallel_for
    (
        "Initialisation", 
        policy, 
        KOKKOS_LAMBDA ( const int i , const int j )
        {   
            if ((i==1)&&(j==1)) {
                printf("Rank: %d, (%d, %d), (%d, %d)\n", mpi_decomposition.rank, mpi_decomposition.coords.x, mpi_decomposition.coords.y, nx, ny);
            }
            const double x = (i + mpi_coords.x*nx)*dx - 0.5*dx;
            const double y = (j + mpi_coords.y*ny)*dy - 0.5*dy;
            U(i, j) = initial_condition(x, y) ;
        }
    );
}

//Apply a finite difference scheme on the view U, using the view U_
void stencil_kernel(Kokkos::View<double**>& U, const Kokkos::View<double**>& U_, const double dx, const double dy, const double dt, const double kappa, const Kokkos::MDRangePolicy<Kokkos::Rank<2>> &policy)
{
    Kokkos::parallel_for
    (
        "heat_equation_kernel", 
        policy, 
        KOKKOS_LAMBDA ( const int i , const int j )
        {   
            U(i, j) = U_(i, j) + dt*kappa*( (U_(i+1, j) - 2.0*U_(i, j) + U_(i-1, j))/(dx*dx) + (U_(i, j+1) - 2.0*U_(i, j) + U_(i, j-1))/(dy*dy) );
        }
    );   
}

//Print performance data
void print_perf(const double elapsed_time, const int nx, const int ny, const int nstep)
{
    //Calculate performances
    int num_threads = Kokkos::DefaultExecutionSpace::concurrency();
    double MCellUpdate = double(nx)*double(ny)*double(nstep)/1e6;
    double MCellUpdatePerSec = MCellUpdate/elapsed_time;
    // Print performances
    printf("Elapsed time: %f\nNumber of threads: %d\nMCell update per second: %f\n\n", elapsed_time, num_threads, MCellUpdatePerSec);

}

//Apply boundary condition on view U
void MPIBoundaryCondition(Kokkos::View<double**>& U, MPI_DECOMPOSITION& mpi_decomposition)
{

    mpi_decomposition.fill_buffers_from_U(U, mpi_decomposition.up);

    mpi_decomposition.send_recv_buffers(mpi_decomposition.up);

    mpi_decomposition.fill_U_from_buffers(U, mpi_decomposition.up);

    mpi_decomposition.fill_buffers_from_U(U, mpi_decomposition.down);

    mpi_decomposition.send_recv_buffers(mpi_decomposition.down);

    mpi_decomposition.fill_U_from_buffers(U, mpi_decomposition.down);

}

//Write U on disk through PDI
void write_solution_to_file(const  Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>& U_IO, const Kokkos::View<double**>& U, int& nwrite, double time)
{   
    // Copy the view to the host mirror
    // The deepcopy handles the transpose automatically if U and U'IO layouts are differents
    Kokkos::deep_copy(U_IO, U);

    // Expose the solution
    PDI_multi_expose("write_data",
                 "nwrite", &nwrite, PDI_OUT,
                 "time", &time, PDI_OUT,
                 "main_field", U_IO.data(), PDI_OUT,
                  NULL);
    
    //Increment writing counter
    nwrite=nwrite+1;
}

//Main loop
void heat_equation(int argc, char* argv[], const MPI_Comm main_comm, const PC_tree_t conf)
{   
    /// ---- Tunable parameters ----

    //Domain size, final time and diffusion coefficient
    const double Lx = 1.0;
    const double Ly = 1.0;
    const double Tend = 1;
    const double kappa = 0.1;

    //CFL number, number of cells per proc, mpi decomposition and max number of iter
    const double cfl = 0.9;

    const int nx = 128;
    const int ny = 128;
    Coordinates mpi_max_coords;
    mpi_max_coords.x=2;
    mpi_max_coords.y=2;

    const int nmax = 9999999;  

    /// ---- Untunable parameters ----

    //Number of cells in the whole domain
    const int Nx = nx*mpi_max_coords.x;
    const int Ny = ny*mpi_max_coords.y;

    //Cell size
    const double dx = Lx/Nx;
    const double dy = Ly/Ny;

    //time step calculated from the diffusion coefficient
    const double inv_dt_x = 1.0/(0.5*dx*dx/kappa);
    const double inv_dt_y = 1.0/(0.5*dy*dy/kappa);
    double dt = cfl/(inv_dt_x + inv_dt_y); //Not const for time adjustment, but constant CFL
    
    //predicted amount of steps to progess 10 percents
    const int freq_write = std::floor(Tend/(10*dt));

    //Size of the arrays
    const int size_x = nx + 2;
    const int size_y = ny + 2;

    //Policies for looping over the whole solution, skipping ghost cells
    const Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy({1,1},{nx+1, ny+1});

    //Compute memory required on host and device
    const int U_mem_size = size_x*size_y*sizeof(double);
    const int device_mem_size = 2*U_mem_size;
    const int host_mem_size = U_mem_size;

    // Get MPI info and compute rank info (2D index and neighbors, see mpi.hpp)
    int mpi_rank; MPI_Comm_rank(main_comm, &mpi_rank);
    int mpi_size; MPI_Comm_size(main_comm, &mpi_size);
    MPI_DECOMPOSITION mpi_decomposition(mpi_rank, mpi_size, mpi_max_coords, main_comm, nx, ny);

    //Send meta-data to PDI
    PDI_multi_expose("init_PDI",
                    "mpi_coords_x", &mpi_decomposition.coords.x, PDI_OUT,
                    "mpi_coords_y", &mpi_decomposition.coords.y, PDI_OUT,
                    "mpi_max_coords_x", &mpi_decomposition.max_coords.x, PDI_OUT,
                    "mpi_max_coords_y", &mpi_decomposition.max_coords.y, PDI_OUT,
                    "nx", &nx, PDI_OUT,
                    "ny", &ny, PDI_OUT,
                     NULL);

    //Print simulation information
    if (mpi_rank==0)
    {
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
    printf("Memory size of U, per proc: %f MB\n", U_mem_size / 1e6);
    printf("Allocation on device, per proc: %f MB\n", device_mem_size / 1e6);
    printf("Allocation on host, per proc: %f MB\n", host_mem_size / 1e6);
    printf("--------------------------------------------------\n");
    }

    // Print MPU info (see mpi_decomposition.hpp)
    Kokkos::fence();
    MPI_Barrier(mpi_decomposition.comm);
    mpi_decomposition.printDetails();
    Kokkos::fence();
    MPI_Barrier(mpi_decomposition.comm);

    //Allocate the arrays
    Kokkos::View<double**> U ("Solution U on device", size_x, size_y);
    Kokkos::View<double**> U_("Intermediate Solution U on device", size_x, size_y);
    
    // Declare mirror array of U on host
    auto U_host = Kokkos::create_mirror(U);

    //Host array with forced layout for compatibility with PDI
    //PDI can only write from host memory and assumes a right layout
    //U's default device layout may be left or right depending on the backend
    //U_host inherits the layout from U
    Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> U_IO ("I/O array for PDI", size_x, size_y);
    
    //Initialisation
    Initialisation(U, dx, dy, mpi_decomposition, policy);

    //Set time time and indexes to 0
    double time = 0.0;
    int nstep = 0;
    int nwrite = 0;

    //write initial condition
    write_solution_to_file(U_IO, U, nwrite, time);

    //Loop over the time steps
    Kokkos::Timer timer;
    while (time < Tend && nstep < nmax)
    {
        Kokkos::fence(); // Kokkos kernels are asynchronous. This waits until all kernels are done before continuing
        MPI_Barrier(mpi_decomposition.comm);

        // if t+dt > Tend, reduce dt to reach Tend exactly
        if (time + dt > Tend)
        {
            dt = Tend - time;
        }

        //Fill U's ghost cell with MPI periodic BC
        MPIBoundaryCondition(U, mpi_decomposition);

        //Copy U's values in U_ to prepare udpate
        Kokkos::deep_copy(U_, U);

        //Update U
        stencil_kernel(U, U_, dx, dy, dt, kappa, policy);

        //Update time and iteration number
        time = time + dt;
        nstep = nstep + 1;

        //Print progress information
        if (nstep % 500 == 0 && mpi_rank==0)
        {    
            printf("Time step: %d, Time: %f\n", nstep, time);
        }

        //Write solution every 10% of progression
        if (nstep % freq_write == 0)
        {
            write_solution_to_file(U_IO, U, nwrite, time);
        }

    }
    double elapsed_time = timer.seconds();

    //Write solution
    write_solution_to_file(U_IO, U, nwrite, time);

    //print info and reason for stopping
    if (mpi_rank==0)
    {
    if (time >= Tend)
    {
        printf("Simulation finished because time >= Tend\n");
    }
    else if (nstep >= nmax)
    {
        printf("Simulation finished because nstep >= nmax\n");
    }
    printf("Time step: %d, Time: %f\n", nstep, time);
    
    //print performances
    print_perf(elapsed_time, nx, ny, nstep);
    }
}