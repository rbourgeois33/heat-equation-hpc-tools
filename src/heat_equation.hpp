#include <fstream>
#include <iostream>

KOKKOS_INLINE_FUNCTION
double initial_condition(double x, double y)
{
    return std::sin(2*M_PI*(x))*std::sin(2*M_PI*(y));
}

//Initialize the view U with the initial condition function above
void Initialisation(Kokkos::View<double**>& U, double dx, double dy, const Kokkos::MDRangePolicy<Kokkos::Rank<2>> &policy)
{
    Kokkos::parallel_for
    (
        "Initialisation", 
        policy, 
        KOKKOS_LAMBDA ( const int i , const int j )
        {   
            double x_i=double(i)*dx;
            double y_j=double(j)*dy;
            U(i, j) = initial_condition(x_i, y_j) ;
        }
    );
}

//Apply a finite difference scheme on the view U, using the view U_
void stencil_kernel(Kokkos::View<double**>& U, Kokkos::View<double**>& U_, double dx, double dy, double dt, double kappa, const Kokkos::MDRangePolicy<Kokkos::Rank<2>> &policy)
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
void print_perf(double elapsed_time, int nx, int ny, int nstep)
{
    //Calculate performances
    int num_threads = Kokkos::DefaultExecutionSpace::concurrency();
    double MCellUpdate = double(nx)*double(ny)*double(nstep)/1e6;
    double MCellUpdatePerSec = MCellUpdate/elapsed_time;
    // Print performances
    printf("Elapsed time: %f\nNumber of threads: %d\nMCell update per second: %f\n\n", elapsed_time, num_threads, MCellUpdatePerSec);

}

//Apply boundary condition on view U
void BoundaryCondition(Kokkos::View<double**>& U, int nx, int ny)
{
    Kokkos::parallel_for
    (
        "BoundaryCondition_i", 
        Kokkos::RangePolicy<>(1, nx+1), 
        KOKKOS_LAMBDA ( const int i )
        {   
            U(i, 0   ) = U(i, ny);
            U(i, ny+1) = U(i, 1 );
        }
    );

    Kokkos::parallel_for
    (
        "BoundaryCondition_j", 
        Kokkos::RangePolicy<>(1, ny+1), 
        KOKKOS_LAMBDA ( const int j )
        {   
            U(0   , j) = U(ny  , j);
            U(nx+1, j) = U(1   , j);
        }
    );
}

//Write U on disk through PDI
void write_solution_to_file(Kokkos::View<double**>::HostMirror& U_host, Kokkos::View<double**>& U)
{   
    // Copy the view to the host mirror
    Kokkos::deep_copy(U_host, U);

    // Expose the solution
    PDI_multi_expose("write_data",
                 "main_field", U_host.data(), PDI_OUT,
                  NULL);

    // Print writing information
    std::cout << "\n ... " << "Solution written to file: " << " ...\n "<<std::endl;
}

//Main loop
void heat_equation(int argc, char* argv[], MPI_Comm main_comm, PC_tree_t conf)
{   

    // Get MPI info
    int mpi_rank; MPI_Comm_rank(main_comm, &mpi_rank);
	int mpi_size; MPI_Comm_size(main_comm, &mpi_size);

    //Domain size, final time and diffusion coefficient
    double Lx = 1.0;
    double Ly = 1.0;
    double Tend = 1.0;
    double kappa = 0.1;

    //CFL number, number of cells and max number of iterations
    double cfl = 0.9;

    int nx = 1024;
    int ny = 1024;

    int niter = 100;

    //Cell size
    double dx = Lx/nx;
    double dy = Ly/ny;

    //time step calculated from the diffusion coefficient
    double inv_dt_x = 1.0/(0.5*dx*dx/kappa);
    double inv_dt_y = 1.0/(0.5*dy*dy/kappa);
    double dt = cfl/(inv_dt_x + inv_dt_y);

    //Size of the arrays
    int ngc = 1; 
    int size_x = nx + 2*ngc;
    int size_y = ny + 2*ngc;

    //Policies for looping over the whole solution, skipping ghost cells
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy({ngc,ngc},{size_x-ngc, size_y-ngc});

    //Compute memory required on host and device
    int U_mem_size = size_x*size_y*sizeof(double);
    int device_mem_size = 2*U_mem_size;
    int host_mem_size = U_mem_size;

    //Initialize PDI meta-data
    PDI_multi_expose("init_PDI",
                    "mpi_rank", &mpi_rank, PDI_OUT,
                    "mpi_size", &mpi_size, PDI_OUT,
                    "nx", &nx, PDI_OUT,
                    "ny", &ny, PDI_OUT,
                     NULL);

    //Print simulation information
    if (mpi_rank==0)
    {
    Kokkos::print_configuration(std::cout);

    std::cout << "------------ Simulation Information --------------"  << std::endl;
    std::cout << "Diffusion coefficient: " << kappa << std::endl;
    std::cout << "Cell size: " << dx << std::endl;
    std::cout << "Time step: " << dt << std::endl;
    std::cout << "Number of cells: " << nx << " , "<< ny << std::endl;
    std::cout << "Number of iterations: " << niter << std::endl;
    std::cout << "Number of ghost cells: " << ngc << std::endl;
    std::cout << "CFL number: " << cfl << std::endl;
    std::cout << "Final time: " << Tend << std::endl;
    std::cout << "Domain size: " << Lx << " , " << Ly << std::endl;
    std::cout << "Memory size of U: " << U_mem_size/1e6 << " MB" << std::endl;
    std::cout << "Allocation on device: " << device_mem_size/1e6 << " MB" << std::endl;
    std::cout << "Allocation on host: " << host_mem_size/1e6 << " MB" << std::endl;
    std::cout << "--------------------------------------------------"  << std::endl;
    }
    
    //Allocate the arrays
    Kokkos::View<double**> U ("Solution U on device", size_x, size_y);
    Kokkos::View<double**> U_("Intermediate Solution U on device", size_x, size_y);
    
    // Declare mirror array of U on host
    auto U_host = Kokkos::create_mirror(U);

    //Initialisation
    Initialisation(U, dx, dy, policy);

    //Write solution
    write_solution_to_file(U_host,U);

    //Set time and n to 0 
    double t = 0.0;
    int nstep = 0;

    //Loop over the time steps
    Kokkos::Timer timer;
    while (t < Tend && nstep < niter)
    {
        // if t+dt > Tend, reduce dt to reach Tend exactly
        if (t + dt > Tend)
        {
            dt = Tend - t;
        }

        //Fill U's ghost cell with periodic BC
        BoundaryCondition(U, nx, ny);

        //Copy U's values in U_ to prepare udpate
        Kokkos::deep_copy(U_, U);

        //Update U
        stencil_kernel(U, U_, dx, dy, dt, kappa, policy);

        //Update time and iteration number
        t = t + dt;
        nstep = nstep + 1;

        //Print progress information
        if (nstep % 500 == 0)
        {
            printf("Time step: %d, Time: %f\n", nstep, t);
        }
        
    }
    double elapsed_time = timer.seconds();

    //print info and reason for stopping
    if (mpi_rank==0)
    {
    if (t >= Tend)
    {
        printf("Simulation finished because t >= Tend\n");
    }
    else if (nstep >= niter)
    {
        printf("Simulation finished because nstep >= niter\n");
    }
    printf("Time step: %d, Time: %f\n", nstep, t);
    
    //print performances
    print_perf(elapsed_time, nx, ny, nstep);
    }
}