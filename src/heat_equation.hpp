#include <fstream>
#include <iostream>

KOKKOS_INLINE_FUNCTION
double initial_condition(double x, double y)
{
    return std::sin(2*M_PI*(x))*std::sin(2*M_PI*(y));
}

void Initialisation(Kokkos::View<double**>& U, int i0, int j0, int iend, int jend, double dx, double dy)
{
    Kokkos::parallel_for
    (
        "Initialisation", 
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({i0, j0}, {iend+1, jend+1}), 
        KOKKOS_LAMBDA ( const int i , const int j )
        {   
            double x_i=double(i)*dx;
            double y_j=double(j)*dy;
            U(i, j) = initial_condition(x_i, y_j) ;
        }
    );
}

void copy(Kokkos::View<double**>& U, Kokkos::View<double**>& U_, int i0, int j0, int iend, int jend)
{
    Kokkos::parallel_for
    (
        "Copy", 
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({i0, j0}, {iend+1, jend+1}), 
        KOKKOS_LAMBDA ( const int i , const int j )
        {   
            U_(i, j) = U(i, j) ;
        }
    );
}

void stencil_kernel(Kokkos::View<double**>& U, Kokkos::View<double**>& U_, int i0, int j0, int iend, int jend, double dx, double dy, double dt, double kappa)
{
    Kokkos::parallel_for
    (
        "heat_equation_kernel", 
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({i0, j0}, {iend+1, jend+1}), 
        KOKKOS_LAMBDA ( const int i , const int j )
        {   
            U(i, j) = U_(i, j) + dt*kappa*( (U_(i+1, j) - 2.0*U_(i, j) + U_(i-1, j))/(dx*dx) + (U_(i, j+1) - 2.0*U_(i, j) + U_(i, j-1))/(dy*dy) );
        }
    );   
}

void print_perf(double elapsed_time, int nx, int ny, int nstep)
{
    //Calculate performances
    int num_threads = Kokkos::DefaultExecutionSpace::concurrency();
    double MCellUpdate = double(nx)*double(ny)*double(nstep)/1e6;
    double MCellUpdatePerSec = MCellUpdate/elapsed_time;
    // Print performances
    printf("Elapsed time: %f\nNumber of threads: %d\nMCell update per second: %f\n\n", elapsed_time, num_threads, MCellUpdatePerSec);

}

void BoundaryCondition(Kokkos::View<double**>& U, int i0, int j0, int iend, int jend)
{
    Kokkos::parallel_for
    (
        "BoundaryCondition_i", 
        Kokkos::RangePolicy<>(i0, iend+1), 
        KOKKOS_LAMBDA ( const int i )
        {   
            U(i, j0  -1) = U(i, j0  );
            U(i, jend+1) = U(i, jend);
        }
    );

    Kokkos::parallel_for
    (
        "BoundaryCondition_j", 
        Kokkos::RangePolicy<>(j0, jend+1), 
        KOKKOS_LAMBDA ( const int j )
        {   
            U(i0  -1, j) = U(i0  , j);
            U(iend+1, j) = U(iend, j);
        }
    );
}

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
    int start_x = 0;
    int start_y = 0;
    int end_x = size_x-1;
    int end_y = size_y-1;

    //Start and end of the solution
    int start_sol_x = ngc; 
    int start_sol_y = ngc;
    int end_sol_x = size_x - ngc - 1;
    int end_sol_y = size_y - ngc - 1;

    //Compute memory required on host and device
    int U_mem_size = size_x*size_y*sizeof(double);
    int device_mem_size = 2*U_mem_size;
    int host_mem_size = U_mem_size;

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
    Initialisation(U, start_x, start_y, end_x, end_y, dx, dy);

    //Send to PDI
    int dsize[2]={size_x, size_y};

    PDI_multi_expose("init_PDI",
                    "mpi_rank", &mpi_rank, PDI_OUT,
                    "mpi_size", &mpi_size, PDI_OUT,
                    "dsize", &dsize, PDI_OUT,
                     NULL);

    Kokkos::deep_copy(U_host,U);
    
    PDI_multi_expose("write_data",
                     "main_field", U_host.data(), PDI_OUT,
                      NULL);

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

        BoundaryCondition(U, start_sol_x, start_sol_y, end_sol_x, end_sol_y);
        copy(U, U_, start_x, start_y, end_x, end_y);
        stencil_kernel(U, U_, start_sol_x, start_sol_y, end_sol_x, end_sol_y, dx, dy, dt, kappa);

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