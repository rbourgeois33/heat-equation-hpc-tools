#include "mpi_decomposition.hpp"

KOKKOS_INLINE_FUNCTION
double initial_condition(double x, double y)
{
    const double rmax=0.2;
    const double r = sqrt((x-0.5)*(x-0.5)+(y-0.5)*(y-0.5));
    return r<rmax ? 1:0;
}

//Initialize the view U with the initial condition function above
void Initialisation(Kokkos::View<double**>& U, const double dx, const double dy, const int nx, const int ny, const int mpi_coords[2], const Kokkos::MDRangePolicy<Kokkos::Rank<2>> &policy)
{
    Kokkos::parallel_for
    (
        "Initialisation", 
        policy, 
        KOKKOS_LAMBDA ( const int i , const int j )
        {   
            double x_i=(double(i + mpi_coords[0]*nx)-0.5)*dx;
            double y_j=(double(j + mpi_coords[1]*ny)-0.5)*dy;
            U(i, j) = initial_condition(x_i, y_j) ;
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
void BoundaryCondition(Kokkos::View<double**>& U, const int nx, const int ny, const MPI_DECOMPOSITION& mpi_decomposition)
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

    // y>0 send / receive

    // // Fill send buffer with top data and fence
    // Kokkos::parallel_for("Fill send buffer with top data",  
    // Kokkos::RangePolicy<>(1, nx+1), 
    // KOKKOS_LAMBDA ( const int i ){mpi_decomposition.send_buffer_x(i) = U(i, ny);});
    // Kokkos::fence();

    // //Fence, Copy send buffer to host and send
    // Kokkos::deep_copy(mpi_decomposition.send_buffer_x_host, mpi_decomposition.send_buffer_x);
    // MPI_Send(mpi_decomposition.send_buffer_x_host.data(), nx, MPI_DOUBLE, mpi_decomposition.rank_top, 0, mpi_decomposition.comm);

    // //Receive buffer and copy to device
    // MPI_Recv(mpi_decomposition.recv_buffer_x_host.data(), nx, MPI_DOUBLE, mpi_decomposition.rank_bottom, 0, mpi_decomposition.comm, MPI_STATUS_IGNORE);
    // Kokkos::deep_copy(mpi_decomposition.recv_buffer_x, mpi_decomposition.recv_buffer_x_host);

    // //Fill bot BC with buffer data
    // Kokkos::parallel_for("Fill recv bot BC with buffer data",  
    // Kokkos::RangePolicy<>(1, nx+1), 
    // KOKKOS_LAMBDA ( const int i ){U(i, 0) = mpi_decomposition.recv_buffer_x(i);});
    // Kokkos::fence();

    // // y<0 send receive

    // // Fill send buffer with bot data and fence
    // Kokkos::parallel_for("Fill send buffer with top data",  
    // Kokkos::RangePolicy<>(1, nx+1), 
    // KOKKOS_LAMBDA ( const int i ){mpi_decomposition.send_buffer_x(i) = U(i, 1);});
    // Kokkos::fence();

    // //Fence, Copy send buffer to host and send
    // Kokkos::deep_copy(mpi_decomposition.send_buffer_x_host, mpi_decomposition.send_buffer_x);
    // MPI_Send(mpi_decomposition.send_buffer_x_host.data(), nx, MPI_DOUBLE, mpi_decomposition.rank_bottom, 1, mpi_decomposition.comm);

    // //Receive buffer and copy to device
    // MPI_Recv(mpi_decomposition.recv_buffer_x_host.data(), nx, MPI_DOUBLE, mpi_decomposition.rank_top, 1, mpi_decomposition.comm, MPI_STATUS_IGNORE);
    // Kokkos::deep_copy(mpi_decomposition.recv_buffer_x, mpi_decomposition.recv_buffer_x_host);

    // //Fill bot BC with buffer data
    // Kokkos::parallel_for("Fill recv top BC with buffer data",  
    // Kokkos::RangePolicy<>(1, nx+1), 
    // KOKKOS_LAMBDA ( const int i ){U(i, ny) = mpi_decomposition.recv_buffer_x(i);});
    // Kokkos::fence();

}

//Write U on disk through PDI
void write_solution_to_file(const Kokkos::View<double**>::HostMirror& U_host, const Kokkos::View<double**>& U, int& nwrite, double time)
{   
    // Copy the view to the host mirror
    Kokkos::deep_copy(U_host, U);

    // Expose the solution
    PDI_multi_expose("write_data",
                 "nwrite", &nwrite, PDI_OUT,
                 "time", &time, PDI_OUT,
                 "main_field", U_host.data(), PDI_OUT,
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
    int mpi_max_coords[2]={3,3};

    const int nmax = 9999999;  

    /// ---- Untunable parameters ----

    //Number of cells in the whole domain
    const int Nx = nx*mpi_max_coords[0];
    const int Ny = ny*mpi_max_coords[1];

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
    const int ngc = 1; 
    const int size_x = nx + 2*ngc;
    const int size_y = ny + 2*ngc;

    //Policies for looping over the whole solution, skipping ghost cells
    const Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy({ngc,ngc},{size_x-ngc, size_y-ngc});

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
                    "mpi_coords", &mpi_decomposition.coords, PDI_OUT,
                    "mpi_max_coords", &mpi_decomposition.max_coords, PDI_OUT,
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
    std::cout << "Number of cells per proc: " << nx << " , "<< ny << std::endl;
    std::cout << "Total of cells per proc: " << Nx << " , "<< Ny << std::endl;
    std::cout << "Number of iterations: " << nmax << std::endl;
    std::cout << "Number of ghost cells: " << ngc << std::endl;
    std::cout << "CFL number: " << cfl << std::endl;
    std::cout << "Final time: " << Tend << std::endl;
    std::cout << "Domain size: " << Lx << " , " << Ly << std::endl;
    std::cout << "Memory size of U, per proc: " << U_mem_size/1e6 << " MB" << std::endl;
    std::cout << "Allocation on device, per proc: " << device_mem_size/1e6 << " MB" << std::endl;
    std::cout << "Allocation on host, per proc: " << host_mem_size/1e6 << " MB" << std::endl;
    std::cout << "--------------------------------------------------"  << std::endl;
    }

    // Print MPU info (see mpi.hpp)
    MPI_Barrier(mpi_decomposition.comm);
    mpi_decomposition.printDetails();
    MPI_Barrier(mpi_decomposition.comm);

    //Allocate the arrays
    Kokkos::View<double**> U ("Solution U on device", size_x, size_y);
    Kokkos::View<double**> U_("Intermediate Solution U on device", size_x, size_y);
    
    // Declare mirror array of U on host
    auto U_host = Kokkos::create_mirror(U);

    //Initialisation
    Initialisation(U, dx, dy, nx, ny, mpi_decomposition.coords, policy);

    //Set time time and indexes to 0
    double time = 0.0;
    int nstep = 0;
    int nwrite = 0;

    //write initial condition
    write_solution_to_file(U_host, U, nwrite, time);

    //Loop over the time steps
    Kokkos::Timer timer;
    while (time < Tend && nstep < nmax)
    {
        // if t+dt > Tend, reduce dt to reach Tend exactly
        if (time + dt > Tend)
        {
            dt = Tend - time;
        }

        //Fill U's ghost cell with periodic BC
        BoundaryCondition(U, nx, ny, mpi_decomposition);

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
            write_solution_to_file(U_host, U, nwrite, time);
        }

    }
    double elapsed_time = timer.seconds();

    //Write solution
    write_solution_to_file(U_host, U, nwrite, time);

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