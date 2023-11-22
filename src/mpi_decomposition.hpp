// Class that stores all the necessary local mpi info such as coordinates and neighbors

class MPI_DECOMPOSITION {

public:

    int rank; //mpi_rank
    int size; // mpi_size
    int max_coords[2]; // Array to store the maximum ranks in X and Y dimensions
    int nx, ny; //Size of the local domain
    MPI_Comm comm; // mpi communicator

    int coords[2]; // 2D oordinates corresponding to the rank
    int left_rank, right_rank, top_rank, bottom_rank; // mpi_rank of the neighbor

    // send / recv MPI buffers for BC on host and device, size unknown here
    Kokkos::View<double*> send_buffer_y;
    Kokkos::View<double*> recv_buffer_y;
    Kokkos::View<double*, Kokkos::HostSpace> send_buffer_y_host;
    Kokkos::View<double*, Kokkos::HostSpace> recv_buffer_y_host;

    Kokkos::View<double*> send_buffer_x;
    Kokkos::View<double*> recv_buffer_x;
    Kokkos::View<double*, Kokkos::HostSpace> send_buffer_x_host;
    Kokkos::View<double*, Kokkos::HostSpace> recv_buffer_x_host;

    // Indexes for readability
    int up = 0;
    int down = 1;

    MPI_DECOMPOSITION(int rank_, int size_, int max_coords_[2], MPI_Comm comm_, int nx_, int ny_) {

        //Get input values
        rank = rank_;
        size = size_;
        max_coords[0] = max_coords_[0];
        max_coords[1] = max_coords_[1];
        nx = nx_;
        ny = ny_;
        comm=comm_;

        //Allocate MPI buffers
        send_buffer_y = Kokkos::View<double*>("send buffer for y BC", nx);
        recv_buffer_y = Kokkos::View<double*>("recv buffer for y BC", nx);
        send_buffer_y_host = Kokkos::create_mirror(send_buffer_y);
        recv_buffer_y_host = Kokkos::create_mirror(recv_buffer_y);

        send_buffer_x = Kokkos::View<double*>("send buffer for x BC", ny);
        recv_buffer_x = Kokkos::View<double*>("recv buffer for x BC", ny);
        send_buffer_x_host = Kokkos::create_mirror(send_buffer_x);
        recv_buffer_x_host = Kokkos::create_mirror(recv_buffer_x);

        // Validate the configuration
        if (max_coords[0] * max_coords[1] != size) {
            throw std::invalid_argument("The product of max_coords dimensions does not match size.");
        }

        // Compute coordinates and neighbors
        computeCoords();
        computeNeighborRanks();
    }

    void computeCoords() {
        coords[0] = rank / max_coords[1];
        coords[1] = rank % max_coords[1];
    }

    int coordsToRank(const int coords[2]) {
        return coords[0] * max_coords[1] + coords[1];
    }

    void computeNeighborRanks() {
        // Right neighbor
        int right[2] = {(coords[0] + 1) % max_coords[0], coords[1]};
        right_rank = coordsToRank(right);

        // Left neighbor
        int left[2] = {(coords[0] + max_coords[0] - 1) % max_coords[0], coords[1]};
        left_rank = coordsToRank(left);

        // Bottom neighbor
        int bottom[2] = {coords[0], (coords[1] + 1) % max_coords[1]};
        top_rank = coordsToRank(bottom);

        // Top neighbor
        int top[2] = {coords[0], (coords[1] + max_coords[1] - 1) % max_coords[1]};
        bottom_rank = coordsToRank(top);
    }

    //Print info to user
    void printDetails() {
    printf("Rank: %d, Coordinates: (%d, %d), ", rank, coords[0], coords[1]);
    printf("Neighbors - Right: %d, Left: %d, Bottom: %d, Top: %d\n", right_rank, left_rank, bottom_rank, top_rank);
    }

    void send_recv_buffers(int direction)
    {
        int send_rank_x = direction==up ? top_rank : bottom_rank;
        int recv_rank_x = direction==up ? bottom_rank : top_rank;

        int send_rank_y = direction==up ? right_rank : left_rank;
        int recv_rank_y = direction==up ? left_rank : right_rank;

        int tag_x = direction==up ? 0 : 1;
        int tag_y = direction==up ? 2 : 3;
        
        //Copy send to host
        Kokkos::deep_copy(send_buffer_y_host, send_buffer_y);
        Kokkos::deep_copy(send_buffer_x_host, send_buffer_x);

        //Send
        MPI_Send(send_buffer_y_host.data(), nx, MPI_DOUBLE, send_rank_x, tag_x, comm);
        MPI_Send(send_buffer_x_host.data(), ny, MPI_DOUBLE, send_rank_y, tag_y, comm);

        //Receive
        MPI_Recv(recv_buffer_y_host.data(), nx, MPI_DOUBLE, recv_rank_x, tag_x, comm, MPI_STATUS_IGNORE);
        MPI_Recv(recv_buffer_x_host.data(), ny, MPI_DOUBLE, recv_rank_y, tag_y, comm, MPI_STATUS_IGNORE);

        //Copy recv to device
        Kokkos::deep_copy(recv_buffer_y, recv_buffer_y_host);
        Kokkos::deep_copy(recv_buffer_x, recv_buffer_x_host);
    }

    void fill_buffers_from_U(Kokkos::View<double**>& U, int direction)
    {   
    // Deduce the y index of the extracted values from U 
    int n_extract_y = direction==up ? ny : 1;
    int n_extract_x = direction==up ? nx : 1;

    // Fill buffer from U
    Kokkos::parallel_for("fill buffer x from U",  
    Kokkos::RangePolicy<>(0, nx), 
    KOKKOS_CLASS_LAMBDA ( const int i ){send_buffer_y(i) = U(i+1, n_extract_y);}); //When using KOKKOS_LAMBDA in a class, one has to specify it using KOKKOS_CLASS_LAMBDA

    Kokkos::parallel_for("fill buffer y from U",  
    Kokkos::RangePolicy<>(0, ny), 
    KOKKOS_CLASS_LAMBDA ( const int j ){send_buffer_x(j) = U(n_extract_x, j+1);});

    Kokkos::fence(); //ensure the filling is done before going on
    MPI_Barrier(comm);
    }

    void fill_U_from_buffers(Kokkos::View<double**>& U, int direction)
    {   
    // Deduce the y index of the filled values in U
    int n_fill_y = direction==up ? 0 : ny+1;
    int n_fill_x = direction==up ? 0 : nx+1;

    // Fill U from buffer
    Kokkos::parallel_for("fill U from x buffer",  
    Kokkos::RangePolicy<>(0, nx), 
    KOKKOS_CLASS_LAMBDA ( const int i ){U(i+1, n_fill_y) = recv_buffer_y(i);});

    Kokkos::parallel_for("fill U from y buffer",  
    Kokkos::RangePolicy<>(0, ny), 
    KOKKOS_CLASS_LAMBDA ( const int j ){U(n_fill_x, j+1) = recv_buffer_x(j);});

    Kokkos::fence(); //ensure the filling is done before going on
    MPI_Barrier(comm);
    }
};
