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
    // Kokkos::View<double*> send_buffer_x;
    // Kokkos::View<double*> recv_buffer_x;
    // Kokkos::View<double*, Kokkos::HostSpace> send_buffer_x_host;
    // Kokkos::View<double*, Kokkos::HostSpace> recv_buffer_x_host;

    // Kokkos::View<double*> send_buffer_y;
    // Kokkos::View<double*> recv_buffer_y;
    // Kokkos::View<double*, Kokkos::HostSpace> send_buffer_y_host;
    // Kokkos::View<double*, Kokkos::HostSpace> recv_buffer_y_host;

    MPI_DECOMPOSITION(int rank_, int size_, int max_coords_[2], MPI_Comm comm_, int nx_, int ny_) {

        //Get input values
        rank = rank_;
        size = size_;
        max_coords[0] = max_coords_[0];
        max_coords[1] = max_coords_[1];
        nx = nx_;
        ny = ny_;
        comm=comm_;

        // Allocate MPI buffers
        // send_buffer_x = Kokkos::View<double*>("send buffer for x BC", nx);
        // recv_buffer_x = Kokkos::View<double*>("recv buffer for x BC", nx);
        // send_buffer_x_host = Kokkos::create_mirror(send_buffer_x);
        // recv_buffer_x_host = Kokkos::create_mirror(recv_buffer_x);

        // send_buffer_y = Kokkos::View<double*>("send buffer for y BC", ny);
        // recv_buffer_y = Kokkos::View<double*>("recv buffer for y BC", ny);
        // send_buffer_y_host = Kokkos::create_mirror(send_buffer_y);
        // recv_buffer_y_host = Kokkos::create_mirror(recv_buffer_y);

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
        bottom_rank = coordsToRank(bottom);

        // Top neighbor
        int top[2] = {coords[0], (coords[1] + max_coords[1] - 1) % max_coords[1]};
        top_rank = coordsToRank(top);
    }

    //Print info to user
    void printDetails() {
        std::cout << "Rank: " << rank << ", Coordinates: (" << coords[0] << ", " << coords[1] << "), ";
        std::cout << "Neighbors - Right: " << right_rank << ", Left: " << left_rank << ", Bottom: " << bottom_rank << ", Top: " << top_rank << std::endl;
    }
};
